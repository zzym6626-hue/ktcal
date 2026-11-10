"""Frozen 3ET+2025 final-holdout evaluation; separate from validation tuning."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .combined_3et import (
    SCALE, _aggregate_tracker_stats, _sha256,
    moving_average_sequence_predictions,
    load_matcher_checkpoint, load_stnet_checkpoint,
)
from .evaluation_protocol import (
    hard_argmax_decode, official_postprocess, soft_expectation_decode,
    summarize_pixel_errors, trim_record_chunks,
)
from .tracker import PaperTracker
from scripts.evaluate_official_3et_protocol import (
    OFFICIAL_RECORDING_LENGTHS, _confidence, load_event_statistics,
    validate_official_h5,
)

_OFFLINE_FILTERS = ('none', 'ma3', 'ma5', 'ma7', 'ma9', 'ma11', 'ma13', 'ma15')


@torch.inference_mode()
def evaluate_ktcal_official_holdout(
    h5_path, stnet_checkpoint, matcher_checkpoint, output, *,
    events=None, stnet_reference=None, batch_size=4, threads=4, device='cuda',
    recording_lengths=OFFICIAL_RECORDING_LENGTHS, chunk_size=200,
    stnet_decode='hard', offline_filter='none',
):
    """Score frozen validation-selected KTCAL without altering validation guards.

    The optional recording length/chunk size override is for synthetic tests only;
    the CLI deliberately exposes only the released official layout.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if batch_size < 1 or threads < 1:
        raise ValueError('Batch size and threads must be positive')
    if stnet_decode not in ('hard', 'soft'):
        raise ValueError("stnet_decode must be 'hard' or 'soft'")
    if offline_filter not in _OFFLINE_FILTERS:
        raise ValueError(
            "offline_filter must be 'none' or "
            "'ma3/ma5/ma7/ma9/ma11/ma13/ma15'")
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    h5_path = Path(h5_path).resolve()
    official_layout = tuple(recording_lengths) == OFFICIAL_RECORDING_LENGTHS and chunk_size == 200
    if official_layout:
        shapes = validate_official_h5(h5_path)
    else:
        with h5py.File(h5_path, 'r') as handle:
            label_key = 'label' if 'label' in handle else 'labels'
            shapes = {'frames_shape': tuple(handle['frames'].shape),
                      'labels_shape': tuple(handle[label_key].shape),
                      'label_key': label_key}
        if shapes['frames_shape'][1] != chunk_size:
            raise ValueError('Synthetic chunk size must match HDF5 sequence length')
    torch.set_num_threads(threads)
    resolved_device = torch.device(device)
    stnet, stnet_config, divisor = load_stnet_checkpoint(stnet_checkpoint, resolved_device)
    matcher, matcher_config = load_matcher_checkpoint(matcher_checkpoint, resolved_device)

    chunks = {'stnet_only': [], 'ktcal_online': [], 'ktcal_retrospective': []}
    truth_chunks, confidence_chunks, stats_rows = [], [], []
    with h5py.File(h5_path, 'r') as handle:
        frames = handle['frames']
        labels = handle[shapes['label_key']]
        for start in range(0, shapes['frames_shape'][0], batch_size):
            stop = min(start + batch_size, shapes['frames_shape'][0])
            raw = torch.from_numpy(frames[start:stop].astype(np.float32))
            inputs = raw.to(resolved_device)
            if divisor != 1.:
                inputs = inputs / divisor
            logits_x, logits_y = stnet(inputs)
            decoder = (hard_argmax_decode if stnet_decode == 'hard'
                       else soft_expectation_decode)
            global_norm = decoder(logits_x, logits_y)
            global_px = global_norm.astype(np.float32) * SCALE
            chunks['stnet_only'].append(global_norm)
            confidence_chunks.append(_confidence(logits_x, logits_y).cpu().numpy())
            truth_chunks.append(labels[start:stop, :, :2])
            for index in range(stop - start):
                tracker = PaperTracker(
                    stnet, matcher, device=resolved_device, capture='fine',
                    subpixel=True, fusion_alpha=.75,
                    max_prediction_variance_px2=10.,
                    max_consecutive_prediction_skips=1,
                )
                online_px, retrospective_px, stats = tracker.track(
                    raw[index], global_predictions=global_px[index],
                )
                chunks['ktcal_online'].append(
                    online_px.astype(np.float32) / SCALE)
                chunks['ktcal_retrospective'].append(
                    retrospective_px.astype(np.float32) / SCALE)
                stats_rows.append(stats)
            print(f'Processed {stop}/{shapes["frames_shape"][0]} chunks', flush=True)

    truth = trim_record_chunks(np.concatenate(truth_chunks), recording_lengths,
                               chunk_size=chunk_size)
    predictions = {
        name: trim_record_chunks(
            np.concatenate(values) if name == 'stnet_only' else np.stack(values),
            recording_lengths,
                                 chunk_size=chunk_size)
        for name, values in chunks.items()
    }
    if offline_filter != 'none':
        window = int(offline_filter[2:])
        lengths = [int(length) for length in recording_lengths]
        boundaries = np.cumsum(lengths)[:-1]
        targets = {'stnet_only': 'stnet_offline', 'ktcal_online': 'ktcal_offline'}
        for source, target in targets.items():
            sequences = np.split(predictions[source], boundaries)
            predictions[target] = np.concatenate(
                moving_average_sequence_predictions(sequences, window))
    valid_frames = sum(recording_lengths)
    if len(truth) != valid_frames or any(len(p) != valid_frames for p in predictions.values()):
        raise ValueError('Official recording trim did not produce expected valid frames')
    metrics = {name: summarize_pixel_errors(pred, truth)
               for name, pred in predictions.items()}

    h5_sha256 = _sha256(h5_path)
    stnet_sha256 = _sha256(stnet_checkpoint)
    if stnet_reference:
        baseline = json.loads(Path(stnet_reference).read_text())
        expected = baseline['raw_metrics']
        if (baseline['h5_sha256'] != h5_sha256
                or baseline['checkpoint_sha256'] != stnet_sha256
                or expected['frames'] != valid_frames
                or any(abs(metrics['stnet_only'][key] - expected[key]) > 1e-5
                       for key in ('p3', 'mean_error_px'))):
            raise ValueError('STNet-only predictions do not match official reference')

    post_metrics = None
    events_sha256 = None
    if events is not None:
        events_data = load_event_statistics(events, expected_frames=valid_frames)
        confidences = trim_record_chunks(np.concatenate(confidence_chunks),
                                         recording_lengths, chunk_size=chunk_size)
        post_metrics = {
            name: summarize_pixel_errors(
                official_postprocess(
                    pred, confidence=confidences, num_events_y=events_data[:, 1],
                    recording_lengths=recording_lengths,
                ), truth,
            )
            for name, pred in predictions.items()
        }
        events_sha256 = _sha256(events)
    processed_frames = shapes['frames_shape'][0] * chunk_size
    report = {
        'schema': 1,
        'protocol': '3ET+2025 official test_200_200 full-200-frame bidirectional STNet + KTCAL',
        'h5': str(h5_path),
        'h5_sha256': h5_sha256,
        'frames_shape': list(shapes['frames_shape']),
        'labels_shape': list(shapes['labels_shape']),
        'recording_lengths': list(recording_lengths),
        'valid_frames': valid_frames,
        'stnet_checkpoint': str(Path(stnet_checkpoint).resolve()),
        'stnet_checkpoint_sha256': stnet_sha256,
        'stnet_best_epoch': stnet_config.get('best_epoch'),
        'stnet_input_divisor': divisor,
        'stnet_decode': stnet_decode,
        'stnet_readout': stnet_config.get('readout', 'last'),
        'offline_filter': offline_filter,
        'matcher_checkpoint': str(Path(matcher_checkpoint).resolve()),
        'matcher_checkpoint_sha256': _sha256(matcher_checkpoint),
        'matcher_selected_epoch': matcher_config.get('selected_epoch'),
        'matching': {
            'capture': 'fine', 'search_size': [35, 30],
            'locator': 'parabolic_subpixel', 'fusion_alpha': .75,
            'max_prediction_variance_px2': 10.,
            'max_consecutive_prediction_skips': 1,
        },
        'metrics': metrics,
        'post_metrics': post_metrics,
        'events_sha256': events_sha256,
        'tracker_stats': _aggregate_tracker_stats(stats_rows, processed_frames),
        'latency_comparable': False,
        'final_holdout_used': official_layout,
    }
    report['tracker_stats']['global_prediction_source'] = (
        'precomputed_full_200_frame_bidirectional_stnet')
    report['tracker_stats']['scored_frames'] = valid_frames
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
