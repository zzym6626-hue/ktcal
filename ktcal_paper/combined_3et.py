"""Validation-only composition of 3ET STNet predictions and KTCAL tracking."""

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from models.STNet import STNet

from .matching import PaperMatcher
from .evaluation_protocol import (
    hard_argmax_decode, soft_expectation_decode, summarize_pixel_errors,
)
from .tracker import PaperTracker


SCALE = np.array([80.0, 60.0], dtype=np.float32)
FINAL_HOLDOUT_SHAPE = (50, 200, 2, 60, 80)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_validation_h5(path):
    """Validate a released flat validation HDF5 without reading frame data."""
    path = Path(path).resolve()
    if path.name.lower() == 'test_200_200.h5':
        raise ValueError('Refusing to read the final holdout')
    if not path.is_file():
        raise FileNotFoundError(f'Validation HDF5 not found: {path}')
    with h5py.File(path, 'r') as handle:
        label_key = 'label' if 'label' in handle else 'labels'
        if 'frames' not in handle or label_key not in handle:
            raise ValueError('Validation HDF5 requires frames and label datasets')
        frames_shape = tuple(handle['frames'].shape)
        labels_shape = tuple(handle[label_key].shape)
    if frames_shape == FINAL_HOLDOUT_SHAPE:
        raise ValueError('Refusing to read the final holdout by official shape')
    if (len(frames_shape) != 5 or frames_shape[2:] != (2, 60, 80)
            or frames_shape[0] < 1 or frames_shape[1] < 1):
        raise ValueError(
            'Validation frames must have shape [N,T,2,60,80], '
            f'got {frames_shape}')
    if (len(labels_shape) != 3 or labels_shape[:2] != frames_shape[:2]
            or labels_shape[-1] < 2):
        raise ValueError(
            'Validation labels must have shape [N,T,at least 2] and match frames')
    return path, label_key, frames_shape, labels_shape


def load_stnet_checkpoint(path, device):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'STNet checkpoint not found: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError('STNet checkpoint must contain model_state_dict')
    config = dict(checkpoint.get('config', {}))
    if 'best_epoch' in checkpoint:
        config['best_epoch'] = checkpoint['best_epoch']
    args = SimpleNamespace(
        stnet_block3_kernel=int(config.get('block3_kernel', 5)),
        stnet_unidirectional=bool(config.get('unidirectional', False)),
        stnet_decode='argmax',
        stnet_readout=config.get('readout', 'last'),
        stnet_temporal_spatial_kernel=int(
            config.get('temporal_spatial_kernel', 1)),
    )
    model = STNet(args)
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as error:
        raise ValueError(f'Incompatible STNet checkpoint: {error}') from error
    divisor = float(config.get('input_divisor', 1.0))
    if divisor <= 0:
        raise ValueError('STNet input divisor must be positive')
    return model.to(device).eval(), config, divisor


def load_matcher_checkpoint(path, device):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Matcher checkpoint not found: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'matcher_state' not in checkpoint:
        raise ValueError('Matcher checkpoint must contain matcher_state')
    config = dict(checkpoint.get('config', {}))
    finetune = checkpoint.get('matcher_finetune', {})
    if isinstance(finetune, dict) and 'selected_epoch' in finetune:
        config['selected_epoch'] = finetune['selected_epoch']
    matcher = PaperMatcher(
        int(config.get('ncc_channels', 16)),
        float(config.get('temperature', .1)),
    )
    try:
        matcher.load_state_dict(checkpoint['matcher_state'], strict=True)
    except RuntimeError as error:
        raise ValueError(f'Incompatible matcher checkpoint: {error}') from error
    return matcher.to(device).eval(), config


class CausalLastStepSTNet(torch.nn.Module):
    """Adapt SimDR logits to on-demand, causal 16-frame tracker coordinates."""

    def __init__(self, model, divisor=1.0):
        super().__init__()
        if divisor <= 0:
            raise ValueError('STNet input divisor must be positive')
        self.model = model
        self.divisor = float(divisor)

    def forward(self, frames):
        if frames.ndim != 5 or frames.shape[1] != 16:
            raise ValueError('Expected 16 historical frames per current prediction')
        logits_x, logits_y = self.model(frames.float() / self.divisor)
        return torch.stack((
            logits_x[:, -1].argmax(-1).to(logits_x.dtype) / logits_x.shape[-1],
            logits_y[:, -1].argmax(-1).to(logits_y.dtype) / logits_y.shape[-1],
        ), dim=-1)


@torch.inference_mode()
def precompute_global_predictions(model, frames, device, input_divisor=1.0,
                                    decode='hard'):
    """Return STNet coordinates in pixel units for `[B,T,...]`.

    ``decode='soft'`` uses softmax expectation instead of hard argmax.
    """
    if input_divisor <= 0:
        raise ValueError('STNet input divisor must be positive')
    if decode not in ('hard', 'soft'):
        raise ValueError("STNet decode must be 'hard' or 'soft'")
    inputs = frames.to(device=device, dtype=torch.float32)
    if input_divisor != 1.0:
        inputs = inputs / input_divisor
    logits_x, logits_y = model(inputs)
    decoder = hard_argmax_decode if decode == 'hard' else soft_expectation_decode
    return decoder(logits_x, logits_y).astype(np.float32) * SCALE


def summarize_sequence_predictions(predictions, labels):
    if not predictions or len(predictions) != len(labels):
        raise ValueError('Prediction and label sequence lists must be nonempty and match')
    for predicted, expected in zip(predictions, labels):
        if np.asarray(predicted).shape != np.asarray(expected).shape:
            raise ValueError('Each prediction sequence must match its labels')
    return summarize_pixel_errors(
        np.concatenate(predictions, axis=0),
        np.concatenate(labels, axis=0),
    )


def moving_average_sequence_predictions(sequences, window):
    """Apply an edge-padded moving average to each prediction sequence."""
    if not sequences:
        raise ValueError('Prediction sequence list must be nonempty')
    if window < 1 or window % 2 == 0:
        raise ValueError('Moving-average window must be a positive odd integer')
    kernel = np.ones(window, dtype=np.float64) / window
    padding = window // 2
    result = []
    for sequence in sequences:
        values = np.asarray(sequence, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError('Each prediction sequence must have shape [T, 2]')
        padded = np.pad(values, ((padding, padding), (0, 0)), mode='edge')
        filtered = np.stack([
            np.convolve(padded[:, axis], kernel, mode='valid')
            for axis in range(values.shape[1])
        ], axis=1)
        result.append(filtered.astype(np.float32))
    return result


def _aggregate_tracker_stats(rows, frames):
    count_keys = (
        'tier1', 'tier2', 'tier3', 'stnet_calls', 'online_sync_resets',
        'async_applied', 'async_discarded', 'border_peaks', 'psr_rejections',
        'fusion_calls', 'replayed_frames', 'skipped_matches',
    )
    result = {key: int(sum(row.get(key, 0) for row in rows)) for key in count_keys}
    result['frames'] = int(frames)
    result['trigger_ratio'] = result['stnet_calls'] / frames
    result['global_prediction_source'] = 'precomputed_full_100_frame_bidirectional_stnet'
    return result


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_combined_3et(h5_path, stnet_checkpoint, matcher_checkpoint, output,
                          *, batch_size=4, threads=4, device='cuda',
                          fusion_alpha=None, max_prediction_variance_px2=None,
                          max_consecutive_prediction_skips=1, stnet_decode='hard',
                          offline_filter='none'):
    if batch_size < 1 or threads < 1:
        raise ValueError('Batch size and threads must be positive')
    if stnet_decode not in ('hard', 'soft'):
        raise ValueError("stnet_decode must be 'hard' or 'soft'")
    if offline_filter not in ('none', 'ma3', 'ma5', 'ma7', 'ma9',
                              'ma11', 'ma13', 'ma15'):
        raise ValueError(
            "offline_filter must be 'none' or 'ma3/ma5/ma7/ma9/ma11/ma13/ma15'")
    h5_path, label_key, frames_shape, labels_shape = validate_validation_h5(
        h5_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(threads)
    resolved_device = torch.device(device)
    stnet, stnet_config, input_divisor = load_stnet_checkpoint(
        stnet_checkpoint, resolved_device)
    matcher, matcher_config = load_matcher_checkpoint(
        matcher_checkpoint, resolved_device)

    stnet_sequences = []
    online_sequences = []
    corrected_sequences = []
    label_sequences = []
    tracker_stats = []
    stnet_ms = 0.
    tracking_ms = 0.
    stnet_filter_ms = 0.
    ktcal_filter_ms = 0.
    with h5py.File(h5_path, 'r') as handle:
        frames = handle['frames']
        labels = handle[label_key]
        for start in range(0, frames_shape[0], batch_size):
            stop = min(start + batch_size, frames_shape[0])
            raw = frames[start:stop].astype(np.float32)
            frame_batch = torch.from_numpy(raw)
            label_batch = labels[start:stop, :, :2].astype(np.float32)
            if start == 0:
                # Warm both paths on the same input without including setup in throughput.
                warm_px = precompute_global_predictions(
                    stnet, frame_batch, resolved_device, input_divisor,
                    decode=stnet_decode)
                PaperTracker(
                    stnet, matcher, device=resolved_device, capture='fine',
                    subpixel=True, fusion_alpha=fusion_alpha,
                    max_prediction_variance_px2=max_prediction_variance_px2,
                    max_consecutive_prediction_skips=max_consecutive_prediction_skips,
                ).track(frame_batch[0], global_predictions=warm_px[0])
            _synchronize(resolved_device)
            started = time.perf_counter()
            global_px = precompute_global_predictions(
                stnet, frame_batch, resolved_device, input_divisor,
                decode=stnet_decode)
            _synchronize(resolved_device)
            stnet_ms += (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            for index in range(stop - start):
                tracker = PaperTracker(
                    stnet, matcher, device=resolved_device, capture='fine',
                    subpixel=True, fusion_alpha=fusion_alpha,
                    max_prediction_variance_px2=max_prediction_variance_px2,
                    max_consecutive_prediction_skips=max_consecutive_prediction_skips)
                online_px, corrected_px, stats = tracker.track(
                    frame_batch[index], global_predictions=global_px[index])
                stnet_sequences.append(global_px[index] / SCALE)
                online_sequences.append(online_px.astype(np.float32) / SCALE)
                corrected_sequences.append(
                    corrected_px.astype(np.float32) / SCALE)
                label_sequences.append(label_batch[index])
                tracker_stats.append(stats)
            _synchronize(resolved_device)
            tracking_ms += (time.perf_counter() - started) * 1000

    metrics = {
        'stnet_only': summarize_sequence_predictions(
            stnet_sequences, label_sequences),
        'ktcal_online': summarize_sequence_predictions(
            online_sequences, label_sequences),
        'ktcal_retrospective': summarize_sequence_predictions(
            corrected_sequences, label_sequences),
    }
    if offline_filter != 'none':
        window = int(offline_filter[2:])
        started = time.perf_counter()
        smoothed_stnet = moving_average_sequence_predictions(
            stnet_sequences, window)
        stnet_filter_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        smoothed_ktcal = moving_average_sequence_predictions(
            online_sequences, window)
        ktcal_filter_ms = (time.perf_counter() - started) * 1000
        metrics['stnet_offline'] = summarize_sequence_predictions(
            smoothed_stnet, label_sequences)
        metrics['ktcal_offline'] = summarize_sequence_predictions(
            smoothed_ktcal, label_sequences)
    total_frames = frames_shape[0] * frames_shape[1]
    ms_per_frame = {
        'stnet_only': stnet_ms / total_frames,
        'ktcal_online': (stnet_ms + tracking_ms) / total_frames,
        'ktcal_retrospective': (stnet_ms + tracking_ms) / total_frames,
    }
    if offline_filter != 'none':
        ms_per_frame['stnet_offline'] = (
            stnet_ms + stnet_filter_ms) / total_frames
        ms_per_frame['ktcal_offline'] = (
            stnet_ms + tracking_ms + ktcal_filter_ms) / total_frames
    report = {
        'schema': 1,
        'protocol': '3ET test_aug full-100-frame bidirectional STNet + KTCAL',
        'h5': str(h5_path),
        'h5_sha256': _sha256(h5_path),
        'frames_shape': list(frames_shape),
        'labels_shape': list(labels_shape),
        'stnet_checkpoint': str(Path(stnet_checkpoint).resolve()),
        'stnet_checkpoint_sha256': _sha256(stnet_checkpoint),
        'matcher_checkpoint': str(Path(matcher_checkpoint).resolve()),
        'matcher_checkpoint_sha256': _sha256(matcher_checkpoint),
        'stnet_best_epoch': stnet_config.get('best_epoch'),
        'stnet_input_divisor': input_divisor,
        'matcher_selected_epoch': matcher_config.get('selected_epoch'),
        'stnet_decode': stnet_decode,
        'offline_filter': offline_filter,
        'matching': {'capture': 'fine', 'search_size': [35, 30],
                     'locator': 'parabolic_subpixel',
                     'fusion_alpha': fusion_alpha,
                     'max_prediction_variance_px2': max_prediction_variance_px2,
                     'max_consecutive_prediction_skips': max_consecutive_prediction_skips},
        'metrics': metrics,
        'latency': {
            'mode': 'offline_batch_throughput',
            'frames': total_frames,
            'device': str(resolved_device),
            'batch_size': batch_size,
            'threads': threads,
            'warmup': 'first_batch_stnet_and_first_sequence_tracker',
            'scope': 'model forward, transfer, decode, tracking, optional filtering; '
                     'excludes HDF5 reads, model loading, metrics and report I/O',
            'stnet_total_ms': stnet_ms,
            'ktcal_tracking_total_ms': tracking_ms,
            'stnet_filter_total_ms': stnet_filter_ms,
            'ktcal_filter_total_ms': ktcal_filter_ms,
            'ms_per_frame': ms_per_frame,
        },
        'tracker_stats': _aggregate_tracker_stats(
            tracker_stats, total_frames),
        'latency_comparable': False,
        'final_holdout_used': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


@torch.inference_mode()
def evaluate_online_causal_3et(h5_path, stnet_checkpoint, matcher_checkpoint,
                               output, *, threads=4, device='cuda',
                               max_prediction_variance_px2=10.,
                               max_consecutive_prediction_skips=1):
    """Actual on-demand online tracker; no global or future-frame predictions."""
    if threads < 1:
        raise ValueError('Threads must be positive')
    h5_path, label_key, frames_shape, labels_shape = validate_validation_h5(
        h5_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(threads)
    resolved_device = torch.device(device)
    stnet, stnet_config, divisor = load_stnet_checkpoint(
        stnet_checkpoint, resolved_device)
    matcher, matcher_config = load_matcher_checkpoint(
        matcher_checkpoint, resolved_device)
    model = CausalLastStepSTNet(stnet, divisor).eval()
    predictions, labels, trackers = [], [], []
    latencies, processing, wall = [], [], []
    with h5py.File(h5_path, 'r') as handle:
        for sequence in range(frames_shape[0]):
            frames = torch.from_numpy(
                handle['frames'][sequence].astype(np.float32))
            truth = handle[label_key][sequence, :, :2].astype(np.float32)
            tracker = PaperTracker(
                model, matcher, device=resolved_device, capture='fine',
                subpixel=True, fusion_alpha=None,
                max_prediction_variance_px2=max_prediction_variance_px2,
                max_consecutive_prediction_skips=max_consecutive_prediction_skips)
            if resolved_device.type == 'cuda':
                torch.cuda.synchronize(resolved_device)
            started = time.perf_counter()
            online_px, _, stats = tracker.track(frames)
            if resolved_device.type == 'cuda':
                torch.cuda.synchronize(resolved_device)
            wall.append((time.perf_counter() - started) * 1000)
            predictions.append(online_px.astype(np.float32) / SCALE)
            labels.append(truth)
            trackers.append(stats)
            latencies.extend(stats['frame_latency_ms'])
            processing.extend(stats['frame_processing_ms'])
    metrics = summarize_sequence_predictions(predictions, labels)
    tracker_stats = _aggregate_tracker_stats(trackers, metrics['frames'])
    tracker_stats['global_prediction_source'] = 'on_demand_causal_16_step'
    report = {
        'schema': 1,
        'protocol': '3ET validation 16-frame causal history on-demand STNet + KTCAL',
        'h5': str(h5_path), 'h5_sha256': _sha256(h5_path),
        'frames_shape': list(frames_shape),
        'labels_shape': list(labels_shape),
        'stnet_checkpoint': str(Path(stnet_checkpoint).resolve()),
        'stnet_checkpoint_sha256': _sha256(stnet_checkpoint),
        'stnet_best_epoch': stnet_config.get('best_epoch'),
        'stnet_readout': stnet_config.get('readout', 'last'),
        'matcher_checkpoint': str(Path(matcher_checkpoint).resolve()),
        'matcher_checkpoint_sha256': _sha256(matcher_checkpoint),
        'matcher_selected_epoch': matcher_config.get('selected_epoch'),
        'matching': {
            'capture': 'fine', 'search_size': [35, 30],
            'locator': 'parabolic_subpixel', 'fusion_alpha': None,
            'max_prediction_variance_px2': max_prediction_variance_px2,
            'max_consecutive_prediction_skips': max_consecutive_prediction_skips,
        },
        'metrics': {'ktcal_online': metrics},
        'latency': {
            'average_frame_ms': float(np.mean(latencies)),
            'median_frame_ms': float(np.median(latencies)),
            'average_processing_ms': float(np.mean(processing)),
            'average_wall_ms_per_sequence': float(np.mean(wall)),
            'timing_mode': 'as_fast_as_possible',
        },
        'tracker_stats': tracker_stats,
        'final_holdout_used': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
