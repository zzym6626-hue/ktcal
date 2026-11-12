#!/usr/bin/env python3
"""Independently reproduce a 3ET report and verify raw low-tier STNet invariance."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from ktcal_paper.combined_3et import (
    _sha256, load_matcher_checkpoint, load_stnet_checkpoint, validate_validation_h5,
)
from ktcal_paper.offline_batch import OfflineBatchTracker
from ktcal_paper.offline_experiment import spatial_view, undo_view, validate_candidate_metrics


def filter_records(raw, kind):
    if kind == 'none':
        return np.asarray(raw).copy()
    window = int(kind[2:])
    if not kind.startswith('ma') or window < 1 or window % 2 == 0:
        raise ValueError('Independent verifier requires an odd moving-average filter')
    kernel = np.ones(window) / window
    return np.stack([
        np.column_stack([
            np.convolve(np.pad(record[:, axis] / scale,
                               (window // 2, window // 2), mode='edge'),
                        kernel, 'valid').astype(np.float32) * scale
            for axis, scale in enumerate((80., 60.))
        ]) for record in raw
    ])


def score_pixels(prediction, labels):
    if (prediction.shape != labels.shape or not np.isfinite(prediction).all()
            or not np.isfinite(labels).all() or prediction.size == 0):
        raise ValueError('Finite aligned nonempty pixel trajectories required')
    error = np.linalg.norm(prediction - labels, axis=-1)
    return dict(frames=int(error.size), mean_error_px=float(error.mean()),
                median_error_px=float(np.median(error)),
                **{'p%d' % threshold: float((error < threshold).mean())
                   for threshold in (1, 3, 5, 10, 15)})


def validate_reproduction(expected, actual, expected_counts, actual_counts):
    validate_candidate_metrics(expected, actual, actual_counts)
    if (expected_counts != actual_counts
            or sum(actual_counts['tier%d' % tier] for tier in (1, 2, 3)) != actual_counts['frames']
            or actual_counts['frames'] != actual['frames']):
        raise ValueError('Independent tracker count mismatch')


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--expected-report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    with Path(args.expected_report).open() as handle:
        expected = json.load(handle)
    config = expected['config']
    h5, label_key, shape, _ = validate_validation_h5(config['h5'])
    if h5.name != 'test_aug.h5' or shape != (93, 100, 2, 60, 80):
        raise ValueError('Full validation only')
    if _sha256(h5) != expected['h5_sha256']:
        raise ValueError('HDF5 SHA mismatch')
    if _sha256(config['stnet']) != expected['stnet_sha256']:
        raise ValueError('STNet SHA mismatch')
    if _sha256(config['matcher']) != expected['matcher_sha256']:
        raise ValueError('Matcher SHA mismatch')
    device = config['device']
    torch.set_num_threads(4)
    model, _, divisor = load_stnet_checkpoint(config['stnet'], device)
    matcher, _ = load_matcher_checkpoint(config['matcher'], device)
    with h5py.File(h5, 'r') as handle:
        frames = torch.from_numpy(handle['frames'][:].astype(np.float32))
        labels = handle[label_key][:, :, :2].astype(np.float64) * [80., 60.]
    views = [tuple(map(int, item.split(':'))) for item in config['views'].split(',')]
    predictions = [[] for _ in views]
    with torch.inference_mode():
        for begin in range(0, len(frames), config['batch_size']):
            source = frames[begin:begin + config['batch_size']].to(device) / divisor
            for index, view in enumerate(views):
                lx, ly = model(spatial_view(source, view))
                px = torch.softmax(lx / config['temperature'], dim=-1)
                py = torch.softmax(ly / config['temperature'], dim=-1)
                # Independent torch expectation in pixels, not the evaluator's NumPy decoder.
                coordinates = torch.stack((
                    (px * torch.arange(80, device=device)).sum(-1),
                    (py * torch.arange(60, device=device)).sum(-1)), -1).cpu().numpy()
                predictions[index].append(undo_view(coordinates, view))
    cache = np.stack([np.concatenate(values) for values in predictions])
    if config['ensemble'] == 'median':
        global_px = np.median(cache, axis=0)
    elif config['ensemble'] == 'mean':
        global_px = np.mean(cache, axis=0)
    elif config['ensemble'] == 'trimmed':
        trim = len(cache) // 5
        global_px = np.sort(cache, axis=0)[trim:len(cache) - trim].mean(axis=0)
    else:
        raise ValueError('Unknown ensemble method')
    options = dict(
        device=device, search_size=tuple(map(int, config['search_size'].split('x'))),
        confidence_threshold=config['confidence'], min_psr=config['min_psr'],
        q_pos=config['q_pos'], q_vel=config['q_vel'], r_px=config['r_px'],
        fusion_alpha=config['fusion_alpha'], medium_policy=config['medium_policy'],
        tolerance_px=config['tolerance'], template_refresh=config['template_refresh'],
        reset_velocity_scale=config['reset_velocity_scale'], cuda_graph=not config['no_graph'],
    )
    outputs, counts, invariance_max = [], {}, 0.
    for begin in range(0, len(frames), config['track_batch']):
        source = frames[begin:begin + config['track_batch']]
        prediction = global_px[begin:begin + len(source)]
        tracker = OfflineBatchTracker(matcher, **options)
        raw, stats = tracker.track(source, prediction, return_trace=True)
        altered_globals = prediction.copy()
        altered_globals[stats['tiers'] == 1] = [1000., -1000.]
        altered, other = tracker.track(source, altered_globals, return_trace=True)
        if not np.array_equal(stats['tiers'], other['tiers']):
            raise ValueError('Low-tier STNet coordinates affected gates')
        if not np.isfinite(raw).all() or not np.isfinite(altered).all():
            raise ValueError('Nonfinite independent tracker output')
        delta = float(np.max(np.abs(raw - altered)))
        invariance_max = max(invariance_max, delta)
        if delta > 1e-8:
            raise ValueError('Low-tier STNet coordinates affected raw trajectory')
        outputs.append(raw)
        for key, value in stats.items():
            if isinstance(value, int):
                counts[key] = counts.get(key, 0) + value
    metric = score_pixels(filter_records(np.concatenate(outputs), config['filter']), labels)
    validate_reproduction(expected['metrics']['ktcal'], metric, expected['tracker_stats'], counts)
    report = dict(
        expected_report_sha256=_sha256(args.expected_report), h5_sha256=expected['h5_sha256'],
        stnet_sha256=expected['stnet_sha256'], matcher_sha256=expected['matcher_sha256'],
        independent_metrics=metric, tracker_stats=counts,
        low_global_invariance_max_error_px=invariance_max,
        independent_torch_decoder=True, independent_pixel_metrics=True,
        full_9300_frames=True, final_holdout_used=False, labels_used_for_inference=False, passed=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    main()
