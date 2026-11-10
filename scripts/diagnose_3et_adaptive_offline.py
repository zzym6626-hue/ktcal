#!/usr/bin/env python3
"""Validation-only 3ET adaptive offline correction and per-frame diagnostics."""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from ktcal_paper.adaptive_offline import (
    adaptive_smooth, diagnose_frame_groups, select_reliable_match,
)
from ktcal_paper.combined_3et import (
    SCALE, _sha256, load_matcher_checkpoint, load_stnet_checkpoint,
    moving_average_sequence_predictions, precompute_global_predictions,
    summarize_sequence_predictions, validate_validation_h5,
)
from ktcal_paper.tracker import PaperTracker


def _combine_diagnostics(rows):
    combined = {}
    for group in rows[0]:
        combined[group] = {}
        for name in rows[0][group]:
            entries = [row[group][name] for row in rows]
            count = sum(row['frames'] for row in entries)
            combined[group][name] = {
                'frames': count,
                'mean_error_px': (sum(row['mean_error_px'] * row['frames']
                                     for row in entries if row['frames']) / count
                                  if count else None),
                'p3': (sum(row['p3'] * row['frames'] for row in entries
                           if row['frames']) / count if count else None),
            }
    return combined


@torch.inference_mode()
def evaluate_adaptive_offline(h5_path, stnet_checkpoint, matcher_checkpoint,
                              output, *, batch_size=4, threads=4, device='cuda'):
    h5_path = Path(h5_path).resolve()
    if h5_path.name != 'test_aug.h5':
        raise ValueError('This diagnostic accepts only test_aug.h5 validation data')
    h5_path, label_key, frame_shape, _ = validate_validation_h5(h5_path)
    output = Path(output).resolve()
    sidecar = output.with_suffix('.npz')
    if output.exists() or sidecar.exists():
        raise FileExistsError('Refusing to overwrite existing report/sidecar')
    if batch_size < 1 or threads < 1:
        raise ValueError('batch_size and threads must be positive')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(threads)
    selected_device = torch.device(device)
    stnet, config, divisor = load_stnet_checkpoint(stnet_checkpoint, selected_device)
    matcher, _ = load_matcher_checkpoint(matcher_checkpoint, selected_device)

    names = ('stnet_soft', 'stnet_ma11', 'stnet_adaptive', 'ktcal_ma11',
             'ktcal_adaptive', 'ktcal_guard_adaptive')
    predictions = {key: [] for key in names}
    truths, records, diagnostics = [], [], []
    postprocess_ms = 0.
    with h5py.File(h5_path, 'r') as handle:
        for start in range(0, frame_shape[0], batch_size):
            stop = min(start + batch_size, frame_shape[0])
            raw = handle['frames'][start:stop].astype(np.float32)
            labels = handle[label_key][start:stop, :, :2].astype(np.float32)
            global_px = precompute_global_predictions(
                stnet, torch.from_numpy(raw), selected_device, divisor,
                decode='soft')
            for index in range(stop - start):
                tracker = PaperTracker(stnet, matcher, device=selected_device,
                                       capture='fine', subpixel=True,
                                       fusion_alpha=.75, record_gate_trace=True)
                online_px, _, stats = tracker.track(
                    torch.from_numpy(raw[index]),
                    global_predictions=global_px[index])
                reference = global_px[index] / SCALE
                online = online_px.astype(np.float32) / SCALE
                trace = stats['gate_trace']
                confidence = np.array([
                    item['confidence'] if item['route'] == 'matched_update'
                    and not item['border_peak'] else 1. for item in trace],
                    dtype=np.float32)
                event_counts = np.count_nonzero(raw[index], axis=(1, 2, 3))
                started = time.perf_counter()
                trusted, fallback = select_reliable_match(
                    reference, online, confidence)
                baseline = moving_average_sequence_predictions([online], 11)[0]
                stnet_ma = moving_average_sequence_predictions([reference], 11)[0]
                stnet_adaptive, speed = adaptive_smooth(reference, reference)
                adaptive, _ = adaptive_smooth(online, reference)
                guarded, _ = adaptive_smooth(trusted, reference)
                postprocess_ms += (time.perf_counter() - started) * 1000
                for key, value in (('stnet_soft', reference),
                                   ('stnet_ma11', stnet_ma),
                                   ('stnet_adaptive', stnet_adaptive),
                                   ('ktcal_ma11', baseline),
                                   ('ktcal_adaptive', adaptive),
                                   ('ktcal_guard_adaptive', guarded)):
                    predictions[key].append(value)
                truths.append(labels[index])
                diagnostics.append(diagnose_frame_groups(
                    reference, baseline, labels[index], event_counts))
                records.append({
                    'stnet_px': reference * SCALE,
                    'ktcal_px': online * SCALE,
                    'baseline_px': baseline * SCALE,
                    'adaptive_px': adaptive * SCALE,
                    'guarded_px': guarded * SCALE,
                    'truth_px': labels[index] * SCALE,
                    'motion_px_per_frame': speed,
                    'event_nonzero_pixels': event_counts,
                    'matcher_confidence': confidence,
                    'fallback': fallback,
                    'tier': np.asarray(stats['frame_tiers'], dtype=np.uint8),
                })
            print(f'Validated {stop}/{frame_shape[0]} chunks', flush=True)
    metrics = {key: summarize_sequence_predictions(value, truths)
               for key, value in predictions.items()}
    arrays = {key: np.stack([row[key] for row in records])
              for key in records[0]}
    report = {
        'protocol': 'validation test_aug, frozen full-100 bidirectional STNet + KTCAL',
        'h5': str(h5_path), 'h5_sha256': _sha256(h5_path),
        'stnet_checkpoint': str(Path(stnet_checkpoint).resolve()),
        'stnet_checkpoint_sha256': _sha256(stnet_checkpoint),
        'stnet_readout': config.get('readout', 'last'),
        'matcher_checkpoint_sha256': _sha256(matcher_checkpoint),
        'policy': {'slow_px_per_frame': .5, 'fast_px_per_frame': 1.5,
                   'short_ma': 3, 'long_ma': 11, 'matcher_min_confidence': .7,
                   'matcher_max_disagreement_px': 2.},
        'metrics': metrics,
        'ma11_diagnostics': _combine_diagnostics(diagnostics),
        'fallback_frames': int(arrays['fallback'].sum()),
        'postprocess_ms_per_frame': postprocess_ms / metrics['stnet_soft']['frames'],
        'latency_comparable': False,
        'per_frame_npz': str(sidecar),
        'final_holdout_used': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sidecar, **arrays)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--h5', required=True, help='Only test_aug.h5 accepted')
    parser.add_argument('--stnet-checkpoint', required=True)
    parser.add_argument('--matcher-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    report = evaluate_adaptive_offline(
        args.h5, args.stnet_checkpoint, args.matcher_checkpoint, args.output,
        device=args.device, batch_size=args.batch_size, threads=args.threads)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
