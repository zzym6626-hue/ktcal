#!/usr/bin/env python3
"""Validation-only offline simulation of deferred KTCAL medium corrections.

Full-100-frame bidirectional STNet is always computed in advance. This is not
an online speed measurement or evidence that the global model can be skipped.
The medium-frame answer is emitted before inspecting its STNet counterpart;
the counterpart is applied before the next frame (or at sequence end).
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from ktcal_paper.combined_3et import (
    SCALE, _sha256, load_matcher_checkpoint, load_stnet_checkpoint,
    moving_average_sequence_predictions, precompute_global_predictions,
    summarize_sequence_predictions, validate_validation_h5,
)
from ktcal_paper.matching import downsample
from ktcal_paper.tracker import PaperTracker, gate


@torch.inference_mode()
def simulate_deferred_sequence(frames, global_predictions, tracker, tolerance_px=1.):
    """Issue matched KF result; on the next tick refine <= tolerance, else reset.

    Correction occurs before the next frame is processed, so no already
    processed future observations are replayed. Emitted values stay unchanged;
    corrected values are a separate retrospective trajectory.
    """
    globals_px = np.asarray(global_predictions, dtype=np.float64)
    if (len(frames) < 1 or globals_px.shape != (len(frames), 2)
            or not np.isfinite(globals_px).all() or not np.isfinite(tolerance_px)
            or tolerance_px <= 0):
        raise ValueError('Expected finite [T,2] global predictions and positive tolerance')
    if tracker.fusion_alpha is not None:
        raise ValueError('Three-tier offline diagnostic forbids STNet fusion on the fast path')

    kf = tracker._kalman()
    template = coarse_template = None
    issued, corrected = [], []
    pending = None
    tiers = []
    stats = {'tier1': 0, 'tier2': 0, 'tier3': 0,
             'medium_rollbacks': 0, 'medium_refinements': 0,
             'medium_disagreement_px': [], 'medium_corrections': [],
             'border_peaks': 0, 'stnet_requests_simulated': 0,
             'global_predictions_precomputed': len(frames)}

    def apply_pending():
        nonlocal pending, template, coarse_template
        if pending is None:
            return
        index, previous = pending
        global_pos = globals_px[index]
        difference = float(np.linalg.norm(issued[index] - global_pos))
        stats['medium_disagreement_px'].append(difference)
        if difference > tolerance_px:
            # Discard that frame's matched Kalman update and reset its state.
            corrected[index] = kf.reset(global_pos, previous).copy()
            stats['medium_rollbacks'] += 1
            stats['medium_corrections'].append('rollback')
        else:
            corrected[index] = kf.update(global_pos).copy()
            stats['medium_refinements'] += 1
            stats['medium_corrections'].append('refine')
        template, coarse_template = tracker._template_at(
            frames[index:index + 1], global_pos)
        pending = None

    for index, frame in enumerate(frames):
        apply_pending()
        if kf.x is None:
            kf.initialize(globals_px[index])
            result = kf.x[:2].copy()
            tier = 3
        else:
            previous = kf.x[:2].copy()
            prediction = kf.predict()
            center = torch.as_tensor(prediction, device=tracker.device,
                                     dtype=torch.float32)[None]
            coarse_frame = (downsample(frame[None].to(tracker.device))
                            if tracker.capture == 'coarse' else None)
            match, confidence, details = tracker._match(
                template, coarse_template, frame, coarse_frame, center)
            stats['border_peaks'] += int(details['border_peak'])
            tier = gate(np.linalg.norm(prediction - match), confidence,
                        tracker.tau1, tracker.tau2,
                        border_peak=details['border_peak'],
                        psr=details['psr'], min_psr=tracker.min_psr,
                        escalate_tier=tracker.border_tier)
            if tier == 3:
                result = kf.reset(globals_px[index], previous).copy()
            else:
                result = kf.update(match).copy()
                if tier == 2:
                    pending = (index, previous)
        if tier == 3:
            template, coarse_template = tracker._template_at(
                frames[index:index + 1], globals_px[index])
        stats['tier%d' % tier] += 1
        stats['stnet_requests_simulated'] += int(tier > 1)
        tiers.append(tier)
        issued.append(result.copy())
        corrected.append(result.copy())
    apply_pending()  # the last frame's retrospective correction
    stats['tiers'] = tiers
    return np.asarray(issued), np.asarray(corrected), stats


@torch.inference_mode()
def evaluate_deferred_medium(h5_path, stnet_checkpoint, matcher_checkpoint,
                             output, *, batch_size=4, threads=4, device='cuda',
                             tau1=1., tau2=3., tolerance_px=1.):
    if Path(h5_path).name != 'test_aug.h5':
        raise ValueError('Offline validation accepts only test_aug.h5')
    h5_path, label_key, shape, _ = validate_validation_h5(h5_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing report')
    if batch_size < 1 or threads < 1 or not 0 <= tau1 < tau2 or tolerance_px <= 0:
        raise ValueError('Invalid batch size, threads, thresholds or tolerance')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(threads)
    selected_device = torch.device(device)
    stnet, config, divisor = load_stnet_checkpoint(stnet_checkpoint, selected_device)
    matcher, _ = load_matcher_checkpoint(matcher_checkpoint, selected_device)
    names = ('stnet_ma11', 'reference_fusion075_ma11',
             'proposed_issued_raw', 'proposed_corrected_raw',
             'proposed_issued_ma11', 'proposed_corrected_ma11')
    predictions = {key: [] for key in names}
    truths, traces, group_errors = [], [], {tier: [] for tier in (1, 2, 3)}
    with h5py.File(h5_path, 'r') as handle:
        for start in range(0, shape[0], batch_size):
            stop = min(start + batch_size, shape[0])
            raw = handle['frames'][start:stop].astype(np.float32)
            labels = handle[label_key][start:stop, :, :2].astype(np.float32)
            global_px = precompute_global_predictions(
                stnet, torch.from_numpy(raw), selected_device, divisor, decode='soft')
            for offset in range(stop - start):
                frames = torch.from_numpy(raw[offset])
                proposed = PaperTracker(stnet, matcher, device=selected_device,
                                        tau1=tau1, tau2=tau2, border_tier=2,
                                        capture='fine', subpixel=True)
                issued, corrected, stats = simulate_deferred_sequence(
                    frames, global_px[offset], proposed, tolerance_px=tolerance_px)
                reference = PaperTracker(stnet, matcher, device=selected_device,
                                         capture='fine', subpixel=True,
                                         fusion_alpha=.75)
                old, _, _ = reference.track(frames, global_predictions=global_px[offset])
                truths.append(labels[offset])
                sequences = {
                    'stnet_ma11': moving_average_sequence_predictions(
                        [global_px[offset] / SCALE], 11)[0],
                    'reference_fusion075_ma11': moving_average_sequence_predictions(
                        [old / SCALE], 11)[0],
                    'proposed_issued_raw': issued / SCALE,
                    'proposed_corrected_raw': corrected / SCALE,
                    'proposed_issued_ma11': moving_average_sequence_predictions(
                        [issued / SCALE], 11)[0],
                    'proposed_corrected_ma11': moving_average_sequence_predictions(
                        [corrected / SCALE], 11)[0],
                }
                for name, values in sequences.items():
                    predictions[name].append(values)
                errors = np.linalg.norm(
                    issued - labels[offset] * SCALE, axis=1)
                for tier in (1, 2, 3):
                    group_errors[tier].extend(errors[np.asarray(stats['tiers']) == tier])
                traces.append(stats)
            print('Validated %d/%d sequences' % (stop, shape[0]), flush=True)
    metrics = {name: summarize_sequence_predictions(values, truths)
               for name, values in predictions.items()}
    total = sum(len(item['tiers']) for item in traces)
    report = {
        'protocol': 'test_aug full-100 bidirectional offline; deferred correction before next frame',
        'h5': str(h5_path), 'h5_sha256': _sha256(h5_path),
        'stnet_checkpoint': str(Path(stnet_checkpoint).resolve()),
        'stnet_checkpoint_sha256': _sha256(stnet_checkpoint),
        'matcher_checkpoint': str(Path(matcher_checkpoint).resolve()),
        'matcher_checkpoint_sha256': _sha256(matcher_checkpoint),
        'stnet_readout': config.get('readout', 'last'),
        'policy': {'tau1_px': tau1, 'tau2_px': tau2, 'border_tier': 2,
                   'medium_difference_reference': 'matched_updated_kalman_vs_stnet',
                   'medium_rollback_if_gt_px': tolerance_px,
                   'medium_refine_if_lte_px': tolerance_px,
                   'small': 'matched_kalman_direct',
                   'large': 'synchronous_stnet_reset',
                   'ma_window': 11},
        'metrics': metrics,
        'tier_issued_raw': {str(tier): {
            'frames': len(values),
            'p3': float(np.mean(np.asarray(values) < 3)) if values else None,
            'mean_error_px': float(np.mean(values)) if values else None}
            for tier, values in group_errors.items()},
        'stats': {key: int(sum(row[key] for row in traces)) for key in (
            'tier1', 'tier2', 'tier3', 'medium_rollbacks',
            'medium_refinements', 'stnet_requests_simulated', 'border_peaks',
            'global_predictions_precomputed')},
        'frames': total,
        'latency_comparable': False,
        'no_actual_stnet_skips': True,
        'final_holdout_used': False,
        'limitation': 'Medium correction arrives before next frame: no already-processed '
                      'future frames require replay. Online asynchronous latency not measured.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--h5', required=True)
    parser.add_argument('--stnet-checkpoint', required=True)
    parser.add_argument('--matcher-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--tau1', type=float, default=1.)
    parser.add_argument('--tau2', type=float, default=3.)
    parser.add_argument('--tolerance-px', type=float, default=1.)
    args = parser.parse_args()
    print(json.dumps(evaluate_deferred_medium(
        args.h5, args.stnet_checkpoint, args.matcher_checkpoint, args.output,
        batch_size=args.batch_size, threads=args.threads, device=args.device,
        tau1=args.tau1, tau2=args.tau2, tolerance_px=args.tolerance_px),
        indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
