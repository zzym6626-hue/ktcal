"""Validation-only, ground-truth-aware audit of causal KTCAL gate decisions."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .combined_3et import (
    CausalLastStepSTNet, SCALE, _aggregate_tracker_stats, _sha256,
    load_matcher_checkpoint, load_stnet_checkpoint,
    summarize_sequence_predictions, validate_validation_h5,
)
from .tracker import PaperTracker


ROUTES = ('prediction_only', 'matched_update', 'async_prediction', 'stnet_reset')


@torch.inference_mode()
def counterfactual_bad_fast_frames(trace, truth_normalized, frames, model, device,
                                    *, batch_size=16):
    """Post-hoc oracle: would on-demand STNet recover *known* fast-path misses?

    Never used to make online gate decisions: choosing misses requires labels.
    """
    truth = np.asarray(truth_normalized, dtype=np.float64)
    if (truth.shape != (len(trace), 2) or len(frames) != len(trace)
            or batch_size < 1):
        raise ValueError('Counterfactual trace, labels and frames must align')
    missed = [index for index, row in enumerate(trace)
              if row['route'] != 'stnet_reset'
              and np.linalg.norm(np.asarray(row['output_px']) - truth[index] * SCALE) >= 3]
    counts = {route: {'failed': 0, 'recoverable': 0} for route in ROUTES[:-1]}
    for start in range(0, len(missed), batch_size):
        indices = missed[start:start + batch_size]
        windows = []
        for index in indices:
            history = frames[max(0, index - 15):index + 1]
            if len(history) < 16:
                history = torch.cat([
                    history[:1].expand(16 - len(history), -1, -1, -1), history])
            windows.append(history)
        predictions = model(torch.stack(windows).to(device)).detach().cpu().numpy() * SCALE
        if predictions.shape != (len(indices), 2) or not np.isfinite(predictions).all():
            raise ValueError('Counterfactual model must return finite [B,2] coordinates')
        for index, predicted in zip(indices, predictions):
            route = trace[index]['route']
            counts[route]['failed'] += 1
            counts[route]['recoverable'] += int(
                np.linalg.norm(predicted - truth[index] * SCALE) < 3)
    return {
        'failed_fast_frames': len(missed),
        'stnet_would_recover': sum(row['recoverable'] for row in counts.values()),
        'by_route': counts,
        'oracle_only': True,
    }


def summarize_gate_trace(trace, truth_normalized):
    """Compare emitted predictions and pre-gate observations with labels.

    Labels are used only *after* online tracking. Counts must not be treated
    as a deployable error detector or a guarantee that a reset would help.
    """
    truth = np.asarray(truth_normalized, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[1] != 2 or not len(trace) or len(trace) != len(truth):
        raise ValueError('Gate trace and labels must be nonempty aligned [T,2]')
    output = np.asarray([row['output_px'] for row in trace], dtype=np.float64)
    if output.shape != truth.shape or not np.isfinite(output).all():
        raise ValueError('Gate trace must contain finite output coordinates')
    truth_px = truth * SCALE
    errors = np.linalg.norm(output - truth_px, axis=1)
    groups = {}
    for route in ROUTES:
        selected = np.array([row['route'] == route for row in trace], dtype=bool)
        group = errors[selected]
        groups[route] = {
            'frames': int(group.size),
            'p3': float(np.mean(group < 3)) if group.size else None,
            'mean_error_px': float(np.mean(group)) if group.size else None,
            'median_error_px': float(np.median(group)) if group.size else None,
            'bad_3px': int(np.sum(group >= 3)),
        }
    if sum(group['frames'] for group in groups.values()) != len(trace):
        raise ValueError('Gate trace has an unknown route')
    spoiled = recovered = consensus_wrong = 0
    for index, row in enumerate(trace):
        if row['route'] != 'matched_update':
            continue
        if row['prediction_px'] is None or row['match_px'] is None:
            raise ValueError('Matched-update gate trace requires prediction and match')
        predicted = np.asarray(row['prediction_px'], dtype=np.float64)
        matched = np.asarray(row['match_px'], dtype=np.float64)
        if (predicted.shape != (2,) or matched.shape != (2,)
                or not np.isfinite(predicted).all() or not np.isfinite(matched).all()):
            raise ValueError('Gate observations must be finite [2] coordinates')
        prediction_bad = np.linalg.norm(predicted - truth_px[index]) >= 3
        match_bad = np.linalg.norm(matched - truth_px[index]) >= 3
        output_bad = errors[index] >= 3
        spoiled += int(not prediction_bad and output_bad)
        recovered += int(prediction_bad and not output_bad)
        consensus_wrong += int(
            prediction_bad and match_bad and np.linalg.norm(predicted - matched) < 3)
    return {
        'frames': len(trace), 'p3': float(np.mean(errors < 3)),
        'mean_error_px': float(np.mean(errors)),
        'routes': groups,
        'matched_update': {
            'spoiled_good_prediction': spoiled,
            'recovered_bad_prediction': recovered,
            'prediction_and_match_wrong_but_agree': consensus_wrong,
        },
        'note': 'Ground truth is diagnostic only; a <3 px gate disagreement does not bound gaze error.',
    }


@torch.inference_mode()
def diagnose_online_gate(h5_path, stnet_checkpoint, matcher_checkpoint, output,
                         *, device='cuda', threads=4,
                         max_prediction_variance_px2=10.,
                         max_consecutive_prediction_skips=1):
    """Replay the same on-demand validation path, without changing its gate."""
    h5_path, label_key, frames_shape, _ = validate_validation_h5(h5_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if threads < 1 or (str(device).startswith('cuda') and not torch.cuda.is_available()):
        raise ValueError('Requested device or thread count is unavailable')
    torch.set_num_threads(threads)
    resolved_device = torch.device(device)
    stnet, stnet_config, divisor = load_stnet_checkpoint(stnet_checkpoint, resolved_device)
    matcher, matcher_config = load_matcher_checkpoint(matcher_checkpoint, resolved_device)
    model = CausalLastStepSTNet(stnet, divisor).eval()
    traces, truths, predictions, truth_sequences, tracker_stats = [], [], [], [], []
    oracle_counts = {route: {'failed': 0, 'recoverable': 0} for route in ROUTES[:-1]}
    with h5py.File(h5_path, 'r') as handle:
        for sequence in range(frames_shape[0]):
            frames = torch.from_numpy(handle['frames'][sequence].astype(np.float32))
            truth = handle[label_key][sequence, :, :2].astype(np.float32)
            tracker = PaperTracker(
                model, matcher, device=resolved_device, capture='fine',
                subpixel=True, fusion_alpha=None, record_gate_trace=True,
                max_prediction_variance_px2=max_prediction_variance_px2,
                max_consecutive_prediction_skips=max_consecutive_prediction_skips)
            online_px, _, stats = tracker.track(frames)
            sequence_trace = stats.pop('gate_trace')
            oracle = counterfactual_bad_fast_frames(
                sequence_trace, truth, frames, model, resolved_device)
            for route in oracle_counts:
                for key in ('failed', 'recoverable'):
                    oracle_counts[route][key] += oracle['by_route'][route][key]
            traces.extend(sequence_trace)
            truths.append(truth)
            predictions.append(online_px.astype(np.float32) / SCALE)
            truth_sequences.append(truth)
            tracker_stats.append(stats)
    metrics = summarize_sequence_predictions(predictions, truth_sequences)
    diagnostics = summarize_gate_trace(traces, np.concatenate(truths))
    recovered = sum(row['recoverable'] for row in oracle_counts.values())
    diagnostics['counterfactual_bad_fast_frames'] = {
        'failed_fast_frames': sum(row['failed'] for row in oracle_counts.values()),
        'stnet_would_recover': recovered,
        'by_route': oracle_counts,
        'oracle_selected_p3_if_replaced': (
            round(metrics['p3'] * metrics['frames']) + recovered) / metrics['frames'],
        'oracle_only': True,
        'note': 'Requires labels to choose failures; not a deployable gate or a real online P3.',
    }
    if abs(metrics['p3'] - diagnostics['p3']) > 1e-3:
        raise RuntimeError('Gate trace and online predictions disagree')
    stats = _aggregate_tracker_stats(tracker_stats, metrics['frames'])
    stats['global_prediction_source'] = 'on_demand_causal_16_step'
    report = {
        'schema': 1,
        'protocol': '3ET validation causal-16 KTCAL online gate audit; no fusion',
        'h5': str(h5_path), 'h5_sha256': _sha256(h5_path),
        'stnet_checkpoint': str(Path(stnet_checkpoint).resolve()),
        'stnet_checkpoint_sha256': _sha256(stnet_checkpoint),
        'stnet_best_epoch': stnet_config.get('best_epoch'),
        'matcher_checkpoint': str(Path(matcher_checkpoint).resolve()),
        'matcher_checkpoint_sha256': _sha256(matcher_checkpoint),
        'matcher_selected_epoch': matcher_config.get('selected_epoch'),
        'matching': {
            'capture': 'fine', 'search_size': [35, 30],
            'locator': 'parabolic_subpixel', 'fusion_alpha': None,
            'max_prediction_variance_px2': max_prediction_variance_px2,
            'max_consecutive_prediction_skips': max_consecutive_prediction_skips,
        },
        'metrics': {'ktcal_online': metrics}, 'diagnostics': diagnostics,
        'tracker_stats': stats,
        'latency_comparable': False, 'final_holdout_used': False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
