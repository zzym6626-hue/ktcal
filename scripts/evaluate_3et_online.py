#!/usr/bin/env python3
"""Evaluate past-only, on-demand STNet + KTCAL with independent batch-one streams."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
import torch

from ktcal_paper.combined_3et import (
    SCALE, _sha256, load_matcher_checkpoint, load_stnet_checkpoint,
    validate_validation_h5,
)
from ktcal_paper.evaluation_protocol import summarize_pixel_errors
from ktcal_paper.offline_batch import OfflineBatchTracker
from ktcal_paper.offline_experiment import synchronize
from ktcal_paper.online_inference import (
    CausalOutputFilter, CausalPredictor, CausalViewDecoder, GraphDecoder, LastStepSTNet,
)
from ktcal_paper.streaming_inference import StreamingPredictor
from models.streaming_stnet import StreamingSTNet


def build_parser():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', help='Reviewed JSON inference defaults; explicit flags override')
    parser.add_argument('--h5', default='data/3et/test_aug.h5')
    parser.add_argument('--stnet', default='weights/3et/stnet_best.pt')
    parser.add_argument('--matcher', default='weights/3et/matcher_best.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--history-stride', type=int, default=2)
    parser.add_argument('--right-context', type=int, default=0,
                        help='Synthetic repeated-current frames; reads original current index')
    parser.add_argument('--warm-start-steps', type=int, default=0,
                        help='Streaming only: extra first-received-frame state updates at record start')
    parser.add_argument('--views', default='0:0:0,0:0:1')
    parser.add_argument('--temperature', type=float, default=.65)
    parser.add_argument('--flip-axis-px', type=float,
                        help='Streaming flip coordinate convention; defaults to checkpoint or legacy79')
    parser.add_argument('--ensemble', choices=('mean', 'median'), default='median')
    parser.add_argument('--search-size', default='37x32')
    parser.add_argument('--confidence', type=float, default=.28)
    parser.add_argument('--activity-reset-threshold', type=float,
                        help='Current raw event sum that forces STNet reset and skips fine matching')
    parser.add_argument('--min-psr', type=float, default=2.55)
    parser.add_argument('--no-psr', action='store_true')
    parser.add_argument('--q-pos', type=float, default=.05)
    parser.add_argument('--q-vel', type=float, default=.2)
    parser.add_argument('--r-px', type=float, default=32.)
    parser.add_argument('--fusion-alpha', type=float, default=.75)
    parser.add_argument('--tolerance', type=float, default=1.)
    parser.add_argument('--reset-velocity-scale', type=float, default=1.)
    parser.add_argument('--ema-alpha', type=float)
    parser.add_argument('--ema-max-step', type=float, default=2.5)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--no-graph', action='store_true')
    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument('--no-persistent-state', action='store_true', default=True,
                             help='Copy streaming recurrent state in Python each frame (default)')
    state_group.add_argument('--persistent-state', action='store_false', dest='no_persistent_state',
                             help='Experimental in-graph state copies; slower in the recorded A/B')
    parser.add_argument('--no-template-graph', action='store_true')
    parser.add_argument('--no-optimize-stnet', action='store_true',
                        help='Use original all-timestep STNet for an equivalence/latency comparison')
    parser.add_argument('--save-predictions', action='store_true')
    return parser


def parse_args(argv=None):
    preparser = argparse.ArgumentParser(add_help=False)
    preparser.add_argument('--config')
    supplied, _ = preparser.parse_known_args(argv)
    parser = build_parser()
    if supplied.config:
        configuration = json.loads(Path(supplied.config).read_text())
        allowed = {action.dest for action in parser._actions} - {'help', 'config', 'output'}
        if not isinstance(configuration, dict) or set(configuration) - allowed:
            raise ValueError('Unknown or invalid inference configuration keys')
        for action in parser._actions:
            if action.dest not in configuration:
                continue
            value = configuration[action.dest]
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                valid = isinstance(value, bool)
            elif action.type is int:
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif action.type is float:
                valid = (isinstance(value, (int, float)) and not isinstance(value, bool)
                         or value is None and action.dest in ('ema_alpha', 'min_psr', 'flip_axis_px',
                                                              'activity_reset_threshold'))
            else:
                valid = isinstance(value, str)
            if not valid or action.choices is not None and value not in action.choices:
                raise ValueError(f'Invalid configuration type or choice: {action.dest}')
        parser.set_defaults(**configuration)
    return parser.parse_args(argv)


def validate_args(args):
    if min(args.window, args.history_stride, args.repeats, args.threads) < 1:
        raise ValueError('Positive window, history stride and repetitions required')
    if args.right_context < 0:
        raise ValueError('Synthetic right context must be nonnegative')
    if args.warm_start_steps < 0:
        raise ValueError('Nonnegative warm-start steps required')
    if args.flip_axis_px is not None and (not np.isfinite(args.flip_axis_px)
                                          or not 79. <= args.flip_axis_px <= 80.):
        raise ValueError('Finite streaming flip axis in[79,80] required')
    if args.activity_reset_threshold is not None and (not np.isfinite(args.activity_reset_threshold)
                                                      or args.activity_reset_threshold <= 0):
        raise ValueError('Current activity reset threshold must be finite and positive')
    if any(not np.isfinite(v) or v <= 0 for v in
           (args.temperature, args.q_pos, args.q_vel, args.r_px, args.tolerance)):
        raise ValueError('Positive finite decoding and covariance values required')
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in
           (args.confidence, args.fusion_alpha, args.reset_velocity_scale)):
        raise ValueError('Finite thresholds in [0,1] required')
    if args.min_psr is not None and not np.isfinite(args.min_psr):
        raise ValueError('PSR threshold must be finite')
    CausalOutputFilter(args.ema_alpha, args.ema_max_step)
    size = tuple(map(int, args.search_size.split('x')))
    if len(size) != 2 or any(v < t for v, t in zip(size, (30, 25))):
        raise ValueError('Search must contain the 30x25 template')
    views = []
    for value in args.views.split(','):
        dx, dy, flip = map(int, value.split(':'))
        if abs(dx) >= 80 or abs(dy) >= 60 or flip not in (0, 1):
            raise ValueError('Invalid spatial view')
        views.append((dx, dy, bool(flip)))
    return size, views


def gpu_processes(device):
    if torch.device(device).type != 'cuda':
        return []
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        return ['process snapshot unavailable']
    return [row.strip() for row in result.stdout.splitlines()
            if row.split(',')[0].strip() != str(os.getpid())]


def build_predictor(model, divisor, views, args, device):
    if isinstance(model, StreamingSTNet):
        if args.history_stride != 1 or args.right_context != 0:
            raise ValueError('Streaming requires history_stride=1 and right_context=0')
        return StreamingPredictor(model, divisor, views, args.temperature, args.ensemble,
                                  cuda_graph=device.type == 'cuda' and not args.no_graph,
                                  warm_start_steps=args.warm_start_steps,
                                  flip_axis_px=args.flip_axis_px,
                                  persistent_state=not args.no_persistent_state)
    if args.warm_start_steps:
        raise ValueError('Warm-start steps require a streaming model')
    if args.flip_axis_px is not None and args.flip_axis_px != 79.:
        raise ValueError('Flip-axis overrides require a streaming model')
    if not args.no_optimize_stnet:
        model = LastStepSTNet(model, current_offset=args.right_context)
    decoder = CausalViewDecoder(model, divisor, views, args.temperature, args.ensemble,
                                args.right_context).to(device)
    if device.type == 'cuda' and not args.no_graph:
        decoder = GraphDecoder(decoder)
    return CausalPredictor(decoder, args.window, args.history_stride)


def predictor_work_stats(stats, predictor):
    if not isinstance(predictor, StreamingPredictor):
        return stats
    if stats['stnet_requests'] != predictor.coordinate_head_requests:
        raise ValueError('Tracker and streaming coordinate request counts disagree')
    return dict(stats, model_forward_passes=predictor.feature_forward_passes,
                feature_updates=predictor.feature_updates,
                received_frame_updates=predictor.received_frame_updates,
                synthetic_feature_updates=predictor.synthetic_feature_updates,
                coordinate_head_requests=predictor.coordinate_head_requests,
                coordinate_head_forward_passes=predictor.coordinate_head_forward_passes)


def warm_up_online_tracker(tracker, predictor, frames, output_transform=None):
    """Initialize every batch-one graph even when activity skips all matching."""
    predictor.reset()
    tracker.track(frames[:1], predictor, output_transform=output_transform)
    image = frames[:1, 0].to(tracker.device)
    centers = image.new_tensor([[40., 30.]])
    template = tracker.template_crop(image, centers)
    tracker.matcher(template, image, centers)
    synchronize(tracker.device)


def main(argv=None):
    args = parse_args(argv)
    if args.no_psr:
        args.min_psr = None
    size, views = validate_args(args)
    output = Path(args.output)
    predictions_path = output.with_suffix('.npz')
    if output.exists() or (args.save_predictions and predictions_path.exists()):
        raise FileExistsError('Choose a new output; existing results are never overwritten')
    h5, label_key, shape, _ = validate_validation_h5(args.h5)
    if h5.name != 'test_aug.h5' or shape != (93, 100, 2, 60, 80):
        raise ValueError('Online selection requires full 93x100 test_aug.h5')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261113)
    root = Path(__file__).resolve().parents[1]
    source_names = ('scripts/evaluate_3et_online.py', 'models/STNet.py',
                    'models/streaming_stnet.py', 'models/localization_streaming_stnet.py',
                    'models/current_readout_streaming_stnet.py',
                    'models/raw_current_streaming_stnet.py',
                    'models/memory_readout_streaming_stnet.py',
                    'models/initialized_streaming_stnet.py',
                    'ktcal_paper/streaming_inference.py', 'ktcal_paper/online_inference.py',
                    'ktcal_paper/offline_batch.py', 'ktcal_paper/matching.py',
                    'ktcal_paper/combined_3et.py')
    source_hashes = {name: _sha256(root / name) for name in source_names if (root / name).is_file()}
    model, config, divisor = load_stnet_checkpoint(args.stnet, args.device)
    matcher, matcher_config = load_matcher_checkpoint(args.matcher, args.device)
    predictor = build_predictor(model, divisor, views, args, device)
    streaming = isinstance(predictor, StreamingPredictor)
    tracker = OfflineBatchTracker(
        matcher, device=args.device, search_size=size,
        confidence_threshold=args.confidence, min_psr=args.min_psr,
        q_pos=args.q_pos, q_vel=args.q_vel, r_px=args.r_px,
        fusion_alpha=args.fusion_alpha, medium_policy='state',
        tolerance_px=args.tolerance, template_refresh=0,
        reset_velocity_scale=args.reset_velocity_scale,
        activity_reset_threshold=args.activity_reset_threshold,
        cuda_graph=device.type == 'cuda' and not args.no_graph,
        synchronize_frames=True,
        template_cuda_graph=device.type == 'cuda' and not args.no_graph and not args.no_template_graph,
    )
    with h5py.File(h5, 'r') as handle:
        frames = torch.from_numpy(handle['frames'][:].astype(np.float32))
        labels = handle[label_key][:, :, :2].astype(np.float32)
    print('Loaded. Warming up causal model and matcher graphs...', flush=True)
    measurements, interference = [], []
    with torch.inference_mode():
        warm_up_online_tracker(tracker, predictor, frames, output_transform=CausalOutputFilter(
            args.ema_alpha, args.ema_max_step))
        for repeat in range(args.repeats):
            results, counts, frame_times, total_ms = [], {}, [], 0.
            interference.append(gpu_processes(device))
            for record in range(len(frames)):
                predictor.reset()
                smoother = CausalOutputFilter(args.ema_alpha, args.ema_max_step)
                synchronize(args.device)
                started = time.perf_counter()
                raw, stats = tracker.track(frames[record:record + 1], predictor,
                                           output_transform=smoother)
                synchronize(args.device)
                total_ms += 1000 * (time.perf_counter() - started)
                stats = predictor_work_stats(stats, predictor)
                results.append(raw)
                frame_times.extend(stats['frame_processing_ms'])
                for key, value in stats.items():
                    if isinstance(value, int) and not isinstance(value, bool):
                        counts[key] = counts.get(key, 0) + value
                if (record + 1) % 31 == 0:
                    interference.append(gpu_processes(device))
                    print(f'Repeat {repeat + 1}: {record + 1}/93 records', flush=True)
            predicted = np.concatenate(results)
            if counts['small_fusion_calls'] or counts['global_predictions_precomputed']:
                raise ValueError('Online protocol violated')
            metrics = summarize_pixel_errors(predicted / SCALE, labels)
            if measurements:
                if counts != measurements[0]['tracker_stats']:
                    raise ValueError('Routing changed across repeated evaluations')
                np.testing.assert_allclose(predicted, first_predicted, atol=1e-5, rtol=0)
            else:
                first_predicted = predicted.copy()
            measurements.append(dict(
                total_ms=total_ms, ms_per_frame=total_ms / counts['frames'],
                frame_processing_p50_ms=float(np.median(frame_times)),
                frame_processing_p95_ms=float(np.percentile(frame_times, 95)),
                frame_processing_p99_ms=float(np.percentile(frame_times, 99)),
                metrics=metrics, tracker_stats=counts,
            ))
            print(f"Repeat {repeat + 1}/{args.repeats}: P3={metrics['p3']:.4%}, "
                  f"{measurements[-1]['ms_per_frame']:.3f} ms/frame, "
                  f"STNet={counts['stnet_requests']}/{counts['frames']}", flush=True)
    report = dict(
        schema=1, protocol='3ET test_aug full-9300 causal on-demand batch-one',
        config=dict(vars(args), views=views, track_batch=1, tau1=3., tau2=5.,
                    medium_policy='state', template_refresh=0),
        stnet_config=config, matcher_config=matcher_config,
        stnet_sha256=_sha256(args.stnet), matcher_sha256=_sha256(args.matcher),
        implementation_sha256=source_hashes,
        h5_sha256=_sha256(h5), metrics=metrics, tracker_stats=counts,
        inference_config_sha256=_sha256(args.config) if args.config else None,
        latency=measurements,
        median_ms_per_frame=float(np.median([m['ms_per_frame'] for m in measurements])),
        stnet_input_views=counts.get('feature_updates', counts['stnet_requests']) * len(views),
        coordinate_head_input_views=counts['stnet_requests'] * len(views),
        temporal_inference='causal_streaming' if streaming else 'causal_received_window',
        spatial_flip_axis_px=predictor.flip_axis_px if streaming else 79.,
        recurrent_history='entire_received_record' if streaming else 'limited_window',
        future_frames_used=0, global_predictions_precomputed=0,
        synthetic_right_context='repeat_current_frame' if args.right_context else 'none',
        synthetic_warm_start='repeat_first_received_frame' if args.warm_start_steps else 'none',
        medium_timing='synchronous', output_smoothing='causal EMA' if args.ema_alpha else 'none',
        frame_timing_complete_gpu=True,
        labels_used_for_inference=False, final_holdout_used=False, validation_selected=True,
        activity_gate=('received_raw_frame_sum_forces_large_reset_and_skips_matcher'
                       if args.activity_reset_threshold is not None else 'disabled'),
        latency_scope=('warm sequential H2D + every-frame feature/state updates + requested coordinate heads + soft decode + matcher + KF + D2H + causal output filter; excludes loading/scoring/graph capture'
                       if streaming else 'warm sequential H2D + ring + requested STNet views + soft decode + matcher + KF + D2H + causal output filter; excludes loading/scoring/graph capture'),
        gpu_other_process_snapshots=interference,
        latency_valid_no_observed_gpu_competition=not any(interference),
        environment=dict(torch=torch.__version__, numpy=np.__version__,
                         cudnn_enabled=torch.backends.cudnn.enabled,
                         cudnn_benchmark=torch.backends.cudnn.benchmark,
                         cudnn_deterministic=torch.backends.cudnn.deterministic,
                         cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                         matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                         gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None),
    )
    if any(_sha256(root / name) != value for name, value in source_hashes.items()):
        raise RuntimeError('Inference implementation changed during evaluation; choose a fresh run')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    if args.save_predictions:
        with predictions_path.open('xb') as handle:
            np.savez_compressed(handle, predicted_px=predicted, labels=labels)
    print(f'Report: {output}', flush=True)
    return report


if __name__ == '__main__':
    main()
