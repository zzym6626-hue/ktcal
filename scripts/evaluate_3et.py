#!/usr/bin/env python3
"""Evaluate the released 100-frame STNet + KTCAL on 3ET test_aug.h5."""
import argparse
import json
from pathlib import Path
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
from ktcal_paper.offline_experiment import (
    average_views, predict_views, smooth_pixels, synchronize, validate_candidate_metrics,
)


def build_parser():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--h5', default='data/3et/test_aug.h5')
    parser.add_argument('--stnet-checkpoint', '--stnet', dest='stnet',
                        default='weights/3et/stnet_best.pt')
    parser.add_argument('--matcher-checkpoint', '--matcher', dest='matcher',
                        default='weights/3et/matcher_best.pt')
    parser.add_argument('--output', default='local_run/3et_val.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--track-batch', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--views', default=(
        '0:0:0,0:0:1,-2:0:0,-2:0:1,2:0:0,2:0:1,'
        '0:-2:0,0:-2:1,0:2:0,0:2:1'))
    parser.add_argument('--temperature', type=float, default=.65)
    parser.add_argument('--ensemble', choices=('mean', 'median', 'trimmed'), default='median')
    parser.add_argument('--filter', '--smoothing', dest='filter', default='ma7',
                        choices=('none', 'ma3', 'ma5', 'ma7', 'ma9', 'ma11', 'ma13', 'ma15'))
    parser.add_argument('--search-size', default='37x32', help='WIDTHxHEIGHT fine search')
    parser.add_argument('--confidence', type=float, default=.28)
    parser.add_argument('--min-psr', type=float, default=2.55)
    parser.add_argument('--q-pos', type=float, default=.05)
    parser.add_argument('--q-vel', type=float, default=.2)
    parser.add_argument('--r-px', type=float, default=32.)
    parser.add_argument('--fusion-alpha', type=float, default=.75,
                        help='Applied only on accepted medium-tier refinements')
    parser.add_argument('--tolerance', type=float, default=1.)
    parser.add_argument('--no-graph', action='store_true', help='Disable matcher CUDA Graph')
    return parser


def tracker_options(args):
    return dict(
        device=args.device, search_size=tuple(map(int, args.search_size.split('x'))),
        confidence_threshold=args.confidence, min_psr=args.min_psr,
        q_pos=args.q_pos, q_vel=args.q_vel, r_px=args.r_px,
        medium_policy='state', fusion_alpha=args.fusion_alpha,
        tolerance_px=args.tolerance, template_refresh=0, reset_velocity_scale=1.,
        cuda_graph=torch.device(args.device).type == 'cuda' and not args.no_graph,
    )


def validate_args(args):
    if min(args.batch_size, args.track_batch, args.repeats) < 1:
        raise ValueError('Batch sizes and repeats must be positive')
    positive = (args.temperature, args.tolerance, args.q_pos, args.q_vel, args.r_px)
    if any(not np.isfinite(value) or value <= 0 for value in positive):
        raise ValueError('Inference temperatures, covariances and tolerance must be finite and positive')
    if (not np.isfinite(args.min_psr)
            or any(not np.isfinite(value) or not 0 <= value <= 1
                   for value in (args.confidence, args.fusion_alpha))):
        raise ValueError('Finite valid reliability and fusion thresholds required')
    size = tuple(map(int, args.search_size.split('x')))
    if len(size) != 2 or size[0] < 30 or size[1] < 25:
        raise ValueError('Fine search must contain the 30x25 template')
    views = []
    for value in args.views.split(','):
        dx, dy, flip = map(int, value.split(':'))
        if flip not in (0, 1) or abs(dx) >= 80 or abs(dy) >= 60:
            raise ValueError('Invalid spatial view')
        views.append((dx, dy, bool(flip)))
    return views


def print_summary(report, output):
    print('\n3ET validation: 100-frame sequences')
    print(f"{'Method':<20} {'Mean(px)':>10} {'P3':>9} {'P5':>9} {'P10':>9} {'ms/frame':>10}")
    frames = report['metrics']['ktcal']['frames']
    stnet_ms = float(np.median([row['stnet_filtered_ms'] for row in report['latency']])) / frames
    for key, name, latency in (('stnet', 'STNet', stnet_ms),
                               ('ktcal', 'STNet + KTCAL', report['median_ms_per_frame'])):
        metric = report['metrics'][key]
        print(f"{name:<20} {metric['mean_error_px']:>10.3f} "
              f"{metric['p3']:>8.2%} {metric['p5']:>8.2%} {metric['p10']:>8.2%} {latency:>10.3f}")
    stats = report['tracker_stats']
    print(f"Raw routes: KF {stats['tier1'] / stats['frames']:.2%}, "
          f"medium {stats['tier2'] / stats['frames']:.2%}, reset/init {stats['tier3'] / stats['frames']:.2%}")
    print(f"Low-tier STNet fusion calls: {stats['small_fusion_calls']}")
    print('STNet is precomputed for every frame; KF route ratio is not a network skip ratio.')
    print('Timing is warm 100-frame processing, divided by frame count; excludes disk/model loading.')
    print(f'JSON report: {output}', flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    views = validate_args(args)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Report already exists; choose a new --output: {output}')
    h5, label_key, shape, _ = validate_validation_h5(args.h5)
    if h5.name != 'test_aug.h5' or shape != (93, 100, 2, 60, 80):
        raise ValueError('Released evaluation requires full 93x100 test_aug.h5')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu for a functional check')
    args.no_graph = args.no_graph or device.type != 'cuda'
    torch.set_num_threads(4)
    torch.manual_seed(20261112)
    model, config, divisor = load_stnet_checkpoint(args.stnet, args.device)
    matcher, _ = load_matcher_checkpoint(args.matcher, args.device)
    with h5py.File(h5, 'r') as handle:
        frames = torch.from_numpy(handle['frames'][:].astype(np.float32))
        labels = handle[label_key][:, :, :2].astype(np.float32)
    trackers = {}

    def run_tracking(prediction):
        results, counts = [], {}
        for begin in range(0, len(frames), args.track_batch):
            image = frames[begin:begin + args.track_batch]
            if len(image) not in trackers:
                trackers[len(image)] = OfflineBatchTracker(matcher, **tracker_options(args))
            raw, stats = trackers[len(image)].track(image, prediction[begin:begin + len(image)])
            results.append(raw)
            for key, value in stats.items():
                if isinstance(value, int):
                    counts[key] = counts.get(key, 0) + value
        if counts['small_fusion_calls'] != 0:
            raise ValueError('Low-tier fusion is forbidden')
        raw = np.concatenate(results)
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite tracker output')
        return raw, counts

    print('Loading complete. Warming up STNet and matcher...', flush=True)
    with torch.inference_mode():
        cache = predict_views(model, frames, views, args.device, divisor,
                              args.batch_size, args.temperature)
        run_tracking(average_views(cache, list(range(len(views))), method=args.ensemble))
        measurements = []
        for repeat in range(args.repeats):
            synchronize(args.device)
            start = time.perf_counter()
            cache = predict_views(model, frames, views, args.device, divisor,
                                  args.batch_size, args.temperature)
            prediction = average_views(cache, list(range(len(views))), method=args.ensemble)
            synchronize(args.device)
            stnet_ms = 1000 * (time.perf_counter() - start)
            raw, stats = run_tracking(prediction)
            filtered = smooth_pixels(raw, args.filter)
            synchronize(args.device)
            total_ms = 1000 * (time.perf_counter() - start)
            # Separate STNet-only filter cost; never insert scoring into the combined timer.
            filter_start = time.perf_counter()
            stnet_filtered = smooth_pixels(prediction, args.filter)
            stnet_filtered_ms = stnet_ms + 1000 * (time.perf_counter() - filter_start)
            metrics = {
                'stnet': summarize_pixel_errors(stnet_filtered / SCALE, labels),
                'ktcal': summarize_pixel_errors(filtered / SCALE, labels),
            }
            if repeat:
                validate_candidate_metrics(measurements[0]['metrics']['ktcal'], metrics['ktcal'], stats)
                if measurements[0]['tracker_stats'] != stats:
                    raise ValueError('Tracker count mismatch between repetitions')
            measurements.append(dict(stnet_ms=stnet_ms, stnet_filtered_ms=stnet_filtered_ms,
                                     total_ms=total_ms, ms_per_frame=total_ms / metrics['ktcal']['frames'],
                                     metrics=metrics, tracker_stats=stats))
            print(f"Repeat {repeat + 1}/{args.repeats}: P3={metrics['ktcal']['p3']:.2%}, "
                  f"{measurements[-1]['ms_per_frame']:.3f} ms/frame", flush=True)
    configuration = dict(vars(args), medium_policy='state', template_refresh=0,
                         reset_velocity_scale=1., tau1=3., tau2=5.)
    report = dict(
        schema=1, protocol='3ET test_aug full-100-frame STNet + KTCAL',
        config=configuration, stnet_config=config, h5_sha256=_sha256(h5),
        stnet_sha256=_sha256(args.stnet), matcher_sha256=_sha256(args.matcher),
        metrics=metrics, tracker_stats=stats, latency=measurements,
        median_ms_per_frame=float(np.median([row['ms_per_frame'] for row in measurements])),
        latency_scope='warm H2D + all STNet views + soft decode + matching + KF + D2H + record filter; excludes disk/model loading/scoring/graph capture',
        validation_selected=True, final_holdout_used=False, labels_used_for_inference=False,
        environment=dict(torch=torch.__version__, numpy=np.__version__,
                         gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print_summary(report, output)
    return report


if __name__ == '__main__':
    main()
