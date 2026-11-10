#!/usr/bin/env python3
"""Evaluate fine-tuned STNet + matcher + KTCAL on 3ET test_aug only."""

import argparse

from ktcal_paper.combined_3et import evaluate_combined_3et


def build_parser():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--h5', required=True, help='3ET test_aug.h5')
    parser.add_argument('--stnet-checkpoint', required=True)
    parser.add_argument('--matcher-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fusion-alpha', type=float, default=None)
    parser.add_argument('--max-prediction-variance-px2', type=float, default=None,
                        help='Enable the pre-NCC predictor skip; disabled by default')
    parser.add_argument('--max-consecutive-prediction-skips', type=int, default=1)
    parser.add_argument('--stnet-decode', choices=('hard', 'soft'), default='hard')
    parser.add_argument('--smoothing', dest='offline_filter', default='none',
                       choices=('none', 'ma3', 'ma5', 'ma7', 'ma9',
                                'ma11', 'ma13', 'ma15'),
                       help='Per-sequence moving-average postprocess')
    parser.add_argument('--offline-filter', dest='offline_filter',
                        default=argparse.SUPPRESS,
                        choices=('none', 'ma3', 'ma5', 'ma7', 'ma9',
                                 'ma11', 'ma13', 'ma15'),
                        help=argparse.SUPPRESS)
    return parser


def main():
    args = build_parser().parse_args()
    report = evaluate_combined_3et(
        args.h5, args.stnet_checkpoint, args.matcher_checkpoint, args.output,
        batch_size=args.batch_size, threads=args.threads, device=args.device,
        fusion_alpha=args.fusion_alpha,
        max_prediction_variance_px2=args.max_prediction_variance_px2,
        max_consecutive_prediction_skips=args.max_consecutive_prediction_skips,
        stnet_decode=args.stnet_decode,
        offline_filter=args.offline_filter)
    print('3ET validation: 100-frame STNet + KTCAL')
    print(f"{'Method':<29} {'Frames':>7} {'Mean(px)':>10} "
          f"{'P3':>8} {'P5':>8} {'P10':>8} {'ms/frame':>10}")
    names = {
        'stnet_only': 'STNet',
        'stnet_offline': 'STNet + MA',
        'ktcal_online': 'KTCAL raw',
        'ktcal_retrospective': 'KTCAL replay',
        'ktcal_offline': 'STNet + KTCAL + filter',
    }
    for key, metric in report['metrics'].items():
        latency = report['latency']['ms_per_frame'][key]
        print(f"{names.get(key, key):<29} {metric['frames']:>7} "
              f"{metric['mean_error_px']:>10.3f} "
              f"{metric['p3']:>7.2%} {metric['p5']:>7.2%} "
              f"{metric['p10']:>7.2%} {latency:>10.3f}")
    print(f'JSON report: {args.output}', flush=True)
    print('ms/frame measures batched 100-frame sequence processing.', flush=True)


if __name__ == '__main__':
    main()
