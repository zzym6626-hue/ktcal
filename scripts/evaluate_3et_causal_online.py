#!/usr/bin/env python3
"""Evaluate on-demand causal-16 STNet + no-fusion KTCAL on 3ET validation."""

import argparse
import json

from ktcal_paper.combined_3et import evaluate_online_causal_3et


def build_parser():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--h5', required=True)
    parser.add_argument('--stnet-checkpoint', required=True)
    parser.add_argument('--matcher-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-prediction-variance-px2', type=float, default=10.)
    parser.add_argument('--max-consecutive-prediction-skips', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    return parser


def main():
    args = build_parser().parse_args()
    report = evaluate_online_causal_3et(
        args.h5, args.stnet_checkpoint, args.matcher_checkpoint, args.output,
        max_prediction_variance_px2=args.max_prediction_variance_px2,
        max_consecutive_prediction_skips=args.max_consecutive_prediction_skips,
        device=args.device, threads=args.threads)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
