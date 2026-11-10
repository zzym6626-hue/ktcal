#!/usr/bin/env python3
"""One-shot final evaluation with the validation-frozen KTCAL configuration."""

import argparse
import json

from ktcal_paper.official_holdout import evaluate_ktcal_official_holdout


def main():
    parser = build_parser()
    args = parser.parse_args()
    report = evaluate_ktcal_official_holdout(
        args.h5, args.stnet_checkpoint, args.matcher_checkpoint, args.output,
        stnet_reference=args.stnet_reference, events=args.events,
        batch_size=args.batch_size, threads=args.threads, device=args.device,
        stnet_decode=args.stnet_decode, offline_filter=args.offline_filter,
    )
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--h5', required=True)
    parser.add_argument('--stnet-checkpoint', required=True)
    parser.add_argument('--matcher-checkpoint', required=True)
    parser.add_argument('--stnet-reference')
    parser.add_argument('--events', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stnet-decode', choices=('hard', 'soft'),
                        default='hard')
    parser.add_argument('--offline-filter',
                        choices=('none', 'ma3', 'ma5', 'ma7', 'ma9',
                                 'ma11', 'ma13', 'ma15'),
                        default='none')
    return parser


if __name__ == '__main__':
    main()
