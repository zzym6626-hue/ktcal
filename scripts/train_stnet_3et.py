#!/usr/bin/env python3
"""Train or fine-tune STNet on labeled 3ET+2025 train/validation HDF5."""

import argparse

from ktcal_paper.stnet_sequence import train_stnet_3et


def build_parser():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--train-h5', required=True)
    parser.add_argument('--val-h5', required=True)
    parser.add_argument('--initialize-from', help='Optional STNet checkpoint for fine-tuning')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--readout', choices=('last', 'final_hidden', 'attention'),
                        default='last',
                        help='last, final_hidden, or parameter-free temporal attention')
    parser.add_argument('--selection-metric',
                        choices=('hard_argmax', 'soft_ma11', 'offline_soft_ma11'),
                        metavar='{hard_argmax,soft_ma11}', default='hard_argmax',
                        help='soft_ma11: select by per-record soft decode + MA11 P3')
    parser.add_argument('--persistent-boost', type=float, default=0.,
                        help='Train-only weight added to runs of >=5 frames with >=3px error')
    parser.add_argument('--sparse-boost', type=float, default=0.,
                        help='Train-only weight added to per-record bottom-quartile event frames')
    parser.add_argument('--causal-last-step', action='store_true',
                        help='Train and validate current-frame predictions using historical windows')
    parser.add_argument('--train-stride', type=int, default=8,
                        help='Target-frame stride for causal training; validation always uses stride 1')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--min-lr', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.set_defaults(distill_weight=0., feature_distill_weight=0.)
    parser.add_argument('--min-delta', type=float, default=0.001)
    parser.add_argument('--gradient-clip', type=float, default=5.0)
    parser.add_argument(
        '--temporal-spatial-kernel', type=int, choices=(1, 3), default=1)
    parser.add_argument('--input-divisor', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='auto')
    return parser


def main():
    args = vars(build_parser().parse_args())
    if args['selection_metric'] == 'soft_ma11':
        args['selection_metric'] = 'offline_soft_ma11'
    print(train_stnet_3et(**args), flush=True)


if __name__ == '__main__':
    main()
