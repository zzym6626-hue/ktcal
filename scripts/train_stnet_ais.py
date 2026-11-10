#!/usr/bin/env python3
"""Train the legacy heatmap STNet on prepared AIS subject splits."""
import argparse

from ktcal_paper.stnet_sequence import train_stnet


def main():
    parser = argparse.ArgumentParser('AIS STNet sequence training')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--window', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--val-stride', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--pixel-weight', type=float, default=0.1)
    parser.add_argument('--pixel-beta', type=float, default=1.0)
    parser.add_argument('--min-delta', type=float, default=0.01)
    parser.add_argument('--gradient-clip', type=float, default=5.0)
    parser.add_argument(
        '--temporal-spatial-kernel', type=int, choices=(1, 3), default=1)
    parser.add_argument(
        '--split-mode', choices=('record', 'random'), default='record',
        help='record: recording/subject-disjoint holdout; '
             'random: random chunk holdout from every recording')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--scheduler', choices=('none', 'cosine'), default='none')
    parser.add_argument('--min-lr', type=float, default=None)
    parser.add_argument('--selection-metric',
                        choices=('last_step', 'all_steps'),
                        default='last_step')
    parser.add_argument('--initialize-from', help='Optional STNet checkpoint for fine-tuning')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='auto')
    args = vars(parser.parse_args())
    args['manifest_path'] = args.pop('manifest')
    print(train_stnet(**args), flush=True)


if __name__ == '__main__':
    main()
