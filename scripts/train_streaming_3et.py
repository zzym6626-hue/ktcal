#!/usr/bin/env python3
"""Train a causal streaming STNet; offline teachers are training-only."""
import argparse
import json

from ktcal_paper.streaming_training import train_streaming_3et, build_teacher_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-h5', required=True)
    parser.add_argument('--val-h5')
    parser.add_argument('--output', required=True)
    parser.add_argument('--initialize-from')
    parser.add_argument('--teacher-cache')
    parser.add_argument('--teacher-weight', type=float, default=0.)
    parser.add_argument('--teacher-max-error-px', type=float, default=5.)
    parser.add_argument('--build-teacher-from', help='Only cache offline targets on train_aug and exit')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--hidden-size', type=int, choices=(128, 256), default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--sigma', type=float, default=1.)
    parser.add_argument('--pixel-loss-weight', type=float, default=.1)
    parser.add_argument('--decode-temperature', type=float, default=1.)
    parser.add_argument('--persistent-boost', type=float, default=0.)
    parser.add_argument('--persistent-min-run', type=int, default=5)
    parser.add_argument('--cold-boost', type=float, default=0.)
    parser.add_argument('--cold-frames', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=.2)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--rotation-degrees', type=float, default=0.)
    parser.add_argument('--scale-jitter', type=float, default=0.)
    parser.add_argument('--count-gain-jitter', type=float, default=0.)
    parser.add_argument('--spatial-localizer', action='store_true')
    parser.add_argument('--spatial-loss-weight', type=float, default=.1)
    parser.add_argument('--current-readout', action='store_true')
    parser.add_argument('--raw-current-readout', action='store_true')
    parser.add_argument('--raw-current-loss-weight', type=float, default=.25)
    parser.add_argument('--memory-readout', action='store_true')
    parser.add_argument('--memory-window', type=int, default=8)
    parser.add_argument('--memory-loss-weight', type=float, default=.25)
    parser.add_argument('--learned-initial-state', action='store_true')
    parser.add_argument('--current-loss-weight', type=float, default=.25)
    parser.add_argument('--warm-start-steps', type=int, default=0)
    parser.add_argument('--flip-axis-px', type=float, default=79.)
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--gradient-clip', type=float, default=5.)
    args = vars(parser.parse_args())
    teacher = args.pop('build_teacher_from')
    if teacher:
        import torch
        torch.set_num_threads(args['threads'])
        result = build_teacher_cache(args['train_h5'], teacher, args['output'],
                                     batch_size=args['batch_size'],
                                     device=args['device'] if args['device'] != 'auto' else 'cuda')
    else:
        if args['val_h5'] is None:
            parser.error('--val-h5 is required for training')
        result = train_streaming_3et(**args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
