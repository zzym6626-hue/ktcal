#!/usr/bin/env python3
"""Audit per-frame online KTCAL gate errors on 3ET validation only."""

import argparse
import json

from ktcal_paper.gate_diagnostics import diagnose_online_gate


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--h5', required=True)
    parser.add_argument('--stnet-checkpoint', required=True)
    parser.add_argument('--matcher-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    result = diagnose_online_gate(
        args.h5, args.stnet_checkpoint, args.matcher_checkpoint, args.output,
        device=args.device, threads=args.threads)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
