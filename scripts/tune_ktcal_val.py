"""Sweep KTCal accuracy knobs on the validation split.

Every combination writes its own report directory here; holdout is never opened.
Example (from the repository root):

    ./.venv/bin/python -B scripts/tune_ktcal_val.py \
      --checkpoint local_run/ktcal_paper/seq/bidirectional_best.pt \
      --manifest data/ktcal_paper_ais2024/manifest.json \
      --output local_run/ktcal_paper/tuning_val_20261106
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ktcal_paper.experiment import evaluate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['val', 'holdout'], default='val')
    parser.add_argument('--captures', nargs='+', choices=['fine', 'coarse'], default=['fine', 'coarse'])
    parser.add_argument('--min-psr', nargs='+', type=float, default=[None])
    parser.add_argument('--r-px', nargs='+', type=float, default=[2.])
    parser.add_argument('--q-vel', nargs='+', type=float, default=[.5])
    parser.add_argument('--fusion-alpha', nargs='+', type=float, default=[None],
                        help='Accuracy-first blend; 0.5 was best on the first val sweep')
    parser.add_argument('--border-tier', type=int, choices=[2, 3], default=3)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--max-frames', type=int)
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise SystemExit(f'Refusing to overwrite {root}')
    root.mkdir(parents=True)
    rows = []
    for capture, psr, r_px, q_vel, alpha in itertools.product(
            args.captures, args.min_psr, args.r_px, args.q_vel, args.fusion_alpha):
        name = f'{capture}_psr{psr}_r{r_px}_qv{q_vel}_a{alpha}'
        report = evaluate(args.checkpoint, args.manifest, root / name, split=args.split,
                          mode='ktcal', device=args.device, threads=args.threads,
                          capture=capture, min_psr=psr, r_px=r_px, q_vel=q_vel,
                          border_tier=args.border_tier, fusion_alpha=alpha, realtime=False,
                          max_frames=args.max_frames, allow_partial=args.allow_partial)
        payload = json.loads(Path(report).read_text())
        row = payload['rows'][0]
        rows.append({'run': name, 'capture': capture, 'min_psr': psr, 'r_px': r_px,
                     'q_vel': q_vel, 'fusion_alpha': alpha, 'border_tier': args.border_tier,
                     'mean_error_px': row['mean_error_px'], 'p3': row['p3'],
                     'p5': row['p5'], 'p10': row['p10'],
                     'trigger_ratio': row['trigger_ratio'],
                     'online_latency_ms': row['online_latency_ms'],
                     'frame_deadline_misses': row['frame_deadline_misses']})
        print(json.dumps(rows[-1]), flush=True)
    with (root / 'sweep.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = min(rows, key=lambda row: row['mean_error_px'])
    print(json.dumps({'best': best}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
