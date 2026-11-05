"""Collect only completed, compatible software evaluations into a measured table."""
import argparse
import csv
import json
from pathlib import Path


def collect(root):
    root=Path(root)
    reports=sorted(root.glob('eval_*/report.json'))
    if not reports: raise ValueError('No completed evaluation reports')
    rows=[]; fingerprints=set(); devices=set(); timings=set(); coverage=set()
    for path in reports:
        report=json.loads(path.read_text())
        fingerprints.add(report['manifest_sha256'])
        for row in report['rows']:
            devices.add(row['device']); timings.add(row['timing_mode'])
            coverage.add((row['split'],row['input_frames'],row['max_frames_per_record']))
            rows.append({'experiment':path.parent.name,**row,'inference_mode':row.get('inference_mode','forward'),
                         'readout':row.get('readout','last')})
    if len(fingerprints)!=1 or len(devices)!=1 or len(timings)!=1 or len(coverage)!=1:
        raise ValueError('Do not mix dataset splits, coverage, devices or timing modes in a comparison table')
    with (root/'measured_ablation.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines=['# Measured software ablations','',
        '| Experiment | Architecture | Inference | Readout | Mode | p3 | p5 | p10 | Mean error px | Online ms |',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["experiment"]} | {r["architecture"]} | {r["inference_mode"]} | {r["readout"]} | {r["variant"]} | '
            f'{r["p3"]:.4f} | {r["p5"]:.4f} | {r["p10"]:.4f} | {r["mean_error_px"]:.3f} | {r["online_latency_ms"]:.3f} |')
    lines+=['',f'Dataset: {rows[0]["dataset"]}; device: {rows[0]["device"]}; timing: {rows[0]["timing_mode"]}.',
        'All rows above come from completed runs. No unexecuted paper row is filled in.']
    with (root/'measured_ablation.md').open('x') as f: f.write('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root')
    collect(parser.parse_args().root)
