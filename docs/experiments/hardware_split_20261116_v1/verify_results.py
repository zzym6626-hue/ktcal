"""Independent ordered pixel-CSV scoring and threshold checks for split spike."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--h5', required=True)
    parser.add_argument('--floating-dir', required=True)
    parser.add_argument('--int8-dir', required=True)
    parser.add_argument('--published-report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    h5 = Path(args.h5)
    if h5.name != 'test_aug.h5':
        raise ValueError('test_aug only; never final holdout')
    floating, int8 = Path(args.floating_dir), Path(args.int8_dir)
    fp_report = json.loads((floating / 'report.json').read_text())
    q_report = json.loads((int8 / 'report.json').read_text())
    published = json.loads(Path(args.published_report).read_text())
    if (not fp_report['passed'] or not fp_report['complete'] or not q_report['completed'] or
            not q_report['complete'] or q_report['board_verified'] or not q_report['simulation_only']):
        raise ValueError('complete FP and simulator reports required')
    if sha(h5) != fp_report['h5_sha256'] or sha(h5) != q_report['h5_sha256']:
        raise ValueError('data hashes differ')
    def read(path):
        with path.open() as file:
            rows = list(csv.DictReader(file))
        if len(rows) != 9300 or [(int(v['record']), int(v['frame'])) for v in rows] != [(r, t) for r in range(93) for t in range(100)]:
            raise ValueError('must contain each of 9300 ordered frames exactly once')
        return rows
    fp_rows, q_rows = read(floating / 'predictions.csv'), read(int8 / 'predictions.csv')
    if sha(int8 / 'predictions.csv') != q_report['csv_sha256']:
        raise ValueError('INT8 CSV hash mismatch')
    with h5py.File(h5, 'r') as handle:
        key = 'label' if 'label' in handle else 'labels'
        if handle['frames'].shape != (93, 100, 2, 60, 80):
            raise ValueError('released shape required')
        labels = np.asarray(handle[key][:, :, :2], dtype=np.float32).reshape(9300, 2).astype(np.float64) * [80., 60.]
    definitions = [('reference_stnet', fp_rows, 'pytorch_stnet_x_px', 'pytorch_stnet_y_px'),
                   ('split_fp_stnet', fp_rows, 'stnet_x_px', 'stnet_y_px'),
                   ('reference_ktcal', fp_rows, 'pytorch_x_px', 'pytorch_y_px'),
                   ('split_fp_ktcal', fp_rows, 'x_px', 'y_px'),
                   ('split_int8_stnet', q_rows, 'stnet_x_px', 'stnet_y_px'),
                   ('split_int8_ktcal', q_rows, 'x_px', 'y_px')]
    metrics, errors = {}, {}
    for name, rows, x, y in definitions:
        values = np.array([[float(row[x]), float(row[y])] for row in rows])
        if not np.isfinite(values).all():
            raise ValueError('nonfinite predicted pixels')
        error = np.linalg.norm(values - labels, axis=-1)
        errors[name] = error
        metrics[name] = dict(frames=9300, mean_error_px=float(error.mean()),
                             **{f'p{k}':float(np.mean(error < k)) for k in [3, 5, 10]})
        if name.startswith('split_int8_'):
            expected = q_report['metrics'][name[len('split_int8_'):]]
        else:
            source = 'pytorch_' if name.startswith('reference_') else 'onnx_cpp_'
            kind = name.rsplit('_', 1)[-1]
            expected = fp_report['metrics'][source + kind + '_ma7']
        for k in ['p3', 'p5', 'p10']:
            if metrics[name][k] != expected[k]:
                raise ValueError('independent classification disagreement ' + name + ' ' + k)
        if abs(metrics[name]['mean_error_px'] - expected['mean_error_px']) > 1e-7:
            raise ValueError('independent mean disagreement ' + name)
    mismatch = {}
    deltas = {}
    for kind in ['stnet', 'ktcal']:
        mismatch[kind] = {f'p{k}':int(np.count_nonzero((errors['reference_' + kind] < k) !=
                                                    (errors['split_fp_' + kind] < k))) for k in [3, 5, 10]}
        if any(mismatch[kind].values()):
            raise ValueError('floating individual classification parity failed')
        q = metrics['split_int8_' + kind]
        deltas[kind] = dict(vs_same_cpu={k:q[k] - metrics['reference_' + kind][k]
                                        for k in ['p3', 'p5', 'p10', 'mean_error_px']},
                            vs_published_cuda={k:q[k] - published['metrics'][kind][k]
                                               for k in ['p3', 'p5', 'p10', 'mean_error_px']})
    for data, report in [(fp_rows, fp_report), (q_rows, q_report)]:
        counts = {f'tier{k}':sum(int(row['tier']) == k for row in data) for k in [1, 2, 3]}
        if sum(counts.values()) != 9300 or any(report['tracker_stats'][k] != count for k, count in counts.items()):
            raise ValueError('tier counts disagree')
        if report['tracker_stats']['small_fusion_calls'] != 0:
            raise ValueError('forbidden low global fusion')
    for index in range(1, 4):
        stage = json.loads((int8 / f'spatial{index}_simulation_manifest.json').read_text())
        if sha(int8 / f'spatial{index}.rknn') != stage['rknn_sha256']:
            raise ValueError('simulated RKNN hash mismatch')
    if q_report['simulator_calls'] != [93000, 93000, 93000]:
        raise ValueError('incomplete per-frame per-stage per-view simulator calls')
    report = dict(schema=1, passed=True, independently_scored_metrics=metrics,
                  floating_individual_classification_mismatches=mismatch,
                  int8_simulation_deltas=deltas, floating_report_sha256=sha(floating / 'report.json'),
                  int8_report_sha256=sha(int8 / 'report.json'), floating_csv_sha256=sha(floating / 'predictions.csv'),
                  int8_csv_sha256=sha(int8 / 'predictions.csv'), h5_sha256=sha(h5),
                  float_and_int8_low_global_fusion_calls=0, simulation_only=True, board_verified=False,
                  hardware_latency_measured=False)
    with Path(args.output).open('x') as file:
        json.dump(report, file, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
