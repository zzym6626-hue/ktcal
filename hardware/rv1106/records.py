"""Prepare raw board records and score its CSV (P3/P5/P10), without fitting."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from ktcal_paper.combined_3et import SCALE, _sha256, validate_validation_h5
from ktcal_paper.evaluation_protocol import summarize_pixel_errors


def _validate(path):
    h5, key, shape, _ = validate_validation_h5(path)
    if shape[1] != 100:
        raise ValueError('board input requires independent 100-frame records')
    return h5, key, shape


def export_records(h5_path, output):
    h5, _, shape = _validate(h5_path)
    output = Path(output)
    sidecar = output.with_suffix(output.suffix + '.json')
    if output.exists() or sidecar.exists():
        raise FileExistsError(f'refusing to overwrite {output} or its metadata')
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5, 'r') as handle, output.open('xb') as target:
        for record in handle['frames']:
            record = np.asarray(record, dtype='<f4')
            if not np.isfinite(record).all():
                raise ValueError('non-finite event input')
            target.write(record.tobytes(order='C'))
    metadata = dict(schema=1, h5_sha256=_sha256(h5), binary_sha256=_sha256(output),
                    shape=list(shape), format='little-endian float32 N,T,C,H,W raw counts')
    with sidecar.open('x') as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def score_records(h5_path, predictions, output):
    h5, key, shape = _validate(h5_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    result = np.full(shape[:2] + (2,), np.nan, dtype=np.float64)
    seen = np.zeros(shape[:2], dtype=bool)
    tiers = {1: 0, 2: 0, 3: 0}
    with Path(predictions).open() as handle:
        for row in csv.DictReader(handle):
            record, frame = int(row['record']), int(row['frame'])
            if not (0 <= record < shape[0] and 0 <= frame < 100) or seen[record, frame]:
                raise ValueError('duplicate or out-of-range board prediction')
            seen[record, frame] = True
            result[record, frame] = float(row['x_px']), float(row['y_px'])
            tier = int(row['tier'])
            if tier not in tiers:
                raise ValueError('invalid board tier')
            tiers[tier] += 1
    if not seen.all() or not np.isfinite(result).all():
        raise ValueError('finite predictions required for every frame of every record')
    with h5py.File(h5, 'r') as handle:
        labels = handle[key][:, :, :2].astype(np.float32)
    report = dict(schema=1, h5_sha256=_sha256(h5), predictions_sha256=_sha256(predictions),
                  metrics=summarize_pixel_errors(result / SCALE, labels),
                  tier_counts=tiers, timing_included=False,
                  note='scores supplied board CSV; does not certify model provenance or NPU execution')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--h5', default='data/3et/test_aug.h5')
    export.add_argument('--output', required=True)
    score = sub.add_parser('score')
    score.add_argument('--h5', default='data/3et/test_aug.h5')
    score.add_argument('--predictions', required=True)
    score.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'export':
        export_records(args.h5, args.output)
    else:
        score_records(args.h5, args.predictions, args.output)


if __name__ == '__main__':
    main()
