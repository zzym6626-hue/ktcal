"""Independent exact ordered smoke coverage, pixel metrics, calls and artifact hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--h5', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, h5 = Path(args.root), Path(args.h5).resolve()
    if h5.name != 'test_aug.h5':
        raise ValueError('smoke verification only on explicit test_aug.h5')
    results = []
    for name in ('six_sim_smoke100_v1','seven_sim_smoke100_v1'):
        directory = root/name
        report = json.loads((directory/'report.json').read_text())
        if (report['records'] != 1 or report['complete'] or not report['completed']
                or not report['simulation_only'] or report['board_verified']
                or report['simulator_calls'] != [1000]*6 or report['cpu_tail_calls'] != 10
                or report['tracker_stats']['small_fusion_calls'] != 0
                or not report['low_global_invariant_all_records']
                or not report['raw_transforms_byte_equal_all_records']
                or report['h5_sha256'] != sha(h5)):
            raise ValueError('invalid smoke scope or call/gate coverage')
        csv_file = directory/'predictions.csv'
        if sha(csv_file) != report['csv_sha256']:
            raise ValueError('CSV hash mismatch')
        with csv_file.open() as file:
            rows = list(csv.DictReader(file))
        if [(int(r['record']),int(r['frame'])) for r in rows] != [(0,i) for i in range(100)]:
            raise ValueError('ordered frame coverage differs')
        with h5py.File(h5,'r') as handle:
            if handle['frames'].shape != (93,100,2,60,80):
                raise ValueError('validation shape differs')
            key = 'label' if 'label' in handle else 'labels'
            labels = handle[key][0,:100,:2].astype(np.float32).astype(np.float64)*np.array([80.,60.])
        metrics = {}
        for kind,fields in [('ktcal',('x_px','y_px')),('stnet',('stnet_x_px','stnet_y_px'))]:
            pixels = np.array([[float(row[x]) for x in fields] for row in rows])
            if not np.isfinite(pixels).all():
                raise ValueError('nonfinite output')
            error = np.linalg.norm(pixels-labels,axis=1)
            actual = dict(frames=100,mean_error_px=float(error.mean()),
                **{f'p{k}':float(np.mean(error<k)) for k in (3,5,10)})
            expected = report['metrics'][kind]
            for key in actual:
                if abs(actual[key]-expected[key]) > 1e-12:
                    raise ValueError(f'independent score mismatch {kind}/{key}')
            metrics[kind]=actual
        tiers=np.array([int(row['tier']) for row in rows])
        for i in (1,2,3):
            if int(np.sum(tiers==i)) != report['tracker_stats'][f'tier{i}']:
                raise ValueError('tier count mismatch')
        files={}
        for manifest in directory.glob('*simulation_manifest.json'):
            item=json.loads(manifest.read_text())
            model_name='readout100' if manifest.name=='readout_simulation_manifest.json' else manifest.name.replace('_simulation_manifest.json','')
            model_file=directory/(model_name+'.rknn')
            if sha(model_file)!=item['rknn_sha256']:
                raise ValueError('RKNN artifact hash mismatch')
            files[model_name]=dict(sha256=sha(model_file),bytes=model_file.stat().st_size)
        if len(files)!=(7 if name.startswith('seven') else 6):
            raise ValueError('missing executable artifact')
        results.append(dict(name=name,passed=True,frames=100,metrics=metrics,artifacts=files))
    with Path(args.output).open('x') as file:
        json.dump(dict(passed=True,smoke_only=True,board_verified=False,results=results),file,indent=2)
    print('PASS: 2 complete 100-frame smoke chains, independent metrics/calls/hashes, no low fusion')


if __name__=='__main__':
    main()
