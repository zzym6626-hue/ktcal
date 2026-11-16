"""Conversion preflight only: deterministic raw train records, never labels."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from ktcal_paper.combined_3et import _sha256
from ktcal_paper.offline_experiment import spatial_view

base = Path('/root/ktcal_hardware_preflight_20261116_v1/artifacts')
output = base / 'train_calibration_v1'
output.mkdir(exist_ok=False)
source = Path('/root/autodl-tmp/TDTracker-main/TDTracker-main/3ET+2025/train_aug.h5')
if source.name != 'train_aug.h5':
    raise ValueError('training frames only')
entries = []
with h5py.File(source, 'r') as handle:
    frames = handle['frames']
    if frames.shape[1:] != (100,2,60,80):
        raise ValueError('fixed 100-frame train input required')
    shape = list(frames.shape)
    for record in range(8):
        raw = np.asarray(frames[record:record+1],dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError('finite training counts required')
        for flip in [False, True]:
            path = output / f'train_r{record:04d}_flip{int(flip)}.npy'
            values = spatial_view(torch.from_numpy(raw),(0,0,flip)).contiguous().numpy()
            with path.open('xb') as f:
                np.save(f,values,allow_pickle=False)
            entries.append(dict(record=record,flip=flip,path=str(path),sha256=_sha256(path),shape=list(values.shape)))
dataset = output / 'train_inputs.txt'
with dataset.open('x') as f:
    f.write(''.join(e['path']+'\n' for e in entries))
report = dict(schema=1,purpose='quantized conversion preflight; not accuracy-qualified calibration',
              source=str(source),source_h5_sha256=_sha256(source),source_shape=shape,
              keys_read=['frames'],records_read=list(range(8)),samples=entries,
              raw_counts=True,normalization_applied=False,labels_read=False,validation_used=False,
              dataset=str(dataset),dataset_sha256=_sha256(dataset))
with (output/'provenance.json').open('x') as f:
    json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))
