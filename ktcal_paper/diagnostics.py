"""Train/validation diagnostics, separate from holdout and deployment timing."""
import os
import hashlib
import io
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import WindowDataset, load_manifest
from .experiment import configure, metrics, sha256, write_json
from .model import PaperSTNet


@torch.inference_mode()
def diagnose(checkpoint, manifest_path, output, split='val', stride=16,
             batch_size=16, threads=4, device='auto', plot=False):
    if split not in ('train', 'val'):
        raise ValueError('Diagnostic model selection is restricted to train/val')
    device=configure(42,threads,device)
    path=Path(manifest_path).resolve(); manifest=load_manifest(path)
    checkpoint_bytes=Path(checkpoint).read_bytes()
    checkpoint_hash=hashlib.sha256(checkpoint_bytes).hexdigest()
    state=torch.load(io.BytesIO(checkpoint_bytes),map_location=device,weights_only=False)
    del checkpoint_bytes
    if state.get('schema')!=1 or sha256(path)!=state['manifest_sha256']:
        raise ValueError('Checkpoint and diagnostic dataset manifest differ')
    model=PaperSTNet(**state['model_config']).to(device).eval()
    model.load_state_dict(state['model_state'])
    out=Path(output).resolve(); out.mkdir(parents=True,exist_ok=False)
    dataset=WindowDataset(path,split,window=state['config']['window'],stride=stride)
    predicted=[]; forward=[]; labels=[]
    try:
        for frames,y in DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0):
            a,b=model(frames.to(device),return_forward=True)
            predicted.append(a.cpu().numpy()*[80.,60.])
            forward.append(b.cpu().numpy()*[80.,60.])
            labels.append(y[:,-1,:3].numpy())
        if not labels: raise ValueError('No valid diagnostic windows')
        names=np.array([name for name,end in dataset.indices])
        ends=np.array([end for name,end in dataset.indices])
    finally:
        dataset.close()
    predicted=np.concatenate(predicted); forward=np.concatenate(forward)
    labels=np.concatenate(labels); truth=labels[:,:2]*[80.,60.]; flags=labels[:,2]
    with h5py.File(path.parent/manifest['h5'],'r') as data:
        train_labels=np.concatenate([data[name]['labels'][:][data[name]['valid'][:],:2]
                                     for name in manifest['splits']['train']])*[80.,60.]
    center=train_labels.mean(0)

    def describe(values):
        return {'overall':metrics(values,truth),
            'bias_xy':(values-truth).mean(0).tolist(), 'prediction_std_xy':values.std(0).tolist(),
            'by_record':{name:metrics(values[names==name],truth[names==name]) for name in np.unique(names)},
            'by_original_state':{str(int(flag)):metrics(values[flags==flag],truth[flags==flag])
                                 for flag in np.unique(flags)}}

    report={'schema':1,'kind':'train-validation-diagnostic','dataset':manifest['dataset'],
        'split':split,'stride':stride,'window':state['config']['window'],'batch_size':batch_size,
        'checkpoint_sha256':checkpoint_hash,'manifest_sha256':sha256(path),
        'checkpoint_phase':state['phase'],'checkpoint_epoch':state['epoch'],
        'model_config':state['model_config'],'inference_mode':'forward' if model.deployed else 'bidirectional',
        'device':str(device),'coordinate_scale':[80,60],
        'primary':describe(predicted),'forward':describe(forward),
        'target_mean_xy':truth.mean(0).tolist(),'target_std_xy':truth.std(0).tolist(),
        'train_mean_baseline':metrics(np.tile(center,(len(truth),1)),truth),
        'holdout_used':False,'deployment_timing_measured':False,'paper_results_reproduced':False}
    np.savez_compressed(out/'predictions.npz',primary=predicted,forward=forward,truth=truth,
                        recording=names,frame=ends,original_state=flags)
    if plot:
        # Optional plotting dependency; caches stay inside the run directory.
        os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        recordings=np.unique(names)
        fig,axes=plt.subplots(len(recordings),3,figsize=(12,3*len(recordings)),squeeze=False)
        with h5py.File(path.parent/manifest['h5'],'r') as data:
            for row,name in enumerate(recordings):
                indices=np.flatnonzero(names==name)
                for col,index in enumerate(indices[np.linspace(0,len(indices)-1,3).astype(int)]):
                    frame=data[name]['frames'][ends[index]].astype(float).sum(0)
                    ax=axes[row,col]; ax.imshow(frame,cmap='gray',vmin=0,vmax=30)
                    ax.scatter(*truth[index],color='lime',marker='+',s=90,label='Label')
                    ax.scatter(*predicted[index],color='orange',marker='x',s=60,label='Primary')
                    if model.forward_head is not None:
                        ax.scatter(*forward[index],color='cyan',marker='x',s=50,label='Forward branch')
                    ax.set_title(f'{name} frame {ends[index]}'); ax.set_xlim(0,80); ax.set_ylim(60,0)
        axes[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'overlays.png',dpi=150); plt.close(fig)
    write_json(out/'report.json',report)
    return out/'report.json'
