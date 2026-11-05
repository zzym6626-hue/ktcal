"""AIS 2024 labeled records, subject splits, and lazy causal windows.

Frames are losslessly stored as four-bit BinaRep integers (0..15). Original
label states are retained. Every finite in-sensor, nonzero coordinate is scored
regardless of state; no repeat-coordinate blink heuristic is applied.
"""
import ast
import hashlib
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def record_key(name):
    return tuple(int(v) for v in name.split('_'))


def subject_split(recordings, seed=42):
    subjects=sorted({record_key(n)[0] for n in recordings})
    if len(subjects)<3:
        raise ValueError('At least three subjects are required for independent splits')
    order=np.random.default_rng(seed).permutation(subjects)
    n=max(1,round(len(order)*.15))
    assignment={'holdout':set(order[:n]),'val':set(order[n:2*n]),'train':set(order[2*n:])}
    return {k:sorted([r for r in recordings if record_key(r)[0] in v],key=record_key)
            for k,v in assignment.items()}


def read_labels(path):
    path=Path(path)
    if path.name != 'label.txt':
        raise ValueError('Training requires label.txt, never test placeholder labels')
    raw=np.asarray([ast.literal_eval(s) for s in path.read_text().splitlines() if s.strip()],dtype=np.float32)
    if raw.ndim!=2 or raw.shape[1]!=3 or len(raw)==0:
        raise ValueError(f'Malformed labels: {path}')
    if np.all(raw==0):
        raise ValueError(f'Refusing all-zero placeholder labels: {path}')
    valid=np.isfinite(raw).all(1)&(raw[:,0]>=0)&(raw[:,0]<640)&(raw[:,1]>=0)&(raw[:,1]<480)
    valid &= np.any(raw[:,:2]!=0,axis=1)
    if not valid.any(): raise ValueError(f'No valid coordinate labels: {path}')
    labels=raw.copy(); labels[:,:2]/=np.array([640.,480.],dtype=np.float32)
    return labels,valid


def prepare_ais(source, output, seed=42):
    # Lazy import avoids Tonic/Numba initialization in training/evaluation.
    from tonic.transforms import Compose,ToFrame,ToBinaRep
    source,output=Path(source).resolve(),Path(output).resolve()
    raw_root=source/'event_data/event_data/train'
    files=sorted(raw_root.glob('*/*.h5'),key=lambda p:record_key(p.parent.name))
    if not files: raise ValueError(f'No AIS train recordings under {raw_root}')
    if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    # Validate labels before allocating output storage; test folders cannot enter.
    labels_by_name={p.parent.name:read_labels(p.parent/'label.txt') for p in files}
    splits=subject_split(list(labels_by_name),seed)
    output.mkdir(parents=True)
    transform=Compose([ToFrame(sensor_size=(80,60,2),n_time_bins=4),ToBinaRep(n_frames=1,n_bits=4)])
    records=[]; started=time.perf_counter()
    with h5py.File(output/'frames.h5.partial','w') as out:
        for number,path in enumerate(files,1):
            name=path.parent.name; labels,valid=labels_by_name[name]
            with h5py.File(path,'r') as f: events=f['events'][:]
            if len(events)==0: raise ValueError(f'Empty events: {name}')
            ts=events['t']
            if np.any(ts[1:]<ts[:-1]): raise ValueError(f'Unsorted timestamps: {name}')
            if np.any(events['x']>=640) or np.any(events['y']>=480) or np.any(events['p']>1):
                raise ValueError(f'Invalid event coordinates/polarity: {name}')
            start,end=int(ts[0]),int(ts[-1])
            # Match the archive loader's [start,end) bin-start convention.
            event_bins=(end-start+9999)//10000
            count=min(len(labels),event_bins)
            if count<=0: raise ValueError(f'No aligned frames: {name}')
            group=out.create_group(name)
            frames=group.create_dataset('frames',(count,2,60,80),dtype='uint8',
                chunks=(min(16,count),2,60,80),compression='gzip',compression_opts=1)
            group['labels']=labels[:count]
            group['valid']=valid[:count]
            group['bin_start_us']=start+np.arange(count,dtype=np.int64)*10000
            for i in range(count):
                left=start+i*10000; right=left+10000
                a=np.searchsorted(ts,left,side='left')
                b=np.searchsorted(ts,min(right,end),side='right' if right>end else 'left')
                small=events[a:b]
                if len(small)==0:
                    frames[i]=np.zeros((2,60,80),dtype=np.uint8)
                    continue
                converted=np.empty(len(small),dtype=[('x','i8'),('y','i8'),('t','i8'),('p','i8')])
                converted['x']=small['x']//8; converted['y']=small['y']//8
                converted['t']=small['t']; converted['p']=small['p']
                frame=transform(converted)[0]
                encoded=np.rint(frame*15).astype(np.uint8)
                if not np.isfinite(frame).all() or not np.allclose(encoded/15,frame,atol=1e-6):
                    raise ValueError(f'Unexpected BinaRep values: {name}/{i}')
                frames[i]=encoded
            records.append({'recording':name,'subject':record_key(name)[0],
                'source':str(path),'source_bytes':path.stat().st_size,
                'labels_sha256':hashlib.sha256((path.parent/'label.txt').read_bytes()).hexdigest(),
                'frames':count,'valid_labels':int(valid[:count].sum()),
                'label_count':len(labels),'event_bins':event_bins,
                'omitted_trailing_labels':len(labels)-count,'omitted_event_bins':event_bins-count,
                'event_timestamp_bounds_us':[start,end],
                'state_counts':{str(int(s)):int((labels[:count,2]==s).sum()) for s in np.unique(labels[:count,2])}})
            print(f'Prepared {number}/{len(files)} {name}: {count} frames, {int(valid[:count].sum())} labels',flush=True)
            del events
        out.attrs['completed']=True
    (output/'frames.h5.partial').rename(output/'frames.h5')
    manifest={'schema':1,'dataset':'3ET+ AIS 2024','source':str(source),'h5':'frames.h5',
        'labels_are_placeholders':False,'seed':seed,'splits':splits,'records':records,
        'bin_ms':10.,'resolution':[80,60],'window':16,'representation':'ToFrame(4) + ToBinaRep(4 bits)',
        'frame_storage':'uint8, divide by 15 for model input',
        'alignment':'10ms bins from first event, capped by label count; no tail padding',
        'label_policy':'original state retained; all finite in-sensor nonzero xy regardless of state',
        'preprocessing_seconds':time.perf_counter()-started}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return output/'manifest.json'


def load_manifest(path):
    path=Path(path)
    manifest=json.loads(path.read_text())
    if manifest.get('labels_are_placeholders',True):
        raise ValueError('Refusing missing label provenance or placeholder labels')
    if manifest.get('schema')!=1: raise ValueError('Unsupported dataset manifest')
    seen=set(); subjects=set()
    for split in ('train','val','holdout'):
        names=manifest['splits'][split]
        current={record_key(n)[0] for n in names}
        if seen.intersection(names) or subjects.intersection(current):
            raise ValueError('Subject or recording leakage between splits')
        seen.update(names); subjects.update(current)
    if seen!={r['recording'] for r in manifest['records']}:
        raise ValueError('Manifest split does not cover recordings exactly')
    return manifest


class WindowDataset(Dataset):
    def __init__(self,manifest_path,split,window=16,stride=1):
        if window<1 or stride<1: raise ValueError('Window and stride must be positive')
        self.path=Path(manifest_path).resolve(); self.manifest=load_manifest(self.path)
        self.h5_path=self.path.parent/self.manifest['h5']
        self.window=window; self._handle=None; self.indices=[]
        with h5py.File(self.h5_path,'r') as f:
            for name in self.manifest['splits'][split]:
                valid=f[name]['valid'][:]
                self.indices.extend((name,i) for i in range(0,len(valid),stride) if valid[i])

    def __len__(self): return len(self.indices)

    def __getitem__(self,index):
        if self._handle is None: self._handle=h5py.File(self.h5_path,'r',rdcc_nbytes=16*1024*1024)
        name,end=self.indices[index]; start=max(0,end-self.window+1)
        group=self._handle[name]
        frames=group['frames'][start:end+1].astype(np.float32)/15
        labels=group['labels'][start:end+1]
        valid=group['valid'][start:end+1]
        # Invalid historical labels cannot supervise NCC, but their input frames
        # remain in the causal window. Append validity as a fourth label column.
        labels=np.column_stack([labels,valid.astype(np.float32)])
        pad=self.window-len(frames)
        if pad:
            frames=np.concatenate([np.repeat(frames[:1],pad,0),frames])
            labels=np.concatenate([np.repeat(labels[:1],pad,0),labels])
        return torch.from_numpy(frames),torch.from_numpy(labels.astype(np.float32))

    def close(self):
        if getattr(self,'_handle',None) is not None: self._handle.close(); self._handle=None

    def __getstate__(self):
        state=self.__dict__.copy(); state['_handle']=None; return state

    def __del__(self): self.close()
