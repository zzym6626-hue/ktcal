"""Reproducible software experiments; measured results stay separate from targets."""
import csv
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import WindowDataset,load_manifest
from .matching import PaperMatcher
from .model import PaperSTNet
from .tracker import PaperTracker


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def save_checkpoint(path,value):
    path=Path(path); temporary=path.with_suffix('.pt.partial')
    torch.save(value,temporary); temporary.replace(path)


def configure(seed=42,threads=4,device='auto'):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if torch.get_num_threads()!=threads: torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if device!='cpu' and not device.startswith('cuda'):
        raise ValueError('This implementation validates CPU and CUDA devices')
    return torch.device(device)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def metrics(predicted,target):
    predicted,target=np.asarray(predicted,dtype=float),np.asarray(target,dtype=float)
    if predicted.shape!=target.shape or predicted.ndim!=2 or predicted.shape[1]!=2 or not len(predicted):
        raise ValueError('Metrics require matching nonempty N x 2 coordinate arrays')
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError('Non-finite values cannot be silently excluded from scoring')
    errors=np.linalg.norm(predicted-target,axis=1)
    return {'frames':len(errors),'mean_error_px':float(errors.mean()),
        'mse_px2':float(np.mean((predicted-target)**2)),
        **{f'p{k}':float(np.mean(errors<=k)) for k in (3,5,10)},
        'out_of_bounds_ratio':float(np.mean(np.any((predicted<0)|(predicted>=np.array([80,60])),axis=1)))}


@torch.inference_mode()
def validate(model,loader,device):
    model.eval(); targets=[]; predictions=[]; forward_predictions=[]
    for frames,labels in loader:
        output,forward=model(frames.to(device),return_forward=True)
        predictions.append(output.cpu().numpy()*[80.,60.])
        forward_predictions.append(forward.cpu().numpy()*[80.,60.])
        targets.append(labels[:,-1,:2].numpy()*[80.,60.])
    if not targets: raise ValueError('Validation split has no valid windows')
    target=np.concatenate(targets)
    return {'primary':metrics(np.concatenate(predictions),target),
            'forward':metrics(np.concatenate(forward_predictions),target)}


def train_epoch(model,matcher,loader,optimizer,device,forward_weight=1.):
    model.train(); matcher.train(); sums=np.zeros(4); count=0
    for frames,labels in loader:
        frames,labels=frames.to(device),labels.to(device)
        target=labels[:,-1,:2]; scale=frames.new_tensor([80.,60.])
        optimizer.zero_grad(set_to_none=True)
        prediction,forward=model(frames,return_forward=True)
        main=F.mse_loss(prediction*scale,target*scale)
        auxiliary=(F.mse_loss(forward*scale,target*scale)
                   if model.forward_head is not None else main.new_zeros(()))
        match=matcher.training_loss(frames,labels)
        loss=main+forward_weight*auxiliary+.3*match
        if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
        loss.backward()
        # Clip recurrent gradients; record the choice in every run config.
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(matcher.parameters()),5.,error_if_nonfinite=True)
        optimizer.step()
        n=len(frames); sums+=np.array([loss.item(),main.item(),auxiliary.item(),match.item()])*n; count+=n
    if count==0: raise ValueError('Training split has no valid windows')
    return dict(zip(('loss','coordinate_mse_px2','forward_mse_px2','matching_mse_px2'),(sums/count).tolist()))


DEFAULTS={'variant':'full','epochs':100,'patience':20,'batch_size':16,'lr':.002,
          'finetune_epochs':2,'finetune_lr':.0002,'weight_decay':.01,'forward_weight':1.,
          'stride':1,'val_stride':1,'window':16,'seed':42,'threads':4,'device':'auto',
          'ncc_channels':16,'temperature':.1,'gradient_clip':5.,'purpose':'software-experiment',
          'inference_mode':'forward','scheduler':'none','min_lr':.00001,'readout':'last'}


def train(manifest_path=None,output=None,resume=None,stop_after_epochs=None,initialize_from=None,**options):
    unknown=set(options)-set(DEFAULTS)
    if unknown: raise ValueError(f'Unknown training options: {sorted(unknown)}')
    if resume and initialize_from: raise ValueError('Resume and initialize-from are mutually exclusive')
    initial=None
    if resume:
        resume=Path(resume).resolve(); run=resume.parent
        state=torch.load(resume,map_location='cpu',weights_only=False)
        if state.get('schema')!=1: raise ValueError('Unsupported checkpoint')
        config=state['config']
        if any(v is not None and config.get(k)!=v for k,v in options.items()):
            raise ValueError('Resume uses saved configuration; changed options require a new run')
        path=Path(manifest_path or config['manifest']).resolve()
        if sha256(path)!=state['manifest_sha256']: raise ValueError('Dataset manifest changed since checkpoint')
        if state['phase']=='complete':
            return run/('bidirectional_best.pt' if config.get('inference_mode')=='bidirectional' else 'forward_best.pt')
        if state.get('training_completed') or state.get('finetune_completed_epochs'):
            raise ValueError('Completed best weights cannot resume; use last.pt or initialize-from for a new experiment')
    else:
        if manifest_path is None or output is None: raise ValueError('New run requires manifest and output')
        path=Path(manifest_path).resolve(); run=Path(output).resolve()
        config=dict(DEFAULTS); config.update({k:v for k,v in options.items() if v is not None})
        if config['epochs']<1 or config['patience']<1 or config['batch_size']<1:
            raise ValueError('Epochs, patience and batch size must be positive')
        if config['finetune_epochs']!=2: raise ValueError('The supplied paper requires exactly two fine-tune epochs')
        if config['inference_mode'] not in ('forward','bidirectional'):
            raise ValueError('Unknown inference mode')
        if config['readout'] not in ('last','final_hidden'):
            raise ValueError('Unknown recurrent readout')
        if config['scheduler'] not in ('none','cosine'):
            raise ValueError('Unknown learning-rate scheduler')
        if not 0<config['min_lr']<=config['lr'] or config['finetune_lr']<=0:
            raise ValueError('Require 0 < min_lr <= lr and positive fine-tune lr')
        config['manifest']=str(path)
        if initialize_from:
            source=Path(initialize_from).resolve()
            initial=torch.load(source,map_location='cpu',weights_only=False)
            if initial.get('schema')!=1 or initial.get('manifest_sha256')!=sha256(path):
                raise ValueError('Initialization requires the identical data manifest')
            source_model=initial.get('model_config',{})
            if source_model.get('deployed') is not False or source_model.get('variant')!=config['variant']:
                raise ValueError('Initialization requires the same non-exported architecture')
            if any(initial['config'][k]!=config[k] for k in ('window','ncc_channels','temperature')):
                raise ValueError('Initialization input or matching configuration differs')
            config['initialization']={'checkpoint':str(source),'checkpoint_sha256':sha256(source),
                'source_phase':initial['phase'],'source_epoch':initial['epoch'],
                'source_readout':source_model.get('readout','last'),'target_readout':config['readout'],
                'optimizer':'fresh AdamW; source optimizer and RNG are not restored'}
        state=None
    bidirectional_inference=config.get('inference_mode','forward')=='bidirectional'
    selection_head='primary' if bidirectional_inference else 'forward'
    device=configure(config['seed'],config['threads'],config['device'])
    manifest=load_manifest(path)
    datasets=[WindowDataset(path,s,window=config['window'],stride=config['stride'] if s=='train' else config['val_stride'])
              for s in ('train','val')]
    if not all(len(d) for d in datasets): raise ValueError('Train and validation splits must contain valid windows')
    loaders=[DataLoader(d,batch_size=config['batch_size'],shuffle=(i==0),num_workers=0) for i,d in enumerate(datasets)]
    if state is None:
        run.mkdir(parents=True,exist_ok=False)
        config.update({'resolved_device':str(device),'train_windows':len(datasets[0]),'val_windows':len(datasets[1]),
            'dataset':manifest['dataset'],'bin_ms':manifest['bin_ms'],
            'label_policy':manifest['label_policy'],'coordinate_loss':'MSE in 80x60 pixel units',
            'code_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                'platform':platform.platform(),'machine':platform.machine(),
                'cuda_device':torch.cuda.get_device_name(device) if device.type=='cuda' else None}})
        write_json(run/'config.json',config); write_json(run/'split_manifest.json',manifest)
        state={'schema':1,'config':config,'manifest_sha256':sha256(path),'phase':'bidir','epoch':0,
               'best_val':float('inf'),'no_improve':0,'history':[]}
    model=PaperSTNet(deployed=state['phase']=='finetune',variant=config['variant'],
                     readout=config.get('readout','last')).to(device)
    matcher=PaperMatcher(config['ncc_channels'],config['temperature']).to(device)

    def new_optimizer():
        return torch.optim.AdamW(list(model.parameters())+list(matcher.parameters()),
            lr=config['finetune_lr'] if state['phase']=='finetune' else config['lr'],weight_decay=config['weight_decay'])

    optimizer=new_optimizer()
    if 'model_state' in state:
        model.load_state_dict(state['model_state']); matcher.load_state_dict(state['matcher_state'])
        optimizer.load_state_dict(state['optimizer_state']); restore_rng(state['rng'])
    elif initial is not None:
        model.load_state_dict(initial['model_state']); matcher.load_state_dict(initial['matcher_state'])

    def checkpoint():
        return {**state,'model_state':model.state_dict(),'matcher_state':matcher.state_dict(),
            'optimizer_state':optimizer.state_dict(),'rng':rng_state(),
            'model_config':{'deployed':model.deployed,'variant':model.variant,'readout':model.readout},
            'parameters':{'stnet':sum(p.numel() for p in model.parameters()),
                          'matcher':sum(p.numel() for p in matcher.parameters())}}

    if initial is not None:
        state['initial_validation']=validate(model,loaders[1],device)
        state['best_val']=state['initial_validation'][selection_head]['mean_error_px']
        save_checkpoint(run/'bidirectional_best.pt',checkpoint())
        write_json(run/'initial_validation.json',state['initial_validation'])

    completed_this_call=0
    try:
        while state['phase']!='complete':
            limit=config['epochs'] if state['phase']=='bidir' else config['finetune_epochs']
            while state['epoch']<limit:
                if state['phase']=='bidir' and state['no_improve']>=config['patience']: break
                started=time.perf_counter()
                if state['phase']=='bidir' and config.get('scheduler')=='cosine':
                    progress=state['epoch']/max(1,config['epochs']-1)
                    rate=config['min_lr']+.5*(config['lr']-config['min_lr'])*(1+math.cos(math.pi*progress))
                    for group in optimizer.param_groups: group['lr']=rate
                learning_rate=optimizer.param_groups[0]['lr']
                training=train_epoch(model,matcher,loaders[0],optimizer,device,config['forward_weight'])
                validation=validate(model,loaders[1],device)
                state['epoch']+=1; completed_this_call+=1
                score=validation[selection_head]['mean_error_px']
                improved=score<state['best_val']
                if improved: state['best_val']=score; state['no_improve']=0
                else: state['no_improve']+=1
                row={'phase':state['phase'],'epoch':state['epoch'],'train':training,'val':validation,
                     'seconds':time.perf_counter()-started,'selected':improved,'learning_rate':learning_rate}
                state['history'].append(row)
                snapshot=checkpoint()
                if improved:
                    save_checkpoint(run/('bidirectional_best.pt' if state['phase']=='bidir' else 'forward_best.pt'),snapshot)
                save_checkpoint(run/'last.pt',snapshot)
                write_json(run/'history.json',state['history'])
                print(json.dumps(row),flush=True)
                if stop_after_epochs is not None and completed_this_call>=stop_after_epochs:
                    return run/'last.pt'
            if state['phase']=='bidir':
                if bidirectional_inference:
                    state['phase']='complete'
                    continue
                best=torch.load(run/'bidirectional_best.pt',map_location=device,weights_only=False)
                model.load_state_dict(best['model_state']); matcher.load_state_dict(best['matcher_state'])
                model=model.export_forward().to(device)
                state.update({'phase':'finetune','epoch':0,'best_val':float('inf'),'no_improve':0})
                optimizer=new_optimizer()
                save_checkpoint(run/'last.pt',checkpoint())
            else:
                state['phase']='complete'
        if bidirectional_inference:
            best_path=run/'bidirectional_best.pt'
            best=torch.load(best_path,map_location='cpu',weights_only=False)
            best['training_completed']=True
            best['completed_bidir_epochs']=sum(r['phase']=='bidir' for r in state['history'])
            save_checkpoint(best_path,best)
            state['training_completed']=True
            write_json(run/'training_summary.json',{'status':'complete','dataset':manifest['dataset'],
                'purpose':config['purpose'],'train_windows':len(datasets[0]),'val_windows':len(datasets[1]),
                'bidir_epochs':best['completed_bidir_epochs'],'finetune_epochs':0,
                'inference_mode':'bidirectional','readout':config.get('readout','last'),'selection_head':selection_head,
                'initial_validation':state.get('initial_validation'),
                'best_validation':best['best_val'],'selected_epoch':best['epoch'],
                'parameters':best['parameters'],'checkpoint':best_path.name,'checkpoint_sha256':sha256(best_path),
                'holdout_used_for_model_selection':False,'paper_results_reproduced':False})
            # Publish the terminal resume state after all completion artifacts.
            save_checkpoint(run/'last.pt',checkpoint())
            return best_path
        best=torch.load(run/'forward_best.pt',map_location='cpu',weights_only=False)
        best['finetune_completed_epochs']=config['finetune_epochs']
        best['selected_finetune_epoch']=best['epoch']
        save_checkpoint(run/'forward_best.pt',best)
        state['finetune_completed_epochs']=config['finetune_epochs']
        write_json(run/'training_summary.json',{'status':'complete','dataset':manifest['dataset'],
            'purpose':config['purpose'],'train_windows':len(datasets[0]),'val_windows':len(datasets[1]),
            'bidir_epochs':sum(r['phase']=='bidir' for r in state['history']),
            'finetune_epochs':sum(r['phase']=='finetune' for r in state['history']),
            'best_forward_validation':best['best_val'],'parameters':best['parameters'],
            'checkpoint':'forward_best.pt','checkpoint_sha256':sha256(run/'forward_best.pt'),
            'holdout_used_for_model_selection':False,'paper_results_reproduced':False})
        save_checkpoint(run/'last.pt',checkpoint())
        return run/'forward_best.pt'
    finally:
        for d in datasets: d.close()


def load_deployed(checkpoint,device):
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    if state['config'].get('inference_mode')=='bidirectional':
        if state.get('model_config',{}).get('deployed') or not state.get('training_completed'):
            raise ValueError('Bidirectional evaluation requires a completed non-exported checkpoint')
    else:
        if not state.get('model_config',{}).get('deployed'):
            raise ValueError('Evaluation requires an exported, fine-tuned forward checkpoint')
        if state.get('finetune_completed_epochs')!=2:
            raise ValueError('Evaluation requires forward_best.pt after completing two fine-tune epochs')
    model=PaperSTNet(**state['model_config']).to(device).eval(); model.load_state_dict(state['model_state'])
    matcher=PaperMatcher(state['config']['ncc_channels'],state['config']['temperature']).to(device).eval()
    matcher.load_state_dict(state['matcher_state'])
    return model,matcher,state


@torch.inference_mode()
def evaluate(checkpoint,manifest_path,output,split='holdout',mode='both',device='auto',threads=4,
             tau1=3.,tau2=5.,sigma=1.,realtime=False,max_frames=None):
    device=configure(42,threads,device); path=Path(manifest_path).resolve()
    manifest=load_manifest(path); model,matcher,state=load_deployed(checkpoint,device)
    if sha256(path)!=state['manifest_sha256']:
        raise ValueError('Evaluation manifest differs from the training split')
    out=Path(output).resolve(); out.mkdir(parents=True,exist_ok=False)
    modes=['stnet_only','ktcal'] if mode=='both' else [mode]
    rows=[]; reports={}; scale=np.array([80.,60.])
    # Warm up fixed-shape model computation outside measured per-frame work.
    dummy=torch.zeros(1,16,2,60,80,device=device)
    for _ in range(3): model(dummy)
    if device.type=='cuda': torch.cuda.synchronize(device)
    tracker=PaperTracker(model,matcher,device,tau1,tau2,sigma,manifest['bin_ms'],realtime)
    for variant in modes:
        predicted=[]; targets=[]; retrospective=[]; record_stats=[]; all_states=[]
        with (out/f'{variant}_predictions.csv').open('w') as csvfile,h5py.File(path.parent/manifest['h5'],'r') as data:
            writer=csv.writer(csvfile); writer.writerow(['recording','frame','x','y','target_x','target_y','state','valid','corrected_x','corrected_y'])
            for name in manifest['splits'][split]:
                group=data[name]; count=len(group['frames'])
                if max_frames is not None: count=min(count,max_frames)
                frames=torch.from_numpy(group['frames'][:count].astype(np.float32)/15)
                labels=group['labels'][:count]; valid=group['valid'][:count]
                if variant=='ktcal': online,corrected,stats=tracker.track(frames)
                else:
                    online=[]; timings=[]; processing=[]; arrivals=[]; emissions=[]; started=time.perf_counter()
                    for i in range(count):
                        arrival=started+i*manifest['bin_ms']/1000 if realtime else time.perf_counter()
                        remaining=arrival-time.perf_counter()
                        if realtime and remaining>0: time.sleep(remaining)
                        t0=time.perf_counter(); window=frames[max(0,i-15):i+1]
                        if len(window)<16: window=torch.cat([window[:1].expand(16-len(window),-1,-1,-1),window])
                        xy,_=tracker._infer(window); online.append(xy)
                        emitted=time.perf_counter()
                        timings.append((emitted-arrival)*1000); processing.append((emitted-t0)*1000)
                        arrivals.append((arrival-started)*1000); emissions.append((emitted-started)*1000)
                    online=np.asarray(online); corrected=online.copy()
                    stats={'frame_latency_ms':timings,'wall_ms':(time.perf_counter()-started)*1000,
                           'frame_processing_ms':processing,'frame_arrival_ms':arrivals,'frame_emission_ms':emissions,
                           'frame_deadline_misses':sum(v>manifest['bin_ms'] for v in timings),
                           'timing_mode':'paced' if realtime else 'as_fast_as_possible',
                           'stnet_calls':count,'latency_ms':float(np.mean(timings)),'trigger_ratio':1.}
                truth=labels[:,:2]*scale
                predicted.append(online[valid]); targets.append(truth[valid]); retrospective.append(corrected[valid]); all_states.append(labels[valid,2])
                record_stats.append({'recording':name,'input_frames':count,'metrics':metrics(online[valid],truth[valid]),**stats})
                for i in range(count):
                    writer.writerow([name,i,*online[i],*truth[i],labels[i,2],bool(valid[i]),*corrected[i]])
                print(f'Evaluated {variant} {name}: {count} frames, mean error {record_stats[-1]["metrics"]["mean_error_px"]:.3f}px',flush=True)
        y,t,corrected=np.concatenate(predicted),np.concatenate(targets),np.concatenate(retrospective)
        latencies=np.concatenate([r['frame_latency_ms'] for r in record_stats])
        total=sum(r['input_frames'] for r in record_stats)
        row={'dataset':manifest['dataset'],'variant':variant,'architecture':state['model_config']['variant'],
            'inference_mode':state['config'].get('inference_mode','forward'),
            'readout':state['model_config'].get('readout','last'),
            'split':split,**metrics(y,t),'input_frames':total,
            'online_latency_ms':float(latencies.mean()),'online_latency_p95_ms':float(np.percentile(latencies,95)),
            'processing_ms':float(np.mean(np.concatenate([r['frame_processing_ms'] for r in record_stats]))),
            'wall_ms_per_frame':sum(r['wall_ms'] for r in record_stats)/total,
            'trigger_ratio':sum(r['stnet_calls'] for r in record_stats)/total,
            'device':str(device),'max_frames_per_record':max_frames,
            'parameters_stnet':sum(p.numel() for p in model.parameters()),
            'parameters_matcher':sum(p.numel() for p in matcher.parameters()) if variant=='ktcal' else 0,
            'tau1':tau1,'tau2':tau2,'sigma':sigma,'bin_ms':manifest['bin_ms'],
            'timing_mode':'paced' if realtime else 'as_fast_as_possible',
            'frame_deadline_misses':sum(r['frame_deadline_misses'] for r in record_stats)}
        rows.append(row)
        flags=np.concatenate(all_states)
        reports[variant]={'online':row,'retrospective_unavailable_at_emission':metrics(corrected,t),
            'by_original_label_state':{str(int(flag)):metrics(y[flags==flag],t[flags==flag]) for flag in np.unique(flags)},
            'records':record_stats}
    report={'schema':1,'checkpoint':str(Path(checkpoint).resolve()),'checkpoint_sha256':sha256(checkpoint),
        'training_config':state['config'],'manifest_sha256':sha256(path),'rows':rows,'details':reports,
        'limitations':['AIS2024 is not SEET, 3ET+2025 or Ini-30.',
            'Online outputs are immutable; retrospective correction is scored separately.',
            'Online emission, correction completion and total wall time are separate measurements.',
            'Literal 30x25/35x30 patch geometry limits epsilon to sqrt(12.5) px.',
            'No FLOPs or RTX3090 latency claims are inferred from this run.'],
        'paper_results_reproduced':False}
    write_json(out/'report.json',report)
    with (out/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    table=['# Measured software experiment','',f'Dataset: {manifest["dataset"]}; split: {split}; device: {device}.',
        '','| Variant | Inference | Readout | p3 | p5 | p10 | Mean error (px) | Online ms/frame | STNet calls |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        table.append(f'| {row["variant"]} | {row["inference_mode"]} | {row["readout"]} | {row["p3"]:.4f} | {row["p5"]:.4f} | {row["p10"]:.4f} | {row["mean_error_px"]:.3f} | {row["online_latency_ms"]:.3f} | {row["trigger_ratio"]:.1%} |')
    table+=['','These are measured local results, not reproduced paper tables. See report.json for timing, coverage and retrospective results.']
    (out/'table.md').write_text('\n'.join(table)+'\n')
    return out/'report.json'
