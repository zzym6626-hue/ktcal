"""Pixel-unit Kalman filtering and actual asynchronous medium-error correction.

Instantaneous predictions remain immutable for scoring. Retrospective state
replay is reported separately. A newer synchronous anchor supersedes older work.
"""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from .matching import TEMPLATE, SEARCH, crop_rectangle, response_to_pixel


class PixelKalman:
    def __init__(self):
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]],dtype=float)
        self.Q = np.diag([.1,.1,.5,.5])
        self.R = np.diag([2.,2.])
        self.x = None

    def initialize(self, pos):
        self.x = np.r_[pos, [0.,0.]].astype(float)
        self.P = np.diag([10.]*4)

    def predict(self):
        self.x = self.F@self.x
        self.P = self.F@self.P@self.F.T+self.Q
        return self.x[:2].copy()

    def update(self, pos):
        gain = np.linalg.solve(self.H@self.P@self.H.T+self.R, self.H@self.P).T
        self.x += gain@(pos-self.H@self.x)
        identity = np.eye(4)-gain@self.H
        self.P = identity@self.P@identity.T+gain@self.R@gain.T
        return self.x[:2].copy()

    def reset(self, pos, previous_pos=None):
        velocity = np.zeros(2) if previous_pos is None else np.asarray(pos)-previous_pos
        self.initialize(pos)
        self.x[2:] = velocity  # dt=1 bin; previous_pos is the immediately preceding bin
        return self.x[:2].copy()


def gate(epsilon, confidence, tau1=3., tau2=5., confidence_threshold=.5):
    if not np.isfinite(epsilon) or not np.isfinite(confidence) or confidence < confidence_threshold:
        return 3
    return 1 if epsilon<tau1 else 2 if epsilon<tau2 else 3


class PaperTracker:
    def __init__(self, model, matcher, device='cpu', tau1=3., tau2=5., sigma=1.,
                 bin_ms=10., realtime=False):
        if not 0 <= tau1 < tau2 or sigma < 0 or bin_ms <= 0:
            raise ValueError('Invalid gate or timing configuration')
        self.model, self.matcher = model.eval(), matcher.eval()
        self.device = torch.device(device)
        self.tau1,self.tau2,self.sigma = tau1,tau2,sigma
        self.bin_ms,self.realtime = bin_ms,realtime

    def _infer(self, window):
        started = time.perf_counter()
        # Transfer and compute in the same stream, with an explicit completion.
        with torch.inference_mode():
            if self.device.type == 'cuda':
                stream = torch.cuda.Stream(device=self.device)
                with torch.cuda.stream(stream):
                    xy = self.model(window[None].to(self.device))
                stream.synchronize()
            else:
                xy = self.model(window[None].to(self.device))
            coordinate = xy[0].cpu().numpy()*[80.,60.]
        if not np.isfinite(coordinate).all():
            raise RuntimeError('STNet produced a non-finite prediction')
        return coordinate, (time.perf_counter()-started)*1000

    def _replay(self, observations, snapshots, start):
        """Replay only states affected by a newly completed observation."""
        kf=PixelKalman()
        if start:
            kf.x=snapshots[start-1]['x'].copy(); kf.P=snapshots[start-1]['P'].copy()
        for index,item in enumerate(observations[start:],start):
            reset=False
            if kf.x is None:
                kf.initialize(item['global'])
            else:
                previous=kf.x[:2].copy(); prediction=kf.predict()
                if item['local'] is not None: kf.update(item['local'])
                if item['global'] is not None:
                    if item['force'] or np.linalg.norm(prediction-item['global'])>self.sigma:
                        kf.reset(item['global'],previous); reset=True
                    else:
                        kf.update(item['global'])
            snapshots[index]={'x':kf.x.copy(),'P':kf.P.copy(),'reset':reset}
        return kf

    def _async_infer(self, window):
        position,compute_ms=self._infer(window)
        return position,compute_ms,time.perf_counter()

    @torch.inference_mode()
    def track(self, frames):
        if len(frames)==0: raise ValueError('Cannot track an empty recording')
        frames=frames.cpu()
        kf=PixelKalman(); observations=[]; snapshots=[]; online=[]; pending=[]; template=None
        template_frame=-1; sync_anchor=-1; corrected=None
        stats={'tier1':0,'tier2':0,'tier3':0,'stnet_calls':0,'online_sync_resets':0,
               'async_applied':0,'async_discarded':0,'async_deadline_misses':0,
               'frame_latency_ms':[],'frame_processing_ms':[],'frame_arrival_ms':[],
               'frame_emission_ms':[],'correction_completion_ms':[],
               'correction_application_ms':[],'async_events':[],'replayed_frames':0}
        started=time.perf_counter()

        def collect(wait=False):
            nonlocal kf,template,template_frame,pending
            unfinished=[]; earliest=len(observations); applied=[]
            for index,future,submitted,arrival in pending:
                if not wait and not future.done(): unfinished.append((index,future,submitted,arrival)); continue
                position,compute_ms,finished=future.result()
                completion=(finished-submitted)*1000
                stats['correction_completion_ms'].append(completion)
                arrival_completion=(finished-arrival)*1000
                stats['async_deadline_misses']+=int(arrival_completion>self.bin_ms)
                event={'source_frame':index,'compute_ms':compute_ms,
                    'completion_ms':completion,'application_ms':None,
                    'arrival_to_completion_ms':arrival_completion,
                    'arrival_to_application_ms':None,
                    'discarded':bool(index<sync_anchor)}
                stats['async_events'].append(event)
                if index < sync_anchor:
                    stats['async_discarded']+=1
                    continue
                observations[index]['global']=position
                applied.append((event,submitted,arrival))
                stats['async_applied']+=1; earliest=min(earliest,index)
                if index>template_frame:
                    template=crop_rectangle(frames[index:index+1].to(self.device),
                        torch.as_tensor(position,device=self.device,dtype=torch.float32)[None],TEMPLATE)
                    template_frame=index
            pending=unfinished
            if earliest<len(observations):
                kf=self._replay(observations,snapshots,earliest)
                stats['replayed_frames']+=len(observations)-earliest
            finished_application=time.perf_counter()
            for event,submitted,arrival in applied:
                event['application_ms']=(finished_application-submitted)*1000
                event['arrival_to_application_ms']=(finished_application-arrival)*1000
                stats['correction_application_ms'].append(event['application_ms'])

        with ThreadPoolExecutor(max_workers=1) as worker:
            for i,frame in enumerate(frames):
                arrival=started+i*self.bin_ms/1000 if self.realtime else time.perf_counter()
                if self.realtime:
                    remaining=arrival-time.perf_counter()
                    if remaining>0: time.sleep(remaining)
                t0=time.perf_counter(); collect()
                # Match training's history padding; no future frames enter a window.
                window=frames[max(0,i-15):i+1]
                if len(window)<16:
                    window=torch.cat([window[:1].expand(16-len(window),-1,-1,-1),window])
                item={'global':None,'local':None,'force':False}
                if kf.x is None:
                    result,_=self._infer(window); kf.initialize(result)
                    tier=3; item['global']=result; item['force']=True
                    sync_anchor=i
                else:
                    previous=kf.x[:2].copy(); prediction=kf.predict()
                    center=torch.as_tensor(prediction,device=self.device,dtype=torch.float32)[None]
                    search=crop_rectangle(frame[None].to(self.device),center,SEARCH)
                    local,confidence,_=self.matcher(template,search)
                    match=response_to_pixel(local,center)[0].cpu().numpy()
                    tier=gate(np.linalg.norm(prediction-match),float(confidence[0]),self.tau1,self.tau2)
                    if tier==3:
                        result,_=self._infer(window)
                        item['global']=result; item['force']=True
                        kf.reset(result,previous); stats['online_sync_resets']+=1; sync_anchor=i
                    else:
                        item['local']=match
                        result=kf.update(match)
                        if tier==2:
                            result=prediction
                            submitted=time.perf_counter()
                            pending.append((i,worker.submit(self._async_infer,window.clone()),submitted,arrival))
                observations.append(item)
                snapshots.append({'x':kf.x.copy(),'P':kf.P.copy(),'reset':bool(tier==3 and i>0)})
                if tier==3:
                    template=crop_rectangle(frame[None].to(self.device),
                        torch.as_tensor(result,device=self.device,dtype=torch.float32)[None],TEMPLATE)
                    template_frame=i
                online.append(result.copy())
                stats[f'tier{tier}']+=1
                stats['stnet_calls']+=int(tier>1)
                emitted=time.perf_counter()
                stats['frame_processing_ms'].append((emitted-t0)*1000)
                stats['frame_latency_ms'].append((emitted-arrival)*1000)
                stats['frame_arrival_ms'].append((arrival-started)*1000)
                stats['frame_emission_ms'].append((emitted-started)*1000)
            collect(wait=True)
        corrected=np.asarray([s['x'][:2] for s in snapshots])
        stats['retrospective_resets']=sum(s['reset'] for s in snapshots)
        stats['frame_deadline_misses']=sum(v>self.bin_ms for v in stats['frame_latency_ms'])
        stats['wall_ms']=(time.perf_counter()-started)*1000
        stats['latency_ms']=float(np.mean(stats['frame_latency_ms']))
        stats['trigger_ratio']=stats['stnet_calls']/len(frames)
        stats['timing_mode']='paced' if self.realtime else 'as_fast_as_possible'
        return np.asarray(online),corrected,stats
