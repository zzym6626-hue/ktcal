"""Pixel-unit Kalman filtering and actual asynchronous medium-error correction.

Instantaneous predictions remain immutable for scoring. Retrospective state
replay is reported separately. A newer synchronous anchor supersedes older work.
"""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from .matching import (COARSE_FACTOR, COARSE_SEARCH, COARSE_TEMPLATE, SEARCH,
                       TEMPLATE, crop_rectangle, downsample, response_to_pixel)


class PixelKalman:
    def __init__(self, q_pos=.1, q_vel=.5, r_px=2.):
        if q_pos <= 0 or q_vel <= 0 or r_px <= 0:
            raise ValueError('Kalman covariances must be positive')
        self.q_pos, self.q_vel, self.r_px = q_pos, q_vel, r_px
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]],dtype=float)
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.R = np.diag([r_px, r_px])
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


def gate(epsilon, confidence, tau1=3., tau2=5., confidence_threshold=.5,
         border_peak=False, psr=None, min_psr=None, escalate_tier=3):
    """Three-tier gate with explicit capture-range rejection.

    A border peak or an insignificant peak cannot be used as a Kalman
    observation: the fine search only spans +-2.5 px, so the true displacement
    may be outside the window and the correlated pixel is misleading.
    """
    if escalate_tier not in (2, 3, None):
        raise ValueError('Escalation tier must be 2, 3 or None')
    if not np.isfinite(epsilon) or not np.isfinite(confidence) or confidence < confidence_threshold:
        return 3
    if border_peak and escalate_tier is not None:
        return escalate_tier
    if min_psr is not None and (psr is None or not np.isfinite(psr) or psr < min_psr):
        return escalate_tier
    return 1 if epsilon<tau1 else 2 if epsilon<tau2 else 3


class PaperTracker:
    def __init__(self, model, matcher, device='cpu', tau1=3., tau2=5., sigma=1.,
                 bin_ms=10., realtime=False, q_pos=.1, q_vel=.5, r_px=2.,
                 capture='fine', min_psr=None, border_tier=3, fusion_alpha=None,
                 subpixel=False, max_prediction_variance_px2=None,
                 max_consecutive_prediction_skips=1, record_gate_trace=False):
        if not 0 <= tau1 < tau2 or sigma < 0 or bin_ms <= 0:
            raise ValueError('Invalid gate or timing configuration')
        if capture not in ('fine', 'coarse'):
            raise ValueError("Capture mode must be 'fine' or 'coarse'")
        if border_tier not in (2, 3, None):
            raise ValueError('Border escalation tier must be 2, 3 or None')
        if fusion_alpha is not None and not 0 <= fusion_alpha <= 1:
            raise ValueError('Fusion alpha must lie inside [0, 1]')
        if (max_prediction_variance_px2 is not None and
                (not np.isfinite(max_prediction_variance_px2) or
                 max_prediction_variance_px2 <= 0)):
            raise ValueError('Prediction variance limit must be finite and positive')
        if (isinstance(max_consecutive_prediction_skips, bool) or
                not isinstance(max_consecutive_prediction_skips, int) or
                max_consecutive_prediction_skips < 1):
            raise ValueError('Maximum consecutive prediction skips must be positive')
        self.model, self.matcher = model.eval(), matcher.eval()
        self.device = torch.device(device)
        self.tau1,self.tau2,self.sigma = tau1,tau2,sigma
        self.bin_ms,self.realtime = bin_ms,realtime
        self.q_pos,self.q_vel,self.r_px = q_pos,q_vel,r_px
        self.capture,self.min_psr,self.border_tier = capture,min_psr,border_tier
        self.fusion_alpha = fusion_alpha
        self.subpixel = bool(subpixel)
        self.max_prediction_variance_px2 = max_prediction_variance_px2
        self.max_consecutive_prediction_skips = max_consecutive_prediction_skips
        self.record_gate_trace = bool(record_gate_trace)

    def _kalman(self):
        return PixelKalman(self.q_pos, self.q_vel, self.r_px)

    def _locate(self, template, search):
        """Match with diagnostics when available; tolerate plain matchers."""
        if hasattr(self.matcher, 'match_with_diagnostics'):
            try:
                return self.matcher.match_with_diagnostics(template, search,
                                                           subpixel=self.subpixel)
            except TypeError:
                return self.matcher.match_with_diagnostics(template, search)
        local, confidence, _ = self.matcher(template, search)
        return local, confidence, None, {
            'border_peak': torch.zeros_like(confidence, dtype=torch.bool),
            'psr': torch.full_like(confidence, float('inf'))}

    def _match(self, template, coarse_template, frame, coarse_frame, center):
        """Locate the template, optionally seeding the fine pass with a coarse peak."""
        proposal, border, psr = center, None, None
        if self.capture == 'coarse' and coarse_template is not None:
            small_center = center / COARSE_FACTOR
            signal = coarse_frame if coarse_frame.ndim == 4 else coarse_frame[None]
            coarse_search = crop_rectangle(
                signal.to(self.device), small_center, COARSE_SEARCH)
            local, _, _, coarse = self._locate(coarse_template, coarse_search)
            proposal = response_to_pixel(
                local, small_center, COARSE_TEMPLATE, COARSE_SEARCH) * COARSE_FACTOR
            border = coarse['border_peak']
            psr = coarse['psr']
        images = frame.to(self.device) if frame.ndim == 4 else frame[None].to(self.device)
        search = crop_rectangle(images, proposal, SEARCH)
        local, confidence, _, fine = self._locate(template, search)
        match = response_to_pixel(local, proposal)
        selected_border = fine['border_peak'] if border is None else border
        selected_psr = fine['psr'] if psr is None else torch.minimum(psr, fine['psr'])
        # Transfer all decisions at once; scalar float/bool conversions on CUDA
        # tensors would synchronize the device separately for every field.
        values = torch.cat((match, confidence[:, None], selected_psr[:, None],
                            selected_border[:, None].to(match.dtype)), dim=1).detach().cpu().numpy()[0]
        return (values[:2], float(values[2]),
                {'border_peak': bool(values[4]), 'psr': float(values[3])})

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
        kf=self._kalman()
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

    def _template_at(self, images, center):
        center=torch.as_tensor(center,device=self.device,dtype=torch.float32)[None]
        images=images.to(self.device)
        template=crop_rectangle(images,center,TEMPLATE)
        if self.capture!='coarse': return template,None
        return template,crop_rectangle(downsample(images),center/COARSE_FACTOR,COARSE_TEMPLATE)

    @torch.inference_mode()
    def track(self, frames, global_predictions=None):
        if len(frames)==0: raise ValueError('Cannot track an empty recording')
        if global_predictions is not None:
            if torch.is_tensor(global_predictions):
                global_predictions=global_predictions.detach().cpu().numpy()
            global_predictions=np.asarray(global_predictions,dtype=float)
            if (global_predictions.shape != (len(frames),2)
                    or not np.isfinite(global_predictions).all()):
                raise ValueError(
                    'External global predictions must be finite [T,2] coordinates')
        frames=frames.cpu()
        kf=self._kalman(); observations=[]; snapshots=[]; online=[]; pending=[]; template=None
        template_coarse=None
        template_frame=-1; sync_anchor=-1; corrected=None
        consecutive_prediction_skips=0
        stats={'tier1':0,'tier2':0,'tier3':0,'stnet_calls':0,'online_sync_resets':0,
               'async_applied':0,'async_discarded':0,'async_deadline_misses':0,
               'border_peaks':0,'psr_rejections':0,'fusion_calls':0,
               'skipped_matches':0,'frame_tiers':[],
               'frame_latency_ms':[],'frame_processing_ms':[],'frame_arrival_ms':[],
               'frame_emission_ms':[],'correction_completion_ms':[],
               'correction_application_ms':[],'async_events':[],'replayed_frames':0}
        if self.record_gate_trace:
            stats['gate_trace'] = []
        started=time.perf_counter()

        def infer(index,window):
            if global_predictions is not None:
                return global_predictions[index].copy(),0.
            return self._infer(window)

        def async_infer(index,window):
            if global_predictions is not None:
                return global_predictions[index].copy(),0.,time.perf_counter()
            return self._async_infer(window)

        def collect(wait=False):
            nonlocal kf,template,template_coarse,template_frame,pending
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
                    template,template_coarse=self._template_at(frames[index:index+1],position)
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
                prediction=match=confidence=details=None
                if kf.x is None:
                    result,_=infer(i,window); kf.initialize(result)
                    tier=3; item['global']=result; item['force']=True
                    sync_anchor=i
                    route='stnet_reset'
                else:
                    previous=kf.x[:2].copy(); prediction=kf.predict()
                    skip_match = (self.max_prediction_variance_px2 is not None
                                  and consecutive_prediction_skips < self.max_consecutive_prediction_skips
                                  and np.max(np.diag(kf.P)[:2]) <= self.max_prediction_variance_px2)
                    if skip_match:
                        result=prediction; tier=1
                        route='prediction_only'
                        consecutive_prediction_skips+=1
                        stats['skipped_matches']+=1
                    else:
                        consecutive_prediction_skips=0
                        center=torch.as_tensor(prediction,device=self.device,dtype=torch.float32)[None]
                        frame_small=downsample(frame[None].to(self.device)) if self.capture=='coarse' else None
                        match,confidence,details=self._match(
                            template,template_coarse,frame,frame_small,center)
                        stats['border_peaks']+=int(details['border_peak'])
                        stats['psr_rejections']+=int(self.min_psr is not None and details['psr']<self.min_psr)
                        tier=gate(np.linalg.norm(prediction-match),confidence,self.tau1,self.tau2,
                            border_peak=details['border_peak'],psr=details['psr'],
                            min_psr=self.min_psr,escalate_tier=self.border_tier)
                        route=('matched_update' if tier==1 else
                               'async_prediction' if tier==2 else 'stnet_reset')
                        if tier==3:
                            result,_=infer(i,window)
                            item['global']=result; item['force']=True
                            kf.reset(result,previous); stats['online_sync_resets']+=1; sync_anchor=i
                        else:
                            item['local']=match
                            result=kf.update(match)
                            if tier==2:
                                result=prediction
                                submitted=time.perf_counter()
                                pending.append((i,worker.submit(async_infer,i,window.clone()),submitted,arrival))
                    if tier!=3 and self.fusion_alpha is not None:
                        # Accuracy-only fusion changes the reported coordinate,
                        # not the Kalman state or asynchronous replay.
                        fused,_=infer(i,window)
                        result=self.fusion_alpha*fused+(1-self.fusion_alpha)*result
                        stats['fusion_calls']+=1
                observations.append(item)
                snapshots.append({'x':kf.x.copy(),'P':kf.P.copy(),'reset':bool(tier==3 and i>0)})
                if tier==3:
                    template,template_coarse=self._template_at(frame[None],result)
                    template_frame=i
                online.append(result.copy())
                if self.record_gate_trace:
                    stats['gate_trace'].append({
                        'route': route,
                        'output_px': result.tolist(),
                        'prediction_px': prediction.tolist() if prediction is not None else None,
                        'match_px': match.tolist() if match is not None else None,
                        'confidence': confidence,
                        'border_peak': details['border_peak'] if details is not None else False,
                    })
                stats[f'tier{tier}']+=1
                stats['frame_tiers'].append(tier)
                stats['stnet_calls']+=int(tier>1)+int(self.fusion_alpha is not None and tier<3)
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
        stats['global_prediction_source']='precomputed' if global_predictions is not None else 'model'
        stats['timing_mode']='paced' if self.realtime else 'as_fast_as_possible'
        return np.asarray(online),corrected,stats
