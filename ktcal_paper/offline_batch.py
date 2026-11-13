"""Experimental batched 100-frame KTCAL; small-tier outputs are Kalman only.

Records are independent. STNet coordinates are precomputed for this explicit
offline protocol, and consulted by the tracker only on medium/large gates.
"""
import numpy as np
import torch
from torch import nn
import time

from .matching import TEMPLATE, crop_rectangle, response_to_pixel


class BatchKalman:
    def __init__(self, count, q_pos=.1, q_vel=.5, r_px=2.):
        if count < 1 or any(not np.isfinite(v) or v <= 0 for v in (q_pos, q_vel, r_px)):
            raise ValueError('Positive batch size and Kalman covariances required')
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                           [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.eye(4, dtype=np.float64)[:2]
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
        self.R = np.eye(2) * r_px
        self.x = np.zeros((count, 4), dtype=np.float64)
        self.P = np.tile(np.eye(4) * 10., (count, 1, 1))

    def initialize(self, positions):
        self.x[:, :2] = positions
        self.x[:, 2:] = 0.
        self.P[:] = np.eye(4) * 10.

    def predict(self):
        self.x = self.x @ self.F.T
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:, :2].copy()

    def update(self, positions, mask=None):
        if mask is None:
            mask = np.ones(len(self.x), dtype=bool)
        if np.any(mask):
            covariance = self.P[mask]
            gain = np.linalg.solve(self.H @ covariance @ self.H.T + self.R,
                                   self.H @ covariance).transpose(0, 2, 1)
            innovation = (positions[mask] - self.x[mask, :2])[..., None]
            self.x[mask] += (gain @ innovation)[..., 0]
            identity = np.eye(4) - gain @ self.H
            self.P[mask] = (identity @ covariance @ identity.transpose(0, 2, 1)
                            + gain @ self.R @ gain.transpose(0, 2, 1))
        return self.x[:, :2].copy()

    def reset(self, positions, previous, mask):
        self.x[mask, :2] = positions[mask]
        self.x[mask, 2:] = positions[mask] - previous[mask]
        self.P[mask] = np.eye(4) * 10.


class _FineBatch(nn.Module):
    def __init__(self, matcher, search_size, subpixel):
        super().__init__()
        self.matcher, self.search_size, self.subpixel = matcher, search_size, subpixel
        device = next(matcher.parameters(), torch.empty(0)).device
        self.register_buffer("span", torch.tensor(
            [search_size[0]-TEMPLATE[0],search_size[1]-TEMPLATE[1]],
            dtype=torch.float32,device=device))

    def forward(self, template, frame, center):
        search = crop_rectangle(frame, center, self.search_size)
        local, confidence, _, details = self.matcher.match_with_diagnostics(
            template, search, subpixel=self.subpixel)
        matched = center + (local-.5)*self.span
        return torch.cat((matched, confidence[:, None], details['psr'][:, None],
                          details['border_peak'][:, None].to(matched.dtype)), -1)


class BatchedFineMatcher:
    """Graph output aliases reusable storage; clone it before the next call."""
    def __init__(self, matcher, search_size, subpixel=True, cuda_graph=False):
        self.module = _FineBatch(matcher, search_size, subpixel).eval()
        self.cuda_graph = cuda_graph
        self.graph = self.inputs = self.output = None

    @torch.inference_mode()
    def __call__(self, template, frame, center):
        if not self.cuda_graph:
            return self.module(template, frame, center)
        if frame.device.type != 'cuda':
            raise ValueError('CUDA graph requires CUDA input')
        values = (template, frame, center)
        if self.graph is None or any(
                (old.shape,old.dtype,old.device) != (new.shape,new.dtype,new.device)
                for old,new in zip(self.inputs,values)):
            self.inputs = tuple(value.clone() for value in values)
            stream = torch.cuda.Stream(device=frame.device)
            stream.wait_stream(torch.cuda.current_stream(frame.device))
            with torch.cuda.stream(stream):
                for _ in range(4):
                    self.module(*self.inputs)
            torch.cuda.current_stream(frame.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(frame.device), torch.cuda.graph(self.graph, stream=stream):
                self.output = self.module(*self.inputs)
        for destination, value in zip(self.inputs, values):
            destination.copy_(value)
        self.graph.replay()
        return self.output


class BatchedTemplateCrop:
    """Fixed-shape graph for repeated template crops; returned storage is owned."""
    def __init__(self, cuda_graph=False):
        self.cuda_graph = cuda_graph
        self.graph = self.inputs = self.output = None

    @torch.inference_mode()
    def __call__(self, frame, centers):
        if not self.cuda_graph:
            return crop_rectangle(frame, centers, TEMPLATE)
        if frame.device.type != 'cuda':
            raise ValueError('CUDA template graph requires CUDA input')
        values = (frame, centers)
        if self.graph is None or any(
                (old.shape, old.dtype, old.device) != (new.shape, new.dtype, new.device)
                for old, new in zip(self.inputs, values)):
            self.inputs = tuple(value.clone() for value in values)
            stream = torch.cuda.Stream(device=frame.device)
            stream.wait_stream(torch.cuda.current_stream(frame.device))
            with torch.cuda.stream(stream):
                for _ in range(4):
                    crop_rectangle(*self.inputs, TEMPLATE)
            torch.cuda.current_stream(frame.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(frame.device), torch.cuda.graph(self.graph, stream=stream):
                self.output = crop_rectangle(*self.inputs, TEMPLATE)
        for destination, source in zip(self.inputs, values):
            destination.copy_(source)
        self.graph.replay()
        return self.output.clone()


class OfflineBatchTracker:
    def __init__(self, matcher, device='cpu', search_size=(37, 32),
                 tau1=3., tau2=5., confidence_threshold=.5, min_psr=None,
                 q_pos=.1, q_vel=.5, r_px=2., fusion_alpha=None,
                 medium_policy='state', tolerance_px=1., subpixel=True,
                 cuda_graph=False, template_refresh=0, reset_velocity_scale=1.,
                 synchronize_frames=False, template_cuda_graph=False,
                 activity_reset_threshold=None):
        if not 0 < tau1 < tau2 or not 0 <= confidence_threshold <= 1:
            raise ValueError('Invalid gate thresholds')
        if fusion_alpha is not None and not 0 <= fusion_alpha <= 1:
            raise ValueError('Invalid medium fusion alpha')
        if medium_policy not in ('state', 'output_only', 'reset') or tolerance_px <= 0:
            raise ValueError('Invalid medium correction policy')
        if activity_reset_threshold is not None and (not np.isfinite(activity_reset_threshold)
                                                      or activity_reset_threshold <= 0):
            raise ValueError('Current activity reset threshold must be finite and positive')
        if (len(search_size) != 2 or any(v < t for v, t in zip(search_size, TEMPLATE))):
            raise ValueError('Search must contain template')
        self.device = torch.device(device)
        self.matcher = BatchedFineMatcher(matcher, tuple(search_size), subpixel, cuda_graph)
        self.tau1, self.tau2, self.confidence_threshold = tau1, tau2, confidence_threshold
        self.min_psr, self.covariances = min_psr, (q_pos, q_vel, r_px)
        self.fusion_alpha, self.medium_policy, self.tolerance_px = fusion_alpha, medium_policy, tolerance_px
        self.template_refresh, self.reset_velocity_scale = template_refresh, reset_velocity_scale
        self.synchronize_frames = bool(synchronize_frames)
        self.activity_reset_threshold = activity_reset_threshold
        self.template_crop = BatchedTemplateCrop(template_cuda_graph)

    @torch.inference_mode()
    def track(self, frames, global_predictions, return_trace=False, output_transform=None):
        if frames.ndim != 5 or frames.shape[2:] != (2, 60, 80):
            raise ValueError('Expected [batch,time,2,60,80] records')
        count, length = frames.shape[:2]
        predictor = global_predictions if callable(global_predictions) else None
        globals_px = (np.zeros((count, length, 2), dtype=np.float64) if predictor else
                      np.asarray(global_predictions, dtype=np.float64))
        if count < 1 or length < 1 or globals_px.shape != (count, length, 2) or not np.isfinite(globals_px).all():
            raise ValueError('Finite global predictions must align with nonempty recordings')

        forward_passes = 0

        def request(step, mask):
            nonlocal forward_passes
            if predictor is not None and np.any(mask):
                values = np.asarray(predictor(frames[:, :step + 1], mask), dtype=np.float64)
                if values.shape != (int(mask.sum()), 2) or not np.isfinite(values).all():
                    raise ValueError('Causal predictor must return finite coordinates for requested records')
                globals_px[mask, step] = values
                forward_passes += 1

        frame_times = []
        started = time.perf_counter()
        first = frames[:, 0].to(self.device)
        if predictor is not None and hasattr(predictor, 'observe'):
            predictor.observe(first)
        request(0, np.ones(count, dtype=bool))
        kalman = BatchKalman(count, *self.covariances)
        kalman.initialize(globals_px[:, 0])
        centers = torch.as_tensor(globals_px[:, 0], device=self.device, dtype=torch.float32)
        template = self.template_crop(first, centers)
        output = np.empty((count, length, 2), dtype=np.float64)
        output[:, 0] = globals_px[:, 0]
        if output_transform is not None:
            output[:, 0] = output_transform(output[:, 0])
        stats = {'tier1': 0, 'tier2': 0, 'tier3': count,
                 'fusion_calls': 0, 'small_fusion_calls': 0,
                 'medium_rollbacks': 0, 'medium_refinements': 0,
                 'border_peaks': 0, 'frames': count * length,
                 'stnet_requests': count, 'fusion_policy': 'medium_only',
                 'matcher_requests': 0, 'activity_forced_large_frames': 0,
                 'global_predictions_precomputed': 0 if predictor else count * length}
        tiers = np.full((count, length), 3, dtype=np.int8)
        ages = np.zeros(count, dtype=np.int32)
        innovations = np.zeros((count, length), dtype=np.float64)
        if predictor is not None:
            if self.synchronize_frames and self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            frame_times.append(1000 * (time.perf_counter() - started))
        for step in range(1, length):
            started = time.perf_counter()
            previous = kalman.x[:, :2].copy()
            prediction = kalman.predict()
            image = frames[:, step].to(self.device)
            if predictor is not None and hasattr(predictor, 'observe'):
                predictor.observe(image)
            center = torch.as_tensor(prediction, device=self.device, dtype=torch.float32)
            forced = np.zeros(count, dtype=bool)
            if self.activity_reset_threshold is not None:
                activity = frames[:, step].sum(dim=(1, 2, 3)).detach().cpu().numpy()
                forced = activity >= self.activity_reset_threshold
            if not forced.any():
                values = self.matcher(template, image, center).detach().cpu().numpy()
            else:
                # These records already require a global reset. Their match
                # diagnostics are unobserved; do not run the fine matcher.
                values = np.concatenate((prediction, np.ones((count, 1)),
                                         np.full((count, 1), np.inf), np.zeros((count, 1))), axis=1)
                remaining = ~forced
                if remaining.any():
                    indices = torch.as_tensor(remaining, device=self.device)
                    values[remaining] = self.matcher(template[indices], image[indices], center[indices]
                                                    ).detach().cpu().numpy()
            stats['matcher_requests'] += int((~forced).sum())
            stats['activity_forced_large_frames'] += int(forced.sum())
            matched, confidence, psr, border = values[:, :2], values[:, 2], values[:, 3], values[:, 4].astype(bool)
            difference = np.linalg.norm(prediction - matched, axis=-1)
            large = (forced | ~np.isfinite(difference) | ~np.isfinite(confidence)
                     | (confidence < self.confidence_threshold) | border
                     | (difference >= self.tau2))
            if self.min_psr is not None:
                large |= ~np.isfinite(psr) | (psr < self.min_psr)
            medium = (~large) & (difference >= self.tau1)
            small = ~large & ~medium
            request(step, medium | large)
            kalman.update(matched, small | medium)
            result = kalman.x[:, :2].copy()
            if np.any(medium):
                if self.medium_policy == 'output_only':
                    alpha = 1. if self.fusion_alpha is None else self.fusion_alpha
                    result[medium] = alpha * globals_px[medium, step] + (1. - alpha) * result[medium]
                    stats['fusion_calls'] += int(medium.sum())
                else:
                    disagreement = np.linalg.norm(result - globals_px[:, step], axis=-1)
                    rollback = medium & ((disagreement > self.tolerance_px)
                                         | (self.medium_policy == 'reset'))
                    refine = medium & ~rollback
                    kalman.reset(globals_px[:, step], previous, rollback)
                    kalman.update(globals_px[:, step], refine)
                    if self.fusion_alpha is not None:
                        result[refine] = (self.fusion_alpha * globals_px[refine, step]
                                         + (1. - self.fusion_alpha) * kalman.x[refine, :2])
                        stats['fusion_calls'] += int(refine.sum())
                    else:
                        result[refine] = kalman.x[refine, :2]
                    result[rollback] = kalman.x[rollback, :2]
                    stats['medium_rollbacks'] += int(rollback.sum())
                    stats['medium_refinements'] += int(refine.sum())
            kalman.reset(globals_px[:, step], previous, large)
            kalman.x[large, 2:] *= self.reset_velocity_scale
            result[large] = kalman.x[large, :2]
            refresh = large | (medium & (self.medium_policy != 'output_only'))
            ages += 1
            if self.template_refresh:
                refresh |= small & (ages >= self.template_refresh)
            if np.any(refresh):
                # A small-tier template refresh uses the Kalman estimate, never STNet.
                positions = np.where(small[:, None], result, globals_px[:, step])
                refresh_centers = torch.as_tensor(positions, device=self.device, dtype=torch.float32)
                crops = self.template_crop(image, refresh_centers)
                indices = torch.as_tensor(refresh, device=self.device)
                template = torch.where(indices[:, None, None, None], crops, template)
                ages[refresh] = 0
            output[:, step] = result if output_transform is None else output_transform(result)
            tiers[:, step] = np.where(large, 3, np.where(medium, 2, 1))
            innovations[:, step] = difference
            innovations[forced, step] = np.nan
            for tier, mask in ((1, small), (2, medium), (3, large)):
                stats['tier%d' % tier] += int(mask.sum())
            stats['stnet_requests'] += int((medium | large).sum())
            stats['border_peaks'] += int(border.sum())
            if predictor is not None:
                if self.synchronize_frames and self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                frame_times.append(1000 * (time.perf_counter() - started))
        if predictor is not None:
            stats['model_forward_passes'] = forward_passes
            stats['frame_processing_ms'] = frame_times
            stats['medium_timing'] = 'synchronous'
            stats['frame_timing_complete_gpu'] = self.synchronize_frames
        if return_trace:
            stats['tiers'] = tiers
            stats['innovations'] = innovations
        return output, stats
