"""
KTCal closed-loop predict-and-correct tracker (KTCal paper, Fig 1 / review reply W1).

Faithful on-demand scheduling: STNet is a temporal network that is invoked ONLY
when the error-aware trigger fires, never on every frame. This is the mechanism
behind the paper's >75% latency reduction (Table 6).

Per-frame pipeline over an event-frame sequence:

    1. Kalman prediction (fast main path, always runs).
    2. NCC template matching in a local search window centered on the Kalman
       prediction -> matched coordinate + confidence.
    3. Error-aware three-tier scheduling on epsilon = |kalman_pred - ncc_match|
       (pixel distance in the 80x60 grid):
           epsilon < tau1 (3px)      : output Kalman (corrected by NCC), NO STNet.
           tau1 <= epsilon < tau2 (5): output Kalman now; STNet runs
                                        ASYNCHRONOUSLY to calibrate the filter
                                        state (its cost is off the critical path).
           epsilon >= tau2 (5px)     : force STNet global inference SYNCHRONOUSLY
                                        (its cost is on the critical path).
    4. Trajectory rollback (tau3 = 1px): after an STNet observation, if the
       deviation sigma against the Kalman state exceeds tau3, reset the filter.

STNet runs on a sliding window of the most recent `window_size` event frames and
uses the last bidirectional hidden state (STNet.last_step, review reply W6) to
produce a single global pupil coordinate for the current frame.

Latency accounting uses time.perf_counter():
  - critical-path latency per frame = Kalman + NCC (+ STNet only for tier3).
  - tier2 STNet time is measured separately (async, off the critical path).
"""
import queue
import threading
import time
from collections import deque

import numpy as np
import torch

from models.kalman_filter import KalmanFilter2D

GRID = np.array([80.0, 60.0])  # normalized coord -> 80x60 grid pixels


def _to_grid(norm_xy):
    return np.asarray(norm_xy, dtype=np.float64) * GRID


def _pixel_dist(a_norm, b_norm):
    return float(np.linalg.norm(_to_grid(a_norm) - _to_grid(b_norm)))


class _AsyncSTNetWorker:
    """Run frame-tagged STNet windows on one background thread."""

    _STOP = object()

    def __init__(self, stnet, device, batch_size=4):
        self.stnet = stnet
        self.device = device
        self.batch_size = max(1, batch_size)
        self.requests = queue.Queue()
        self.results = queue.Queue()
        self.stream = (
            torch.cuda.Stream(device=device)
            if torch.device(device).type == 'cuda' else None
        )
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def submit(self, frame_id, window_frames, submitted_at):
        window = torch.stack(tuple(window_frames), dim=0).float()
        self.requests.put((frame_id, window, submitted_at))

    def poll(self):
        ready = []
        while True:
            try:
                ready.append(self.results.get_nowait())
            except queue.Empty:
                return ready

    def close(self):
        self.requests.put(self._STOP)
        self.thread.join()
        return self.poll()

    def _work(self):
        deferred = None
        stopping = False
        while not stopping:
            item = deferred if deferred is not None else self.requests.get()
            deferred = None
            if item is self._STOP:
                break

            batch = [item]
            window_len = item[1].shape[0]
            while len(batch) < self.batch_size:
                try:
                    candidate = self.requests.get_nowait()
                except queue.Empty:
                    break
                if candidate is self._STOP:
                    stopping = True
                    break
                if candidate[1].shape[0] != window_len:
                    deferred = candidate
                    break
                batch.append(candidate)

            try:
                windows = torch.stack([request[1] for request in batch], dim=0).to(self.device)
                if self.stream is None:
                    started_at = time.perf_counter()
                    with torch.no_grad():
                        coords = self.stnet.last_step(windows)
                    compute_ms = (time.perf_counter() - started_at) * 1000
                else:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(self.stream), torch.no_grad():
                        start_event.record(self.stream)
                        coords = self.stnet.last_step(windows)
                        end_event.record(self.stream)
                    self.stream.synchronize()
                    compute_ms = start_event.elapsed_time(end_event)
                coords = coords.cpu().numpy()
                share_ms = compute_ms / len(batch)
                finished_at = time.perf_counter()
                for request, coord in zip(batch, coords):
                    self.results.put({
                        'frame_id': request[0],
                        'coord': coord,
                        'submitted_at': request[2],
                        'finished_at': finished_at,
                        'compute_ms': share_ms,
                    })
            except Exception as exc:
                self.results.put({'error': exc})


class KTCalTracker:
    def __init__(self, stnet, ncc_matcher,
                 tau1=3.0, tau2=5.0, tau3=3.0, ncc_thresh=0.5,
                 template_size=11, search_size=21, window_size=16,
                 ncc_center_penalty=0.0, ncc_boundary_penalty=0.0,
                 hard_reset=False, device='cpu', collect_rollout=False,
                 reset_velocity_mode='legacy', low_conf_policy='sync_stnet',
                 kalman_process_noise=1e-3, kalman_measurement_noise=1e-2,
                 kalman_velocity_decay=1.0, stnet_cooldown=0,
                 disable_tier2=False, severe_conf_threshold=0.2,
                 severe_epsilon_threshold=9.0,
                 cooldown_low_conf_mode='ncc',
                 trigger_classifier=None,
                 trigger_learn_threshold=0.5):

        self.stnet = stnet
        self.ncc = ncc_matcher
        self.tau1 = tau1
        self.tau2 = tau2
        self.tau3 = tau3
        self.ncc_thresh = ncc_thresh
        self.ts = template_size
        self.ss = search_size
        self.window_size = window_size
        self.ncc_center_penalty = ncc_center_penalty
        self.ncc_boundary_penalty = ncc_boundary_penalty
        self.hard_reset = hard_reset
        self.collect_rollout = collect_rollout
        if reset_velocity_mode not in ('legacy', 'previous_position', 'zero', 'preserve'):
            raise ValueError(f'unsupported reset_velocity_mode: {reset_velocity_mode}')
        self.reset_velocity_mode = reset_velocity_mode
        if low_conf_policy not in (
                'sync_stnet', 'kalman', 'async_stnet', 'hold', 'hold_async'):
            raise ValueError(f'unsupported low_conf_policy: {low_conf_policy}')
        self.low_conf_policy = low_conf_policy
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_velocity_decay = kalman_velocity_decay
        if stnet_cooldown < 0:
            raise ValueError('stnet_cooldown must be non-negative')
        self.stnet_cooldown = stnet_cooldown
        self.disable_tier2 = disable_tier2
        self.severe_conf_threshold = severe_conf_threshold
        self.severe_epsilon_threshold = severe_epsilon_threshold
        if cooldown_low_conf_mode not in ('hold', 'ncc', 'anchor'):
            raise ValueError('cooldown_low_conf_mode must be hold, ncc, or anchor')
        self.cooldown_low_conf_mode = cooldown_low_conf_mode
        self.trigger_classifier = trigger_classifier
        if trigger_learn_threshold <= 0.0 or trigger_learn_threshold >= 1.0:
            raise ValueError('trigger_learn_threshold must be within (0, 1)')
        self.trigger_learn_threshold = trigger_learn_threshold
        self.device = device
        # Persistent pupil template patch (1, C, ts, ts), updated each confirmed frame.
        # NCC matches THIS saved template against the current-frame search window,
        # rather than re-cropping a "template" from the current frame.
        self.template_patch = None
        self.template_center_xy = None

    def _crop(self, frame, center_xy, size):
        """Crop a (C, size, size) patch from frame (C, H, W) centered at center_xy
        (normalized), zero-padded at borders. The center is clamped into the valid
        pixel range so a diverged Kalman prediction cannot produce an empty crop."""
        c, h, w = frame.shape
        cx = int(round(min(max(center_xy[0], 0.0), 1.0) * (w - 1)))
        cy = int(round(min(max(center_xy[1], 0.0), 1.0) * (h - 1)))
        half = size // 2
        patch = torch.zeros(c, size, size, device=frame.device, dtype=frame.dtype)
        y0, y1 = cy - half, cy + half + 1
        x0, x1 = cx - half, cx + half + 1
        sy0, sy1 = max(y0, 0), min(y1, h)
        sx0, sx1 = max(x0, 0), min(x1, w)
        py0, px0 = sy0 - y0, sx0 - x0
        patch[:, py0:py0 + (sy1 - sy0), px0:px0 + (sx1 - sx0)] = frame[:, sy0:sy1, sx0:sx1]
        return patch

    def _ncc_match(self, frame, pred_xy):
        """Match the stored template patch against a search window in the current
        frame centered on the Kalman prediction. Returns matched normalized
        coordinate and confidence.

        Coordinate mapping: a peak at response cell (r, c) means the template's
        top-left aligns with the search patch at (r, c), so the template CENTER
        (the matched pupil) sits at:
            image = search_top_left + cell + half_template
        """
        search = self._crop(frame, pred_xy, self.ss).unsqueeze(0)
        coord, conf, response = self.ncc(self.template_patch, search, training=False)
        if self.ncc_center_penalty > 0.0 or self.ncc_boundary_penalty > 0.0:
            score = response[0].clone()
            response_h, response_w = score.shape
            grid_y = torch.arange(response_h, device=score.device, dtype=score.dtype)
            grid_x = torch.arange(response_w, device=score.device, dtype=score.dtype)
            center_y = (response_h - 1) / 2.0
            center_x = (response_w - 1) / 2.0
            norm_y = (grid_y - center_y) / max(center_y, 1.0)
            norm_x = (grid_x - center_x) / max(center_x, 1.0)
            distance_sq = norm_y[:, None] ** 2 + norm_x[None, :] ** 2
            score -= self.ncc_center_penalty * distance_sq
            if self.ncc_boundary_penalty > 0.0:
                score[0, :] -= self.ncc_boundary_penalty
                score[-1, :] -= self.ncc_boundary_penalty
                score[:, 0] -= self.ncc_boundary_penalty
                score[:, -1] -= self.ncc_boundary_penalty
            peak_index = int(score.argmax().item())
            peak_y, peak_x = divmod(peak_index, response_w)
            local = np.array([
                peak_x / max(response_w - 1, 1),
                peak_y / max(response_h - 1, 1),
            ])
            confidence = float(response[0, peak_y, peak_x])
        else:
            local = coord[0].cpu().numpy()
            confidence = float(conf[0])
        c, h, w = frame.shape
        half_s = self.ss // 2
        half_t = self.ts // 2
        ncc_out = self.ss - self.ts + 1          # response map side length
        cx = int(round(min(max(pred_xy[0], 0.0), 1.0) * (w - 1)))
        cy = int(round(min(max(pred_xy[1], 0.0), 1.0) * (h - 1)))
        cell_x = local[0] * (ncc_out - 1)
        cell_y = local[1] * (ncc_out - 1)
        abs_x = (cx - half_s + cell_x + half_t) / (w - 1)
        abs_y = (cy - half_s + cell_y + half_t) / (h - 1)
        boundary_peak = bool(
            local[0] <= 0.0 or local[0] >= 1.0
            or local[1] <= 0.0 or local[1] >= 1.0
        )
        return (
            np.array([float(np.clip(abs_x, 0, 1)), float(np.clip(abs_y, 0, 1))]),
            confidence, boundary_peak,
        )

    def _update_template(self, frame, center_xy):
        """Refresh the persistent pupil template using the confirmed pupil
        position (lets the template track gradual appearance changes)."""
        self.template_patch = self._crop(frame, center_xy, self.ts).unsqueeze(0)
        self.template_center_xy = np.asarray(center_xy, dtype=np.float64).copy()

    def _trigger_risk(self, confidence, epsilon, boundary_peak, template_age):
        """Compute learned trigger risk score in [0, 1]. Returns None when
        no classifier has been provided so the caller can fall back to the
        fixed severe-bypass rules."""
        classifier = self.trigger_classifier
        if classifier is None:
            return None
        epsilon_value = classifier['epsilon_fill']
        epsilon_valid = 0.0
        if epsilon is not None and np.isfinite(epsilon):
            epsilon_value = float(epsilon)
            epsilon_valid = 1.0
        features = np.array([
            float(confidence),
            1.0 if boundary_peak else 0.0,
            epsilon_valid,
            epsilon_value,
            float(template_age),
        ], dtype=np.float64)
        normalized = (features - classifier['mean']) / classifier['std']
        logit = float(np.dot(normalized, classifier['weights']) + classifier['bias'])
        return 1.0 / (1.0 + np.exp(-logit))

    def _run_stnet(self, window_buffer):
        """Run STNet on a BOUNDED sliding window (window_size frames) and return
        the current-frame pupil coordinate via the last bidirectional step.

        Efficiency note (plan P0): STNet formerly ran on the whole growing 0..t
        window, i.e. O(t) per frame -> O(T^2) per sequence, which inflated the
        per-frame latency ~100x. We cap the window to `window_size` (default 16),
        making each triggered call O(window_size). A true O(1) incremental hidden
        cache is NOT applicable here because the temporal model is a *bidirectional*
        GRU whose backward pass requires the whole window; a bounded window is the
        correct constant-cost alternative for streaming."""
        window = torch.stack(list(window_buffer), dim=0).unsqueeze(0).float()  # (1,W,2,60,80)
        coord = self.stnet.last_step(window.to(self.device))                   # (1,2)
        return coord[0].cpu().numpy()

    def _new_kalman_filter(self):
        return KalmanFilter2D(
            process_noise=self.kalman_process_noise,
            measurement_noise=self.kalman_measurement_noise,
            velocity_decay=self.kalman_velocity_decay,
        )

    @staticmethod
    def _hold_filter(kf, position):
        kf.x[:2] = position
        kf.x[2:] = 0.0
        kf.prev_pos = np.asarray(position, dtype=np.float64).copy()
        return kf.x[:2].copy()

    def _reset_filter(self, kf, stnet_xy, previous_position, previous_velocity):
        if self.reset_velocity_mode == 'previous_position':
            return kf.reset(stnet_xy, previous_pos=previous_position)
        if self.reset_velocity_mode == 'zero':
            return kf.reset(stnet_xy, velocity=(0.0, 0.0))
        if self.reset_velocity_mode == 'preserve':
            return kf.reset(stnet_xy, velocity=previous_velocity)
        return kf.reset(stnet_xy)

    def _replay(self, observations, end_index):
        """Rebuild Kalman state and corrected coordinates through end_index."""
        kf = self._new_kalman_filter()
        coords = np.zeros((end_index + 1, 2), dtype=np.float64)
        resets = 0
        for frame_id in range(end_index + 1):
            observation = observations[frame_id]
            if frame_id == 0:
                kf.initialize(observation['stnet'])
                coords[frame_id] = observation['stnet']
                continue

            previous_position = kf.x[:2].copy()
            previous_velocity = kf.x[2:].copy()
            prediction = kf.predict()
            if observation.get('anchor') is not None:
                output = self._hold_filter(kf, observation['anchor'])
            elif observation.get('hold', False):
                output = self._hold_filter(kf, previous_position)
            elif observation['ncc'] is None:
                output = prediction
            else:
                output = kf.update(observation['ncc'], observation['r_ncc'])

            stnet_xy = observation['stnet']
            if stnet_xy is not None:
                if self.hard_reset and _pixel_dist(prediction, stnet_xy) > self.tau3:
                    output = self._reset_filter(kf, stnet_xy, previous_position, previous_velocity)
                    resets += 1
                else:
                    output = kf.update(stnet_xy, 0.1)
            coords[frame_id] = output
        return kf, coords, resets

    @torch.no_grad()
    def track_sequence(self, event_frames):
        """Track one sequence with a Kalman/NCC main path and async STNet correction."""
        sequence_started_at = time.perf_counter()
        event_frames = event_frames.to(self.device)
        seq_len = event_frames.shape[0]

        kf = self._new_kalman_filter()
        worker = _AsyncSTNetWorker(self.stnet, self.device)
        window = deque(maxlen=self.window_size)
        self.template_patch = None
        self.template_center_xy = None
        coords = np.zeros((seq_len, 2), dtype=np.float64)
        online_coords = np.zeros((seq_len, 2), dtype=np.float64)
        observations = [
            {'ncc': None, 'r_ncc': 1.0, 'stnet': None,
             'hold': False, 'anchor': None}
            for _ in range(seq_len)
        ]
        pending = set()
        template_frame_id = -1
        last_stnet_frame = -10 ** 9
        delivery_lag_frames = []
        delivery_lag_ms = []
        stats = {
            'tier1': 0, 'tier2': 0, 'tier3': 0, 'stnet_calls': 0,
            'tier3_bootstrap': 0, 'tier3_low_conf': 0,
            'tier3_large_error': 0, 'low_conf_kalman': 0,
            'low_conf_async': 0, 'low_conf_hold': 0,
            'cooldown_skips': 0, 'low_conf_cooldown_skips': 0,
            'low_conf_cooldown_ncc': 0,
            'large_error_cooldown_skips': 0, 'tier2_skips': 0,
            'severe_cooldown_bypasses': 0, 'boundary_peaks': 0,
            'low_conf_boundary_peaks': 0, 'large_error_boundary_peaks': 0,
            'ncc_calls': 0, 'resets': 0,
            'critical_ms': 0.0, 'async_ms': 0.0,
            'sync_stnet_ms': 0.0, 'async_results': 0,
            'confidence_bins': {
                'lt_0.3': 0, '0.3_to_0.4': 0, '0.4_to_0.5': 0,
                '0.5_to_0.6': 0, 'ge_0.6': 0,
            },
            'epsilon_bins': {
                'lt_3': 0, '3_to_5': 0, '5_to_7': 0,
                '7_to_10': 0, 'ge_10': 0,
            },
            'tier3_records': [],
            'rollout_samples': [],
        }

        def apply_results(results, current_frame):
            nonlocal kf, template_frame_id
            if not results:
                return
            for result in results:
                if 'error' in result:
                    raise RuntimeError('asynchronous STNet inference failed') from result['error']
                frame_id = result['frame_id']
                observations[frame_id]['stnet'] = result['coord']
                pending.discard(frame_id)
                stats['async_ms'] += result['compute_ms']
                stats['async_results'] += 1
                delivery_lag_frames.append(max(0, current_frame - frame_id))
                delivery_lag_ms.append(
                    (result['finished_at'] - result['submitted_at']) * 1000
                )
                if frame_id > template_frame_id:
                    self._update_template(event_frames[frame_id], result['coord'])
                    template_frame_id = frame_id
            kf, corrected, resets = self._replay(observations, current_frame)
            coords[:current_frame + 1] = corrected
            stats['resets'] = resets

        try:
            for frame_id in range(seq_len):
                frame = event_frames[frame_id]
                window.append(frame)
                frame_started_at = time.perf_counter()
                apply_results(worker.poll(), frame_id - 1)

                if not kf.is_initialized():
                    output = self._run_stnet(window)
                    kf.initialize(output)
                    observations[frame_id]['stnet'] = output
                    self._update_template(frame, output)
                    template_frame_id = frame_id
                    stats['stnet_calls'] += 1
                    stats['tier3'] += 1
                    stats['tier3_bootstrap'] += 1
                    last_stnet_frame = frame_id
                else:
                    previous_position = kf.x[:2].copy()
                    previous_velocity = kf.x[2:].copy()
                    prediction = kf.predict()
                    ncc_xy, confidence, boundary_peak = self._ncc_match(frame, prediction)
                    rollout_sample = None
                    if self.collect_rollout:
                        rollout_sample = {
                            'frame_id': frame_id,
                            'template_frame_id': template_frame_id,
                            'template_center': self.template_center_xy.copy(),
                            'search_center': prediction.copy(),
                            'ncc_coord': ncc_xy.copy(),
                            'confidence': confidence,
                            'boundary_peak': boundary_peak,
                            'epsilon': None,
                            'tier': None,
                        }
                    stats['ncc_calls'] += 1
                    if boundary_peak:
                        stats['boundary_peaks'] += 1
                    if confidence < 0.3:
                        stats['confidence_bins']['lt_0.3'] += 1
                    elif confidence < 0.4:
                        stats['confidence_bins']['0.3_to_0.4'] += 1
                    elif confidence < 0.5:
                        stats['confidence_bins']['0.4_to_0.5'] += 1
                    elif confidence < 0.6:
                        stats['confidence_bins']['0.5_to_0.6'] += 1
                    else:
                        stats['confidence_bins']['ge_0.6'] += 1
                    r_ncc = max(0.1, 1.0 - confidence)
                    trigger_stnet = False
                    cooldown_active = (
                        self.stnet_cooldown > 0
                        and frame_id - last_stnet_frame <= self.stnet_cooldown
                    )

                    if (confidence < self.ncc_thresh
                            and self.low_conf_policy != 'sync_stnet'):
                        # Treat low NCC confidence as a missing local observation.
                        # Hold modes suppress unconstrained velocity drift; async
                        # modes still request delayed STNet correction.
                        hold_position = self.low_conf_policy in ('hold', 'hold_async')
                        if hold_position:
                            observations[frame_id]['hold'] = True
                            output = self._hold_filter(kf, previous_position)
                            stats['low_conf_hold'] += 1
                        else:
                            output = prediction
                            stats['low_conf_kalman'] += 1
                        if self.low_conf_policy in ('async_stnet', 'hold_async'):
                            stats['tier2'] += 1
                            stats['low_conf_async'] += 1
                            trigger_stnet = True
                            if rollout_sample is not None:
                                rollout_sample['tier'] = 'tier2'
                        else:
                            stats['tier1'] += 1
                            if rollout_sample is not None:
                                rollout_sample['tier'] = 'tier1'
                    elif (confidence < self.ncc_thresh and cooldown_active
                          and not (confidence < self.severe_conf_threshold
                                  and boundary_peak)
                          and not (
                              self.trigger_classifier is not None
                              and self._trigger_risk(
                                  confidence, None, boundary_peak,
                                  frame_id - template_frame_id) >= self.trigger_learn_threshold)):
                        if self.cooldown_low_conf_mode == 'ncc':
                            cooldown_r = max(1.0, 4.0 * (1.0 - confidence))
                            observations[frame_id]['ncc'] = ncc_xy
                            observations[frame_id]['r_ncc'] = cooldown_r
                            output = kf.update(ncc_xy, cooldown_r)
                            stats['low_conf_cooldown_ncc'] += 1
                        elif self.cooldown_low_conf_mode == 'anchor':
                            anchor = self.template_center_xy.copy()
                            observations[frame_id]['anchor'] = anchor
                            output = self._hold_filter(kf, anchor)
                            stats['low_conf_hold'] += 1
                        else:
                            observations[frame_id]['hold'] = True
                            output = self._hold_filter(kf, previous_position)
                            stats['low_conf_hold'] += 1
                        stats['tier1'] += 1
                        stats['cooldown_skips'] += 1
                        stats['low_conf_cooldown_skips'] += 1
                        if rollout_sample is not None:
                            rollout_sample['tier'] = 'tier1'
                    elif confidence < self.ncc_thresh:
                        if cooldown_active:
                            stats['severe_cooldown_bypasses'] += 1
                        if rollout_sample is not None:
                            rollout_sample['tier'] = 'tier3_low_conf'
                        # No reliable local observation exists: tier3 blocks for
                        # global localization under the existing safe policy.
                        stats['tier3_records'].append({
                            'cause': 'low_conf', 'frame_id': frame_id,
                            'prediction': prediction.copy(), 'ncc': ncc_xy.copy(),
                            'boundary_peak': boundary_peak,
                        })
                        sync_started_at = time.perf_counter()
                        stnet_xy = self._run_stnet(window)
                        stats['sync_stnet_ms'] += (
                            time.perf_counter() - sync_started_at
                        ) * 1000
                        observations[frame_id]['stnet'] = stnet_xy
                        if self.hard_reset and _pixel_dist(prediction, stnet_xy) > self.tau3:
                            output = self._reset_filter(kf, stnet_xy, previous_position, previous_velocity)
                            stats['resets'] += 1
                        else:
                            output = kf.update(stnet_xy, 0.1)
                        self._update_template(frame, stnet_xy)
                        template_frame_id = frame_id
                        stats['tier3'] += 1
                        stats['tier3_low_conf'] += 1
                        if boundary_peak:
                            stats['low_conf_boundary_peaks'] += 1
                        stats['stnet_calls'] += 1
                        last_stnet_frame = frame_id
                    else:
                        observations[frame_id]['ncc'] = ncc_xy
                        observations[frame_id]['r_ncc'] = r_ncc
                        output = kf.update(ncc_xy, r_ncc)
                        epsilon = _pixel_dist(prediction, ncc_xy)
                        if rollout_sample is not None:
                            rollout_sample['epsilon'] = epsilon
                        if epsilon < 3.0:
                            stats['epsilon_bins']['lt_3'] += 1
                        elif epsilon < 5.0:
                            stats['epsilon_bins']['3_to_5'] += 1
                        elif epsilon < 7.0:
                            stats['epsilon_bins']['5_to_7'] += 1
                        elif epsilon < 10.0:
                            stats['epsilon_bins']['7_to_10'] += 1
                        else:
                            stats['epsilon_bins']['ge_10'] += 1
                        if epsilon < self.tau1:
                            stats['tier1'] += 1
                            if rollout_sample is not None:
                                rollout_sample['tier'] = 'tier1'
                        elif epsilon < self.tau2:
                            if self.disable_tier2 or cooldown_active:
                                stats['tier1'] += 1
                                stats['tier2_skips'] += 1
                                if cooldown_active:
                                    stats['cooldown_skips'] += 1
                                if rollout_sample is not None:
                                    rollout_sample['tier'] = 'tier1'
                            else:
                                stats['tier2'] += 1
                                trigger_stnet = True
                                if rollout_sample is not None:
                                    rollout_sample['tier'] = 'tier2'
                        elif (cooldown_active
                              and epsilon < self.severe_epsilon_threshold
                              and not (
                                  self.trigger_classifier is not None
                                  and self._trigger_risk(
                                      confidence, epsilon, boundary_peak,
                                      frame_id - template_frame_id
                                  ) >= self.trigger_learn_threshold)):
                            if self.cooldown_low_conf_mode == 'anchor':
                                anchor = self.template_center_xy.copy()
                                observations[frame_id]['ncc'] = None
                                observations[frame_id]['anchor'] = anchor
                                output = self._hold_filter(kf, anchor)
                            elif self.cooldown_low_conf_mode == 'hold':
                                observations[frame_id]['ncc'] = None
                                observations[frame_id]['hold'] = True
                                output = self._hold_filter(kf, previous_position)
                            stats['tier1'] += 1
                            stats['cooldown_skips'] += 1
                            stats['large_error_cooldown_skips'] += 1
                            if rollout_sample is not None:
                                rollout_sample['tier'] = 'tier1'
                        else:
                            if cooldown_active:
                                stats['severe_cooldown_bypasses'] += 1
                            if rollout_sample is not None:
                                rollout_sample['tier'] = 'tier3_large_error'
                            stats['tier3_records'].append({
                                'cause': 'large_error', 'frame_id': frame_id,
                                'prediction': prediction.copy(), 'ncc': ncc_xy.copy(),
                                'boundary_peak': boundary_peak,
                            })
                            sync_started_at = time.perf_counter()
                            stnet_xy = self._run_stnet(window)
                            stats['sync_stnet_ms'] += (
                                time.perf_counter() - sync_started_at
                            ) * 1000
                            observations[frame_id]['stnet'] = stnet_xy
                            if self.hard_reset and _pixel_dist(prediction, stnet_xy) > self.tau3:
                                output = self._reset_filter(kf, stnet_xy, previous_position, previous_velocity)
                                stats['resets'] += 1
                            else:
                                output = kf.update(stnet_xy, 0.1)
                            self._update_template(frame, stnet_xy)
                            template_frame_id = frame_id
                            stats['tier3'] += 1
                            stats['tier3_large_error'] += 1
                            if boundary_peak:
                                stats['large_error_boundary_peaks'] += 1
                            stats['stnet_calls'] += 1
                            last_stnet_frame = frame_id

                    if trigger_stnet:
                        worker.submit(frame_id, tuple(window), time.perf_counter())
                        pending.add(frame_id)
                        stats['stnet_calls'] += 1
                        last_stnet_frame = frame_id
                    if rollout_sample is not None:
                        stats['rollout_samples'].append(rollout_sample)

                online_coords[frame_id] = output
                coords[frame_id] = output
                stats['critical_ms'] += (
                    time.perf_counter() - frame_started_at
                ) * 1000

            apply_results(worker.close(), seq_len - 1)
        finally:
            if worker.thread.is_alive():
                worker.close()

        if pending:
            raise RuntimeError(f'{len(pending)} asynchronous STNet results were not returned')

        stats['wall_ms'] = (time.perf_counter() - sequence_started_at) * 1000
        stats['avg_critical_ms'] = stats['critical_ms'] / max(seq_len, 1)
        stats['avg_async_ms'] = stats['async_ms'] / max(seq_len, 1)
        stats['avg_sync_stnet_ms'] = stats['sync_stnet_ms'] / max(seq_len, 1)
        stats['avg_wall_ms'] = stats['wall_ms'] / max(seq_len, 1)
        stats['avg_delivery_lag_frames'] = (
            float(np.mean(delivery_lag_frames)) if delivery_lag_frames else 0.0
        )
        stats['avg_delivery_lag_ms'] = (
            float(np.mean(delivery_lag_ms)) if delivery_lag_ms else 0.0
        )
        stats['stnet_ratio'] = stats['stnet_calls'] / max(seq_len, 1)
        stats['online_coords'] = online_coords
        return coords, stats


if __name__ == '__main__':
    from models.STNet import STNet
    from models.ncc_matcher import NCCMatcher

    stnet = STNet().eval()
    ncc = NCCMatcher().eval()
    tracker = KTCalTracker(stnet, ncc)
    frames = torch.rand(100, 2, 60, 80)
    coords, stats = tracker.track_sequence(frames)
    print('coords', coords.shape)
    print('stats', {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})
