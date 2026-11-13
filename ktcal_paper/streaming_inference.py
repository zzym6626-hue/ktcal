"""Sequential causal STNet feature updates with on-demand coordinate heads."""
import math

import numpy as np
import torch
from torch import nn

from .offline_experiment import spatial_view
from .online_inference import GraphDecoder


def _flip_axis(model, axis):
    if axis is None:
        axis = getattr(getattr(model, 'args', None), 'stnet_flip_axis_px', 79.)
    if not math.isfinite(axis) or not 79. <= axis <= 80.:
        raise ValueError('Finite streaming flip axis in[79,80] required')
    return float(axis)


class ViewedStreamingStep(nn.Module):
    """Keep view transforms inside the captured feature-update graph."""
    def __init__(self, model, views, divisor=1.):
        super().__init__()
        self.model, self.views, self.divisor = model, tuple(views), divisor

    def initial_state(self, frame):
        shape = (len(frame) * len(self.views),) + tuple(frame.shape[1:])
        return self.model.initial_state(frame.new_empty(shape))

    def stream_step(self, frame, state):
        inputs = torch.cat([spatial_view(frame, view) for view in self.views], 0) / self.divisor
        return self.model.stream_step(inputs, state)


class PersistentStreamingStep(nn.Module):
    """Copy next recurrent state into stable buffers inside a CUDA capture."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def initial_state(self, frame):
        return self.model.initial_state(frame)

    def stream_step(self, frame, state):
        features, next_state = self.model.stream_step(frame, state)
        for destination, source in zip(state, next_state):
            destination.copy_(source)
        return features, tuple(state)


class StatefulGraphStep(nn.Module):
    def __init__(self, model, enabled=True, persistent_state=False):
        super().__init__()
        self.model, self.enabled = model, enabled
        self.persistent_state = bool(persistent_state)
        self.captured_model = PersistentStreamingStep(model) if persistent_state else model
        self.graph = self.inputs = self.output = None

    @torch.inference_mode()
    def forward(self, frame, state):
        if state is None:
            state = self.model.initial_state(frame)
        if not self.enabled or frame.device.type != 'cuda':
            return self.model.stream_step(frame, state)
        values = (frame,) + tuple(state)
        signatures = lambda tensors: [(v.shape, v.device, v.dtype) for v in tensors]
        if self.graph is None or signatures(values) != signatures(self.inputs):
            self.inputs = tuple(v.clone() for v in values)
            stream = torch.cuda.Stream(device=frame.device)
            stream.wait_stream(torch.cuda.current_stream(frame.device))
            with torch.cuda.stream(stream):
                for _ in range(4):
                    self.captured_model.stream_step(self.inputs[0], self.inputs[1:])
            torch.cuda.current_stream(frame.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(frame.device), torch.cuda.graph(self.graph, stream=stream):
                self.output = self.captured_model.stream_step(self.inputs[0], self.inputs[1:])
        for destination, source in zip(self.inputs, values):
            if destination is not source:
                destination.copy_(source)
        self.graph.replay()
        return self.output


class StreamingHeadDecoder(nn.Module):
    def __init__(self, model, views, temperature=1., ensemble='mean', flip_axis_px=None):
        super().__init__()
        if (not views or not math.isfinite(temperature) or temperature <= 0
                or ensemble not in ('mean', 'median')):
            raise ValueError('Valid streaming view decoder required')
        for dx, dy, flip in views:
            if abs(dx) >= 80 or abs(dy) >= 60 or flip not in (False, True):
                raise ValueError('Invalid streaming spatial view')
        self.model = model
        self.flip_axis_px = _flip_axis(model, flip_axis_px)
        self.views, self.temperature, self.ensemble = tuple(views), temperature, ensemble
        self.register_buffer('x_grid', torch.arange(80, dtype=torch.float32))
        self.register_buffer('y_grid', torch.arange(60, dtype=torch.float32))

    def forward(self, features):
        logits_x, logits_y = self.model.coordinate_logits(features)
        x = (logits_x / self.temperature).softmax(-1)
        y = (logits_y / self.temperature).softmax(-1)
        pixels = torch.stack(((x * self.x_grid).sum(-1), (y * self.y_grid).sum(-1)), -1)
        corrected = []
        for output, (dx, dy, flip) in zip(pixels.chunk(len(self.views), 0), self.views):
            cx = self.flip_axis_px - output[:, 0] if flip else output[:, 0]
            corrected.append(torch.stack((cx - dx, output[:, 1] - dy), -1))
        values = torch.stack(corrected)
        if self.ensemble == 'mean':
            return values.mean(0)
        values = values.sort(0).values
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


class StreamingPredictor:
    def __init__(self, model, divisor=1., views=((0, 0, False),),
                 temperature=1., ensemble='mean', cuda_graph=True, warm_start_steps=0,
                 flip_axis_px=None, persistent_state=False):
        if not math.isfinite(divisor) or divisor <= 0:
            raise ValueError('Positive streaming input divisor required')
        if isinstance(warm_start_steps, bool) or not isinstance(warm_start_steps, int) or warm_start_steps < 0:
            raise ValueError('Nonnegative integer warm-start steps required')
        self.model = model.eval()
        self.flip_axis_px = _flip_axis(model, flip_axis_px)
        device = next(model.parameters()).device
        self.step = StatefulGraphStep(ViewedStreamingStep(model, views, divisor).eval(),
                                      enabled=cuda_graph, persistent_state=persistent_state)
        head = StreamingHeadDecoder(model, views, temperature, ensemble,
                                    self.flip_axis_px).to(device).eval()
        self.decoder = GraphDecoder(head) if cuda_graph else head
        self.divisor, self.views = divisor, tuple(views)
        self.warm_start_steps = warm_start_steps
        self.reset()

    def reset(self):
        self.index, self.state, self.features = -1, None, None
        self.feature_updates = self.feature_forward_passes = 0
        self.received_frame_updates = self.synthetic_feature_updates = 0
        self.coordinate_head_requests = self.coordinate_head_forward_passes = 0

    @torch.inference_mode()
    def observe(self, image):
        self.batch = len(image)
        if self.index == -1:
            for _ in range(self.warm_start_steps):
                self.features, self.state = self.step(image, self.state)
                self.feature_updates += self.batch
                self.synthetic_feature_updates += self.batch
                self.feature_forward_passes += 1
        self.features, self.state = self.step(image, self.state)
        self.index += 1
        self.feature_updates += self.batch
        self.received_frame_updates += self.batch
        self.feature_forward_passes += 1

    @torch.inference_mode()
    def __call__(self, history, mask):
        if self.features is None or history.shape[1] != self.index + 1:
            raise ValueError('Streaming request must match the received prefix')
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.batch,):
            raise ValueError('Streaming request mask must match the received batch')
        requested = int(mask.sum())
        if not requested:
            return np.empty((0, 2), dtype=np.float32)
        features = self.features
        if not mask.all():
            selected = torch.as_tensor(mask, device=features.device)
            features = torch.cat([part[selected] for part in features.chunk(len(self.views), 0)], 0)
        result = self.decoder(features)
        self.coordinate_head_requests += requested
        self.coordinate_head_forward_passes += 1
        return result.detach().cpu().numpy().copy()


@torch.inference_mode()
def predict_streaming_sequences(model, frames, views, *, divisor=1., temperature=1.,
                                 ensemble='mean', batch_size=4, warm_start_steps=0,
                                 flip_axis_px=None):
    """Cache causal full-history coordinates for validation parameter selection.

    Independent records are batched only here. This is never a latency
    measurement; final candidates must use the stateful online evaluator.
    """
    if batch_size < 1 or not math.isfinite(divisor) or divisor <= 0:
        raise ValueError('Positive selection batch size and input divisor required')
    if frames.ndim != 5 or len(frames) < 1 or frames.shape[1] < 1:
        raise ValueError('Nonempty B,T,2,H,W selection recordings required')
    if isinstance(warm_start_steps, bool) or not isinstance(warm_start_steps, int) or warm_start_steps < 0:
        raise ValueError('Nonnegative integer warm-start steps required')
    model.eval()
    device = next(model.parameters()).device
    decoder = StreamingHeadDecoder(model, views, temperature, ensemble,
                                   flip_axis_px).to(device).eval()
    outputs = []
    for begin in range(0, len(frames), batch_size):
        current = frames[begin:begin + batch_size].to(device, dtype=torch.float32)
        batch, length = current.shape[:2]
        if warm_start_steps:
            current = torch.cat((current[:, :1].expand(-1, warm_start_steps, -1, -1, -1),
                                 current), 1)
        inputs = torch.cat([spatial_view(current, view) for view in views], 0) / divisor
        features = model.extract_temporal(inputs)[:, warm_start_steps:]
        outputs.append(decoder(features.reshape(-1, features.shape[-1])).reshape(
            batch, length, 2).cpu().numpy().copy())
    return np.concatenate(outputs)
