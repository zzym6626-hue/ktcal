"""Past-only STNet windows and spatial-view decoding for online correction."""
import math

import numpy as np
import torch
from torch import nn

from .offline_experiment import spatial_view


class LastStepSTNet(nn.Module):
    """Equivalent eval-only STNet readout, without unused historical heads.

    For `last`, the reverse GRU state at the current frame has processed only
    that frame. Evaluate its zero-state gate directly instead of recurring
    through all historical frames. Attention and final-hidden retain BiGRU.
    """
    def __init__(self, model, current_offset=0):
        super().__init__()
        if isinstance(current_offset, bool) or not isinstance(current_offset, int) or current_offset < 0:
            raise ValueError('Nonnegative current-step offset required')
        self.current_offset = current_offset
        self.current_step_logits = True
        self.block1, self.block2, self.block3 = model.block1, model.block2, model.block3
        self.layernorm, self.fc_w, self.fc_h = model.layernorm, model.fc_w, model.fc_h
        self.readout = model.readout
        self.reverse_affine = None
        self.reverse_gru = None
        if self.readout == 'last' and model.gru.bidirectional:
            source = model.gru.state_dict()
            hidden = model.gru.hidden_size
            self.gru = nn.GRU(model.gru.input_size, hidden, batch_first=True)
            self.gru.to(device=model.gru.weight_ih_l0.device, dtype=model.gru.weight_ih_l0.dtype)
            self.gru.load_state_dict({k: v for k, v in source.items() if not k.endswith('_reverse')})
            if current_offset:
                self.reverse_gru = nn.GRU(model.gru.input_size, hidden, batch_first=True).to(
                    device=model.gru.weight_ih_l0.device, dtype=model.gru.weight_ih_l0.dtype)
                self.reverse_gru.load_state_dict({k[:-8]: v for k, v in source.items()
                                                  if k.endswith('_reverse')})
            else:
                self.reverse_affine = nn.Linear(model.gru.input_size, 3 * hidden).to(
                    device=model.gru.weight_ih_l0.device, dtype=model.gru.weight_ih_l0.dtype)
                with torch.no_grad():
                    self.reverse_affine.weight.copy_(source['weight_ih_l0_reverse'])
                    bias = source['bias_ih_l0_reverse'].clone()
                    bias[:2 * hidden] += source['bias_hh_l0_reverse'][:2 * hidden]
                    self.reverse_affine.bias.copy_(bias)
                self.register_buffer('reverse_bias_n', source['bias_hh_l0_reverse'][2 * hidden:].clone())
        else:
            self.gru = model.gru
        self.eval()

    def forward(self, frames):
        if self.training:
            raise ValueError('LastStepSTNet is an inference-only equivalent transform')
        batch, length = frames.shape[:2]
        target = length - 1 - self.current_offset
        if target < 0:
            raise ValueError('Current-step offset exceeds input length')
        x = self.block3(self.block2(self.block1(frames.permute(0, 2, 1, 3, 4))))
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * length, 128, *x.shape[-2:])
        features = self.layernorm(torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
                                  .reshape(batch, length, 2048))
        sequence, hidden = self.gru(features[:, :target + 1] if self.reverse_gru is not None else features)
        last = sequence[:, target]
        if self.reverse_gru is not None:
            reverse, _ = self.reverse_gru(features[:, target:].flip(1))
            last = torch.cat((last, reverse[:, -1]), -1)
        if self.reverse_affine is not None:
            reset, update, candidate = self.reverse_affine(features[:, -1]).chunk(3, -1)
            reverse = (1. - update.sigmoid()) * torch.tanh(
                candidate + reset.sigmoid() * self.reverse_bias_n)
            last = torch.cat((last, reverse), -1)
        elif self.readout == 'final_hidden' and not self.current_offset:
            last = torch.cat((hidden[-2], hidden[-1]), -1)
        elif self.readout == 'attention':
            indices = (target + torch.arange(-5, 6, device=frames.device)).clamp(0, length - 1)
            neighbors = sequence[:, indices]
            scores = (last[:, None] * neighbors).sum(-1) / (sequence.shape[-1] ** .5)
            last = (scores.softmax(-1)[..., None] * neighbors).sum(1)
        return self.fc_w(last)[:, None], self.fc_h(last)[:, None]


class CausalViewDecoder(nn.Module):
    def __init__(self, model, divisor=1., views=((0, 0, False),),
                 temperature=.65, ensemble='median', right_context=0):
        super().__init__()
        if (not math.isfinite(divisor) or divisor <= 0
                or not math.isfinite(temperature) or temperature <= 0
                or ensemble not in ('mean', 'median') or not views):
            raise ValueError('Valid positive decoding configuration required')
        if isinstance(right_context, bool) or not isinstance(right_context, int) or right_context < 0:
            raise ValueError('Nonnegative synthetic right context required')
        for dx, dy, flip in views:
            if abs(dx) >= 80 or abs(dy) >= 60 or flip not in (0, 1, False, True):
                raise ValueError('Invalid spatial view')
        self.model = model.eval()
        self.divisor, self.temperature, self.ensemble = divisor, temperature, ensemble
        self.views = tuple(views)
        self.right_context = right_context
        self.register_buffer('x_grid', torch.arange(80, dtype=torch.float32))
        self.register_buffer('y_grid', torch.arange(60, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor([80., 60.]))

    def forward(self, frames):
        if self.right_context:
            # Held current observations are a causal boundary condition, not
            # future measurements. Read the original current timestep below.
            frames = torch.cat((frames, frames[:, -1:].expand(
                -1, self.right_context, -1, -1, -1)), 1)
        inputs = torch.cat([spatial_view(frames, view) for view in self.views], 0)
        result = self.model(inputs / self.divisor)
        if torch.is_tensor(result):
            if self.right_context:
                raise ValueError('Synthetic context requires per-timestep logits')
            pixels = result * self.scale
        else:
            lx, ly = result
            target = 0 if getattr(self.model, 'current_step_logits', False) else -1 - self.right_context
            x = (lx[:, target] / self.temperature).softmax(-1)
            y = (ly[:, target] / self.temperature).softmax(-1)
            pixels = torch.stack(((x * self.x_grid).sum(-1),
                                  (y * self.y_grid).sum(-1)), -1)
        corrected = []
        for prediction, (dx, dy, flip) in zip(pixels.chunk(len(self.views), 0), self.views):
            x = 79. - prediction[:, 0] if flip else prediction[:, 0]
            corrected.append(torch.stack((x - dx, prediction[:, 1] - dy), -1))
        values = torch.stack(corrected)
        if self.ensemble == 'mean':
            return values.mean(0)
        ordered = values.sort(dim=0).values
        mid = len(self.views) // 2
        return ordered[mid] if len(self.views) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.


class GraphDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.graph = self.input = self.output = None

    @torch.inference_mode()
    def forward(self, frames):
        if frames.device.type != 'cuda':
            return self.model(frames)
        if self.graph is None or (frames.shape, frames.dtype, frames.device) != (
                self.input.shape, self.input.dtype, self.input.device):
            self.input = frames.clone()
            stream = torch.cuda.Stream(device=frames.device)
            stream.wait_stream(torch.cuda.current_stream(frames.device))
            with torch.cuda.stream(stream):
                for _ in range(4):
                    self.model(self.input)
            torch.cuda.current_stream(frames.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(frames.device), torch.cuda.graph(self.graph, stream=stream):
                self.output = self.model(self.input)
        self.input.copy_(frames)
        self.graph.replay()
        return self.output


class CausalPredictor:
    def __init__(self, decoder, window=16, stride=2):
        if (isinstance(window, bool) or not isinstance(window, int) or window < 1
                or isinstance(stride, bool) or not isinstance(stride, int) or stride < 1):
            raise ValueError('Positive integer causal window and stride required')
        self.decoder, self.window, self.stride = decoder, window, stride
        self.capacity = (window - 1) * stride + 1
        self.ring = self.offsets = None
        self.reset()

    def reset(self):
        self.index = -1

    def observe(self, image):
        if self.ring is None or (self.ring.shape[0], self.ring.device, self.ring.dtype) != (
                len(image), image.device, image.dtype):
            self.ring = torch.empty((len(image), self.capacity, *image.shape[1:]),
                                    device=image.device, dtype=image.dtype)
            self.offsets = torch.arange(self.window - 1, -1, -1, device=image.device) * self.stride
        self.index += 1
        self.ring[:, self.index % self.capacity].copy_(image)

    @torch.inference_mode()
    def __call__(self, history, mask):
        if history.shape[1] != self.index + 1:
            raise ValueError('Causal prediction must align with the received frame')
        indices = (self.index - self.offsets).clamp_min(0).remainder(self.capacity)
        inputs = self.ring.index_select(1, indices)
        if not np.asarray(mask).all():
            inputs = inputs[torch.as_tensor(mask, device=inputs.device)]
        return self.decoder(inputs).detach().cpu().numpy().copy()


class CausalOutputFilter:
    def __init__(self, alpha=None, max_step=2.5):
        if (alpha is not None and (not np.isfinite(alpha) or not 0 < alpha <= 1)
                or not np.isfinite(max_step) or max_step <= 0):
            raise ValueError('Valid causal smoothing configuration required')
        self.alpha, self.max_step = alpha, max_step
        self.previous = None

    def __call__(self, value):
        value = np.asarray(value, dtype=np.float64).copy()
        if self.previous is not None and self.alpha is not None:
            mask = np.linalg.norm(value - self.previous, axis=-1) < self.max_step
            value = np.where(np.asarray(mask)[..., None],
                             self.alpha * value + (1. - self.alpha) * self.previous, value)
        self.previous = value.copy()
        return value


def causal_smooth(values, alpha=None, max_step=2.5):
    smoother = CausalOutputFilter(alpha, max_step)
    return np.stack([smoother(value) for value in values])
