"""Causal STNet with identical full-sequence and stateful evaluation."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .STNet import STNet, DecoupledSTBlock


class CausalSTBlock(DecoupledSTBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        convolution = self.temporal[0]
        convolution.padding = (0, convolution.padding[1], convolution.padding[2])

    def forward(self, value):
        spatial = self.spatial(value)
        return self.temporal(F.pad(spatial, (0, 0, 0, 0, 2, 0)))

    def step(self, value, previous):
        spatial = self.spatial(value)
        context = torch.cat((previous, spatial), 2)
        return self.temporal(context), context[:, :, -2:]


class StreamingSTNet(STNet):
    """Update the convolution/GRU state once for each received event frame.

    ``forward`` is for full-sequence supervised training/evaluation. In eval
    mode ``stream_step`` gives exactly its current feature using only one new
    frame and state from the past. Coordinate heads may be invoked separately.
    """
    def __init__(self, args=None, hidden_size=256):
        if isinstance(hidden_size, bool) or hidden_size not in (128, 256):
            raise ValueError('Streaming GRU hidden size must be 128 or 256')
        options = dict(vars(args)) if args is not None else {}
        options.update(stnet_unidirectional=True, stnet_readout='last')
        super().__init__(SimpleNamespace(**options), hidden_size=hidden_size // 2)
        self.streaming_hidden_size = hidden_size
        kernel = getattr(self.args, 'stnet_temporal_spatial_kernel', 1)
        self.block1 = CausalSTBlock(2, 32, 7, 3, pool=True,
                                   temporal_spatial_kernel=kernel)
        self.block2 = CausalSTBlock(32, 64, 5, 2, pool=True,
                                   temporal_spatial_kernel=kernel)
        k3 = getattr(self.args, 'stnet_block3_kernel', 5)
        self.block3 = CausalSTBlock(64, 128, k3, k3 // 2, pool=False,
                                   dropout=True, temporal_spatial_kernel=kernel)

    def _spatial_features(self, value):
        batch, channels, length, height, width = value.shape
        pooled = F.adaptive_avg_pool2d(
            value.permute(0, 2, 1, 3, 4).reshape(batch * length, channels, height, width),
            (4, 4)).reshape(batch, length, 128, 4, 4)
        pooled = self.spatialdropout(pooled)
        return self.layernorm(pooled.reshape(batch, length, 2048))

    def extract_temporal(self, frames):
        if frames.ndim != 5 or frames.shape[2] != 2 or frames.shape[1] < 1:
            raise ValueError('Expected nonempty B,T,2,H,W event frames')
        spatial = self.block3(self.block2(self.block1(frames.permute(0, 2, 1, 3, 4))))
        return self.gru(self._spatial_features(spatial))[0]

    def initial_state(self, frame):
        if frame.ndim != 4 or frame.shape[1] != 2:
            raise ValueError('Expected B,2,H,W current frame')
        batch, _, height, width = frame.shape
        return (
            frame.new_zeros(batch, 32, 2, height, width),
            frame.new_zeros(batch, 64, 2, height // 3, width // 3),
            frame.new_zeros(batch, 128, 2, height // 9, width // 9),
            frame.new_zeros(1, batch, self.streaming_hidden_size),
        )

    def stream_step(self, frame, state=None):
        if self.training:
            raise ValueError('Stateful streaming requires eval mode')
        if state is None:
            state = self.initial_state(frame)
        if len(state) != 4:
            raise ValueError('Expected three temporal caches and a GRU state')
        value, s1 = self.block1.step(frame[:, :, None], state[0])
        value, s2 = self.block2.step(value, state[1])
        value, s3 = self.block3.step(value, state[2])
        sequence, hidden = self.gru(self._spatial_features(value), state[3])
        return sequence[:, 0], (s1, s2, s3, hidden)

    def coordinate_logits(self, features):
        return self.fc_w(features), self.fc_h(features)

    def forward(self, frames, y=None):
        return self.coordinate_logits(self.extract_temporal(frames))

    def forward_with_features(self, frames):
        features = self.extract_temporal(frames)
        x, y = self.coordinate_logits(features)
        return features, x, y

    def last_step(self, frames):
        x, y = self.coordinate_logits(self.extract_temporal(frames)[:, -1])
        mode = getattr(self.args, 'stnet_decode', 'soft')
        if mode == 'argmax':
            return torch.stack((x.argmax(-1).to(x.dtype) / 80.,
                                y.argmax(-1).to(y.dtype) / 60.), -1)
        if mode != 'soft':
            raise ValueError(f'unsupported STNet decode mode: {mode}')
        gx = torch.arange(80, device=x.device, dtype=x.dtype)
        gy = torch.arange(60, device=y.device, dtype=y.dtype)
        return torch.stack(((x.softmax(-1) * gx).sum(-1) / 80.,
                            (y.softmax(-1) * gy).sum(-1) / 60.), -1)

    @torch.no_grad()
    def initialize_from_bidirectional(self, source):
        """Map 128-unit forward/backward groups without mixing GRU gates."""
        state = source.state_dict() if isinstance(source, nn.Module) else source
        if 'gru.weight_ih_l0_reverse' not in state or state['gru.weight_ih_l0'].shape != (384, 2048):
            raise ValueError('Expected the original 128+128 bidirectional STNet')
        own = self.state_dict()
        copied = 0
        for name, value in state.items():
            if name.startswith('gru.') or name not in own:
                continue
            if own[name].shape == value.shape:
                own[name].copy_(value)
                copied += 1
            elif name in ('fc_w.0.weight', 'fc_h.0.weight'):
                own[name].copy_(value[:, :self.streaming_hidden_size])
                copied += 1
        # Preserve the current-frame held-boundary kernel initially. Causal
        # fine-tuning can then learn a distinct second historical coefficient.
        for block in ('block1', 'block2', 'block3'):
            name = block + '.temporal.0.weight'
            original = state[name]
            own[name][:, :, 0].zero_()
            own[name][:, :, 1].copy_(original[:, :, 0])
            own[name][:, :, 2].copy_(original[:, :, 1] + original[:, :, 2])
        hidden = self.streaming_hidden_size
        for name in ('weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0'):
            destination = own['gru.' + name]
            destination.zero_()
            for group, suffix in enumerate(('', '_reverse')[:hidden // 128]):
                original = state['gru.' + name + suffix]
                for gate in range(3):
                    rows = slice(gate * hidden + group * 128,
                                 gate * hidden + (group + 1) * 128)
                    source_rows = slice(gate * 128, (gate + 1) * 128)
                    if name.startswith('weight_hh'):
                        destination[rows, group * 128:(group + 1) * 128].copy_(original[source_rows])
                    else:
                        destination[rows].copy_(original[source_rows])
        return dict(gru_hidden_size=hidden, copied_non_gru_tensors=copied,
                    recurrent_initialization='forward_only' if hidden == 128 else 'two_forward_groups',
                    temporal_initialization='past2_zero_past1_left_current_center_plus_right')
