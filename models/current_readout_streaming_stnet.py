"""A direct current-spatial coordinate residual alongside causal GRU history."""
import torch
from torch import nn

from .streaming_stnet import StreamingSTNet


class CurrentReadoutStreamingSTNet(StreamingSTNet):
    def __init__(self, args=None, hidden_size=256):
        super().__init__(args, hidden_size=hidden_size)
        self.current_readout = nn.Sequential(
            nn.Linear(2048, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(.2),
            nn.Linear(256, 140),
        )
        self.current_readout_scale = nn.Parameter(torch.zeros(()))

    def extract_temporal(self, frames):
        if frames.ndim != 5 or frames.shape[2:] != (2, 60, 80) or frames.shape[1] < 1:
            raise ValueError('Expected nonempty B,T,2,60,80 frames')
        spatial = self.block3(self.block2(self.block1(frames.permute(0, 2, 1, 3, 4))))
        current = self._spatial_features(spatial)
        sequence = self.gru(current)[0]
        return torch.cat((sequence, current), -1)

    def stream_step(self, frame, state=None):
        if self.training:
            raise ValueError('Stateful streaming requires eval mode')
        if frame.ndim != 4 or frame.shape[1:] != (2, 60, 80):
            raise ValueError('Expected B,2,60,80 current frames')
        if state is None:
            state = self.initial_state(frame)
        if len(state) != 4:
            raise ValueError('Expected three temporal caches and a GRU state')
        value, s1 = self.block1.step(frame[:, :, None], state[0])
        value, s2 = self.block2.step(value, state[1])
        value, s3 = self.block3.step(value, state[2])
        current = self._spatial_features(value)
        sequence, hidden = self.gru(current, state[3])
        return torch.cat((sequence[:, 0], current[:, 0]), -1), (s1, s2, s3, hidden)

    def current_logits(self, features):
        logits = self.current_readout(features[..., self.streaming_hidden_size:])
        return logits[..., :80], logits[..., 80:]

    def _coordinate_logits(self, features, current_x, current_y):
        recurrent = features[..., :self.streaming_hidden_size]
        return (self.fc_w(recurrent) + self.current_readout_scale * current_x,
                self.fc_h(recurrent) + self.current_readout_scale * current_y)

    def coordinate_logits(self, features):
        return self._coordinate_logits(features, *self.current_logits(features))

    def forward_with_current_logits(self, frames):
        features = self.extract_temporal(frames)
        current_x, current_y = self.current_logits(features)
        x, y = self._coordinate_logits(features, current_x, current_y)
        return x, y, current_x, current_y

    def initialize_from_streaming(self, state):
        result = self.load_state_dict(state, strict=False)
        permitted = {'current_readout_scale', 'current_readout.0.weight', 'current_readout.0.bias',
                     'current_readout.1.weight', 'current_readout.1.bias',
                     'current_readout.4.weight', 'current_readout.4.bias'}
        if set(result.missing_keys) != permitted or result.unexpected_keys:
            raise ValueError('Expected a compatible streaming backbone without a current readout')
        return dict(recurrent_initialization='streaming_backbone_with_zero_current_readout_residual')
