"""Current-frame residual reusing the first causal backbone's spatial map."""
import torch
from torch import nn
from torch.nn import functional as F

from .current_readout_streaming_stnet import CurrentReadoutStreamingSTNet


class RawCurrentStreamingSTNet(CurrentReadoutStreamingSTNet):
    def __init__(self, args=None, hidden_size=256):
        super().__init__(args, hidden_size=hidden_size)
        self.raw_current_encoder = nn.Sequential(
            nn.Conv2d(32, 32, 5, stride=3, padding=2), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 7 * 9, 128), nn.LayerNorm(128), nn.ReLU(),
        )
        self.raw_current_head = nn.Sequential(nn.Dropout(.2), nn.Linear(128, 140))
        self.raw_current_scale = nn.Parameter(torch.zeros(()))

    def _raw_features(self, spatial):
        batch, channels, length, height, width = spatial.shape
        frames = spatial.permute(0, 2, 1, 3, 4).reshape(batch * length, channels, height, width)
        return self.raw_current_encoder(frames).reshape(batch, length, 128)

    def extract_temporal(self, frames):
        if frames.ndim != 5 or frames.shape[2:] != (2, 60, 80) or frames.shape[1] < 1:
            raise ValueError('Expected nonempty B,T,2,60,80 frames')
        spatial = self.block1.spatial(frames.permute(0, 2, 1, 3, 4))
        raw = self._raw_features(spatial)
        value = self.block1.temporal(F.pad(spatial, (0, 0, 0, 0, 2, 0)))
        value = self.block3(self.block2(value))
        current = self._spatial_features(value)
        sequence = self.gru(current)[0]
        return torch.cat((sequence, current, raw), -1)

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
        raw = self.raw_current_encoder(s1[:, :, -1])
        value, s2 = self.block2.step(value, state[1])
        value, s3 = self.block3.step(value, state[2])
        current = self._spatial_features(value)
        sequence, hidden = self.gru(current, state[3])
        return torch.cat((sequence[:, 0], current[:, 0], raw), -1), (s1, s2, s3, hidden)

    def current_logits(self, features):
        start = self.streaming_hidden_size
        logits = self.current_readout(features[..., start:start + 2048])
        return logits[..., :80], logits[..., 80:]

    def raw_logits(self, features):
        logits = self.raw_current_head(features[..., -128:])
        return logits[..., :80], logits[..., 80:]

    def _combined_logits(self, features, current_x, current_y, raw_x, raw_y):
        x, y = super()._coordinate_logits(features, current_x, current_y)
        return x + self.raw_current_scale * raw_x, y + self.raw_current_scale * raw_y

    def _coordinate_logits(self, features, current_x, current_y):
        return self._combined_logits(features, current_x, current_y, *self.raw_logits(features))

    def forward_with_raw_logits(self, frames):
        features = self.extract_temporal(frames)
        current_x, current_y = self.current_logits(features)
        raw_x, raw_y = self.raw_logits(features)
        x, y = self._combined_logits(features, current_x, current_y, raw_x, raw_y)
        return x, y, current_x, current_y, raw_x, raw_y

    def initialize_from_current_readout(self, state):
        result = self.load_state_dict(state, strict=False)
        permitted = {key for key in self.state_dict() if key.startswith('raw_current_')}
        if set(result.missing_keys) != permitted or result.unexpected_keys:
            raise ValueError('Expected a compatible current-readout checkpoint without raw residual')
        return dict(recurrent_initialization='current_readout_with_zero_raw_current_residual')
