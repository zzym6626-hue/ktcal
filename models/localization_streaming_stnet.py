"""A causal dense localization residual over the streaming STNet readout."""
import torch
from torch import nn
from torch.nn import functional as F

from .streaming_stnet import StreamingSTNet


class LocalizationStreamingSTNet(StreamingSTNet):
    def __init__(self, args=None, hidden_size=256):
        super().__init__(args, hidden_size=hidden_size)
        self.localizer_condition = nn.Linear(hidden_size, 32)
        nn.init.zeros_(self.localizer_condition.weight)
        nn.init.zeros_(self.localizer_condition.bias)
        self.localizer = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )
        self.localizer_scale = nn.Parameter(torch.zeros(()))

    def extract_temporal(self, frames):
        if frames.ndim != 5 or frames.shape[2:] != (2, 60, 80) or frames.shape[1] < 1:
            raise ValueError('Expected nonempty B,T,2,60,80 frames')
        first = self.block1(frames.permute(0, 2, 1, 3, 4))
        final = self.block3(self.block2(first))
        sequence = self.gru(self._spatial_features(final))[0]
        spatial = first.permute(0, 2, 1, 3, 4).flatten(2)
        return torch.cat((sequence, spatial), -1)

    def stream_step(self, frame, state=None):
        if self.training:
            raise ValueError('Stateful streaming requires eval mode')
        if frame.ndim != 4 or frame.shape[1:] != (2, 60, 80):
            raise ValueError('Expected B,2,60,80 current frames')
        if state is None:
            state = self.initial_state(frame)
        if len(state) != 4:
            raise ValueError('Expected three temporal caches and a GRU state')
        first, s1 = self.block1.step(frame[:, :, None], state[0])
        value, s2 = self.block2.step(first, state[1])
        value, s3 = self.block3.step(value, state[2])
        sequence, hidden = self.gru(self._spatial_features(value), state[3])
        features = torch.cat((sequence[:, 0], first[:, :, 0].flatten(1)), -1)
        return features, (s1, s2, s3, hidden)

    def spatial_logits(self, features):
        hidden = self.streaming_hidden_size
        maps = features[..., hidden:].reshape(-1, 32, 20, 26)
        condition = self.localizer_condition(features[..., :hidden].reshape(-1, hidden))
        maps = maps + condition[:, :, None, None]
        return self.localizer(maps)[:, 0].reshape(*features.shape[:-1], 20, 26)

    @staticmethod
    def _expand_marginal(marginal, size):
        # AvgPool3 has spatial stride3 and pixel centers1+3j. Padding before
        # 3x interpolation and cropping preserves that coordinate alignment.
        padded = F.pad(marginal.reshape(-1, 1, marginal.shape[-1]), (1, 1), mode='replicate')
        expanded = F.interpolate(padded, scale_factor=3, mode='linear', align_corners=False)
        return expanded[:, 0, 3:3 + size].reshape(*marginal.shape[:-1], size)

    def _coordinate_logits(self, features, heatmap):
        coarse = features[..., :self.streaming_hidden_size]
        x = self._expand_marginal(torch.logsumexp(heatmap, -2), 80)
        y = self._expand_marginal(torch.logsumexp(heatmap, -1), 60)
        return (self.fc_w(coarse) + self.localizer_scale * x,
                self.fc_h(coarse) + self.localizer_scale * y)

    def coordinate_logits(self, features):
        return self._coordinate_logits(features, self.spatial_logits(features))

    def forward_with_spatial_logits(self, frames):
        features = self.extract_temporal(frames)
        heatmap = self.spatial_logits(features)
        x, y = self._coordinate_logits(features, heatmap)
        return x, y, heatmap

    def initialize_from_streaming(self, state):
        result = self.load_state_dict(state, strict=False)
        permitted = {'localizer_scale', 'localizer_condition.weight', 'localizer_condition.bias',
                     'localizer.0.weight', 'localizer.0.bias', 'localizer.1.weight',
                     'localizer.1.bias', 'localizer.3.weight', 'localizer.3.bias'}
        if set(result.missing_keys) != permitted or result.unexpected_keys:
            raise ValueError('Expected a compatible streaming backbone without a localizer')
        return dict(recurrent_initialization='streaming_backbone_with_zero_localization_residual')
