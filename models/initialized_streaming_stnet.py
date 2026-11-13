"""Seed causal GRU history from the first received current-readout latent."""
import torch
from torch import nn

from .current_readout_streaming_stnet import CurrentReadoutStreamingSTNet


class InitializedCurrentReadoutStreamingSTNet(CurrentReadoutStreamingSTNet):
    def __init__(self, args=None, hidden_size=256):
        super().__init__(args, hidden_size=hidden_size)
        self.initial_hidden_projection = nn.Linear(256, hidden_size)
        self.initial_hidden_scale = nn.Parameter(torch.zeros(()))

    def _current_latent(self, current):
        return self.current_readout[2](self.current_readout[1](self.current_readout[0](current)))

    def _initial_hidden(self, latent):
        return self.initial_hidden_scale * self.initial_hidden_projection(latent).tanh()

    def extract_temporal(self, frames):
        if frames.ndim != 5 or frames.shape[2:] != (2, 60, 80) or frames.shape[1] < 1:
            raise ValueError('Expected nonempty B,T,2,60,80 frames')
        spatial = self.block3(self.block2(self.block1(frames.permute(0, 2, 1, 3, 4))))
        current = self._spatial_features(spatial)
        latent = self._current_latent(current)
        hidden = self._initial_hidden(latent[:, 0])[None]
        sequence = self.gru(current, hidden)[0]
        return torch.cat((sequence, latent), -1)

    def initial_state(self, frame):
        return super().initial_state(frame) + (frame.new_zeros(1, len(frame), 1),)

    def stream_step(self, frame, state=None):
        if self.training:
            raise ValueError('Stateful streaming requires eval mode')
        if frame.ndim != 4 or frame.shape[1:] != (2, 60, 80):
            raise ValueError('Expected B,2,60,80 current frames')
        if state is None:
            state = self.initial_state(frame)
        if len(state) != 5:
            raise ValueError('Expected three temporal caches, GRU state and initialized marker')
        value, s1 = self.block1.step(frame[:, :, None], state[0])
        value, s2 = self.block2.step(value, state[1])
        value, s3 = self.block3.step(value, state[2])
        current = self._spatial_features(value)
        latent = self._current_latent(current)
        hidden = state[3] + self._initial_hidden(latent[:, 0])[None] * (1 - state[4])
        sequence, hidden = self.gru(current, hidden)
        return torch.cat((sequence[:, 0], latent[:, 0]), -1), (
            s1, s2, s3, hidden, torch.ones_like(state[4]))

    def current_logits(self, features):
        latent = features[..., self.streaming_hidden_size:]
        logits = self.current_readout[4](self.current_readout[3](latent))
        return logits[..., :80], logits[..., 80:]

    def initialize_from_current_readout(self, state):
        result = self.load_state_dict(state, strict=False)
        permitted = {'initial_hidden_scale', 'initial_hidden_projection.weight',
                     'initial_hidden_projection.bias'}
        if set(result.missing_keys) != permitted or result.unexpected_keys:
            raise ValueError('Expected compatible current-readout checkpoint without learned initial state')
        return dict(recurrent_initialization='current_readout_backbone_with_zero_initial_hidden_scale')
