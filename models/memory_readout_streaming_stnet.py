"""Current spatial queries attend only to received GRU states."""
import torch
from torch import nn
from torch.nn import functional as F

from .current_readout_streaming_stnet import CurrentReadoutStreamingSTNet


class MemoryReadoutStreamingSTNet(CurrentReadoutStreamingSTNet):
    def __init__(self, args=None, hidden_size=256, memory_window=8):
        if isinstance(memory_window, bool) or not isinstance(memory_window, int) or not 2 <= memory_window <= 32:
            raise ValueError('Integer memory window in[2,32] required')
        super().__init__(args, hidden_size=hidden_size)
        self.memory_window = memory_window
        self.memory_query = nn.Linear(2048, 128)
        self.memory_key = nn.Linear(hidden_size, 128)
        self.memory_value = nn.Linear(hidden_size, 128)
        self.memory_relative_bias = nn.Parameter(torch.zeros(4, memory_window))
        self.memory_head = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(.2),
            nn.Linear(128, 140),
        )
        self.memory_scale = nn.Parameter(torch.zeros(()))

    def initial_state(self, frame):
        return super().initial_state(frame) + (
            frame.new_zeros(len(frame), self.memory_window, self.streaming_hidden_size + 1),)

    def extract_temporal(self, frames):
        features = super().extract_temporal(frames)
        hidden = features[..., :self.streaming_hidden_size]
        padded = F.pad(hidden, (0, 0, self.memory_window - 1, 0))
        history = padded.unfold(1, self.memory_window, 1).transpose(-1, -2)
        valid = F.pad(hidden.new_ones(hidden.shape[:2]), (self.memory_window - 1, 0))
        valid = valid.unfold(1, self.memory_window, 1)
        memory = torch.cat((history, valid[..., None]), -1)
        return torch.cat((features, memory.flatten(-2)), -1)

    def stream_step(self, frame, state=None):
        if state is None:
            state = self.initial_state(frame)
        if len(state) != 5:
            raise ValueError('Expected three temporal caches, GRU state and received history')
        features, next_state = super().stream_step(frame, state[:4])
        row = torch.cat((features[:, :self.streaming_hidden_size], features.new_ones(len(frame), 1)), -1)
        history = torch.cat((state[4][:, 1:], row[:, None]), 1)
        return torch.cat((features, history.flatten(1)), -1), next_state + (history,)

    def current_logits(self, features):
        start = self.streaming_hidden_size
        logits = self.current_readout(features[..., start:start + 2048])
        return logits[..., :80], logits[..., 80:]

    def memory_logits(self, features):
        start = self.streaming_hidden_size
        current = features[..., start:start + 2048]
        history = features[..., start + 2048:].reshape(
            *features.shape[:-1], self.memory_window, start + 1)
        hidden, valid = history[..., :-1], history[..., -1] > 0
        query = self.memory_query(current)
        q = query.reshape(*query.shape[:-1], 4, 32)
        k = self.memory_key(hidden).reshape(*hidden.shape[:-1], 4, 32)
        v = self.memory_value(hidden).reshape(*hidden.shape[:-1], 4, 32)
        scores = torch.einsum('...hd,...khd->...hk', q, k) / (32 ** .5)
        scores = (scores + self.memory_relative_bias).masked_fill(~valid[..., None, :], -torch.inf)
        context = torch.einsum('...hk,...khd->...hd', scores.softmax(-1), v).flatten(-2)
        logits = self.memory_head(torch.cat((query, context), -1))
        return logits[..., :80], logits[..., 80:]

    def _combined_logits(self, features, current_x, current_y, memory_x, memory_y):
        x, y = super()._coordinate_logits(features, current_x, current_y)
        return x + self.memory_scale * memory_x, y + self.memory_scale * memory_y

    def _coordinate_logits(self, features, current_x, current_y):
        return self._combined_logits(features, current_x, current_y, *self.memory_logits(features))

    def forward_with_memory_logits(self, frames):
        features = self.extract_temporal(frames)
        current_x, current_y = self.current_logits(features)
        memory_x, memory_y = self.memory_logits(features)
        x, y = self._combined_logits(features, current_x, current_y, memory_x, memory_y)
        return x, y, current_x, current_y, memory_x, memory_y

    def initialize_from_current_readout(self, state):
        result = self.load_state_dict(state, strict=False)
        permitted = {key for key in self.state_dict() if key.startswith('memory_')}
        if set(result.missing_keys) != permitted or result.unexpected_keys:
            raise ValueError('Expected a compatible current-readout checkpoint without memory residual')
        return dict(recurrent_initialization='current_readout_with_zero_received_memory_residual')
