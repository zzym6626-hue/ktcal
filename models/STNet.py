"""
STNet: Lightweight Spatio-Temporal Network (KTCal paper, Sec 3.2)

A lightweight backbone that replaces TDTracker's heavy FFT + Mamba blocks with:
  - ITD (Implicit Temporal Dynamics): decoupled spatio-temporal convolution
        spatial conv (1, Ks, Ks) + temporal conv (3, 1, 1)
  - ETD (Explicit Temporal Dynamics): Bi-GRU over the temporal window
  - FC pupil-coordinate regression head (SA-SimDR heatmap form)

Reference: paper Table 7 (Full STNet ~2.28M params, ~150M FLOPs) and
review reply W6/W8 (decoupled conv + Bi-GRU are complementary).
"""
import torch
import torch.nn as nn
from itertools import repeat


class SpatialDropout(nn.Module):
    def __init__(self, drop=0.5):
        super().__init__()
        self.drop = drop

    def forward(self, inputs, noise_shape=None):
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim() - 2), inputs.shape[-1])
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        noises = inputs.new().resize_(self.noise_shape)
        if self.drop == 1:
            noises.fill_(0.0)
        else:
            noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
        noises = noises.expand_as(inputs)
        outputs.mul_(noises)
        return outputs


class DecoupledSTBlock(nn.Module):
    """One ITD stage: decoupled spatial (1,Ks,Ks) + temporal (3,1,1) convolution."""

    def __init__(self, in_ch, out_ch, spatial_k, spatial_pad, pool=True, dropout=False):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(1, spatial_k, spatial_k),
                      stride=(1, 1, 1), padding=(0, spatial_pad, spatial_pad)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
        )
        temporal_layers = [
            nn.Conv3d(out_ch, out_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
        ]
        if pool:
            temporal_layers.append(nn.AvgPool3d((1, 3, 3)))
        if dropout:
            temporal_layers.append(nn.Dropout())
        self.temporal = nn.Sequential(*temporal_layers)

    def forward(self, x):
        x = self.spatial(x)
        x = self.temporal(x)
        return x


class STNet(nn.Module):
    """
    Input : x (B, S, 2, 60, 80) event frames
    Output: predict_w (B, S, 80), predict_h (B, S, 60)  -- SA-SimDR heatmaps

    The Bi-GRU produces a per-timestep hidden state (B, S, 256). We regress a
    coordinate heatmap at every timestep so the model stays compatible with the
    existing SA-SimDR decode / metric pipeline. (Review reply W6 notes that a
    single-shot variant can instead take BiGRU_out[:, -1, :]; see last_step().)
    """

    def __init__(self, args=None, hidden_size=128):
        super().__init__()
        self.args = args
        self.hidden_size = hidden_size
        block3_kernel = getattr(args, 'stnet_block3_kernel', 5) if args is not None else 5
        if block3_kernel not in (5, 7):
            raise ValueError('stnet_block3_kernel must be 5 or 7')
        self.unidirectional = bool(getattr(args, 'stnet_unidirectional', False)) \
            if args is not None else False

        # ITD: 3 decoupled spatio-temporal stages (32 -> 64 -> 128)
        self.block1 = DecoupledSTBlock(2, 32, spatial_k=7, spatial_pad=3, pool=True)
        self.block2 = DecoupledSTBlock(32, 64, spatial_k=5, spatial_pad=2, pool=True)
        self.block3 = DecoupledSTBlock(
            64, 128, spatial_k=block3_kernel, spatial_pad=block3_kernel // 2,
            pool=False, dropout=True)

        self.pool = nn.AdaptiveAvgPool3d((128, 4, 4))
        self.spatialdropout = SpatialDropout(0.5)
        feat_dim = 128 * 4 * 4  # 2048
        self.layernorm = nn.LayerNorm(feat_dim)

        # ETD: (bi)directional GRU. When unidirectional, double the hidden size
        # so the FC input dimension stays at 256 for weight compatibility.
        gru_hidden = hidden_size * 2 if self.unidirectional else hidden_size
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=gru_hidden,
                          num_layers=1, batch_first=True,
                          bidirectional=not self.unidirectional)

        # FC pupil-coordinate regression heads (heatmaps over width=80, height=60)
        head_dim = gru_hidden if self.unidirectional else hidden_size * 2
        self.fc_w = nn.Sequential(
            nn.Linear(head_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 80))
        self.fc_h = nn.Sequential(
            nn.Linear(head_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 60))

    def extract_temporal(self, x):
        """Return Bi-GRU output sequence (B, S, 2*hidden)."""
        batch_size, seq_len = x.shape[0], x.shape[1]
        x = x.permute(0, 2, 1, 3, 4)          # (B, C, S, H, W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.permute(0, 2, 1, 3, 4)          # (B, S, C, H, W)
        x = self.pool(x)
        x = self.spatialdropout(x)
        x = x.reshape(batch_size, seq_len, -1)
        x = self.layernorm(x)
        x, _ = self.gru(x)                    # (B, S, 2*hidden)
        return x

    def forward(self, x, y=None):
        feat = self.extract_temporal(x)
        predict_w = self.fc_w(feat)
        predict_h = self.fc_h(feat)
        return predict_w, predict_h

    def last_step(self, x):
        """
        Single-shot regression variant described in review reply W6:
        use only the last bidirectional hidden state for one coordinate.
        Returns normalized (x, y) in [0, 1], shape (B, 2).
        """
        feat = self.extract_temporal(x)          # (B, S, 2*hidden)
        h_final = feat[:, -1, :]                 # (B, 2*hidden)
        w_logits = self.fc_w(h_final)            # (B, 80)
        h_logits = self.fc_h(h_final)            # (B, 60)
        decode_mode = getattr(self.args, 'stnet_decode', 'soft') if self.args is not None else 'soft'
        if decode_mode == 'argmax':
            cx = w_logits.argmax(dim=-1).to(w_logits.dtype) / 80.0
            cy = h_logits.argmax(dim=-1).to(h_logits.dtype) / 60.0
        elif decode_mode == 'soft':
            px = torch.softmax(w_logits, dim=-1)
            py = torch.softmax(h_logits, dim=-1)
            grid_x = torch.arange(80, device=x.device, dtype=px.dtype)
            grid_y = torch.arange(60, device=x.device, dtype=py.dtype)
            cx = (px * grid_x).sum(-1) / 80.0
            cy = (py * grid_y).sum(-1) / 60.0
        else:
            raise ValueError(f'unsupported STNet decode mode: {decode_mode}')
        return torch.stack([cx, cy], dim=-1)


if __name__ == '__main__':
    model = STNet()
    dummy = torch.rand(2, 100, 2, 60, 80)
    pw, ph = model(dummy)
    print('predict_w', pw.shape, 'predict_h', ph.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'params: {n_params / 1e6:.3f} M')
