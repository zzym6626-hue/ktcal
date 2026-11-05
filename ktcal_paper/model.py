"""Last-window coordinate regression; BiGRU training and forward-only deployment.

Unspecified paper choices are intentionally explicit: linear coordinate heads,
16 NCC feature channels (elsewhere), and independent forward-head supervision.
Spatial adaptive pooling follows Fig.2's 128x4x4 features; time is never pooled.
"""
import copy
import torch
from torch import nn


class ITDStage(nn.Module):
    def __init__(self, in_channels, channels, kernel, pool, variant):
        super().__init__()
        temporal = 3 if variant == 'full_conv' else 1
        layers = [nn.Conv3d(in_channels, channels, (temporal, kernel, kernel),
                            padding=(temporal//2, kernel//2, kernel//2)),
                  nn.BatchNorm3d(channels), nn.ReLU()]
        if variant not in ('no_temporal', 'full_conv'):
            layers.extend([nn.Conv3d(channels, channels, (3, 1, 1),
                                     padding=(1, 0, 0), bias=False),
                           nn.BatchNorm3d(channels), nn.ReLU()])
        if pool:
            layers.append(nn.AvgPool3d((1, 3, 3)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PaperSTNet(nn.Module):
    def __init__(self, deployed=False, variant='full', readout='last'):
        super().__init__()
        if variant not in ('full', 'no_temporal', 'no_gru', 'full_conv'):
            raise ValueError(f'Unknown architecture variant: {variant}')
        if readout not in ('last','final_hidden'):
            raise ValueError(f'Unknown recurrent readout: {readout}')
        self.deployed, self.variant = deployed, variant
        self.readout = readout
        self.itd = nn.Sequential(
            ITDStage(2, 32, 7, True, variant),
            ITDStage(32, 64, 5, True, variant),
            ITDStage(64, 128, 5, False, variant))
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.norm = nn.LayerNorm(2048)
        self.dropout = nn.Dropout(.5)
        if variant == 'no_gru':
            self.gru = None
            self.head = nn.Linear(2048, 2)
        else:
            self.gru = nn.GRU(2048, 128, batch_first=True,
                              bidirectional=not deployed)
            self.head = nn.Linear(128 if deployed else 256, 2)
        nn.init.constant_(self.head.bias, .5)
        self.forward_head = (nn.Linear(128, 2)
                             if not deployed and self.gru is not None else None)
        if self.forward_head is not None:
            nn.init.constant_(self.forward_head.bias, .5)

    def forward(self, frames, return_forward=False):
        if frames.ndim != 5 or frames.shape[2:] != (2, 60, 80) or frames.shape[1] < 1:
            raise ValueError('Expected nonempty (B,W,2,60,80) windows')
        b, w = frames.shape[:2]
        x = self.itd(frames.permute(0, 2, 1, 3, 4))
        x = x.permute(0, 2, 1, 3, 4).reshape(b*w, 128, *x.shape[-2:])
        x = self.spatial_pool(x).reshape(b, w, 2048)
        x = self.dropout(self.norm(x))
        if self.gru is not None:
            x, hidden = self.gru(x)
        last = x[:, -1]
        if self.gru is not None and not self.deployed and self.readout == 'final_hidden':
            last = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        coordinate = self.head(last)
        if return_forward:
            forward = self.forward_head(last[:, :128]) if self.forward_head is not None else coordinate
            return coordinate, forward
        return coordinate

    def last_step(self, frames):
        return self(frames)

    def export_forward(self):
        """Copy the learned forward branch exactly; never double its hidden size."""
        if self.deployed:
            return copy.deepcopy(self)
        result = PaperSTNet(deployed=True, variant=self.variant, readout=self.readout)
        reference = next(self.parameters())
        result.to(device=reference.device, dtype=reference.dtype)
        result.itd.load_state_dict(self.itd.state_dict())
        result.norm.load_state_dict(self.norm.state_dict())
        if self.gru is None:
            result.head.load_state_dict(self.head.state_dict())
        else:
            result.gru.load_state_dict({k: v for k, v in self.gru.state_dict().items()
                                       if not k.endswith('_reverse')})
            result.head.load_state_dict(self.forward_head.state_dict())
        result.train(self.training)
        return result
