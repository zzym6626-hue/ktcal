"""Rectangular feature-space cosine matching, implementing paper equations 3–4."""
import torch
from torch import nn
from torch.nn import functional as F

TEMPLATE = (30, 25)  # width, height throughout this package
SEARCH = (35, 30)


def crop_rectangle(images, centers, size):
    width, height = size
    if min(size) <= 0:
        raise ValueError('Patch sizes must be positive')
    h, w = images.shape[-2:]
    yy = torch.arange(height, device=images.device, dtype=images.dtype) - (height-1)/2
    xx = torch.arange(width, device=images.device, dtype=images.dtype) - (width-1)/2
    gy, gx = torch.meshgrid(yy, xx, indexing='ij')
    gx = (gx[None] + centers[:, 0, None, None]) / (w-1)*2-1
    gy = (gy[None] + centers[:, 1, None, None]) / (h-1)*2-1
    return F.grid_sample(images, torch.stack([gx, gy], -1),
                         align_corners=True, padding_mode='zeros')


def response_to_pixel(local, centers, template=TEMPLATE, search=SEARCH):
    span = local.new_tensor([search[0]-template[0], search[1]-template[1]])
    return centers + (local-.5)*span


class PaperMatcher(nn.Module):
    def __init__(self, channels=16, temperature=.1):
        super().__init__()
        self.channels, self.temperature = channels, temperature
        self.extractor = nn.Sequential(
            nn.Conv2d(2, 2, 3, padding=1, groups=2, bias=False),
            nn.Conv2d(2, channels, 1, bias=False), nn.ReLU())

    def ncc_response(self, template, search):
        t, s = self.extractor(template), self.extractor(search)
        th, tw = t.shape[-2:]
        if s.shape[-2] < th or s.shape[-1] < tw:
            raise ValueError('Search patch is smaller than template')
        windows = F.unfold(s, (th, tw))
        t = t.flatten(1).unsqueeze(-1)
        numerator = (windows*t).sum(1)
        norm = (windows.square().sum(1)*t.square().sum(1)).clamp_min(1e-12).sqrt()
        return (numerator/norm).reshape(s.shape[0], s.shape[-2]-th+1, s.shape[-1]-tw+1)

    def locate(self, response, soft=False):
        b, h, w = response.shape
        flat = response.flatten(1)
        yy, xx = torch.meshgrid(torch.arange(h, device=response.device),
                                 torch.arange(w, device=response.device), indexing='ij')
        coordinates = torch.stack([xx.flatten()/max(w-1,1), yy.flatten()/max(h-1,1)], -1).to(response.dtype)
        if soft:
            local = torch.softmax(flat/self.temperature, -1) @ coordinates
        else:
            local = coordinates[flat.argmax(-1)]
        return local, flat.max(-1).values

    def forward(self, template, search, soft=False):
        response = self.ncc_response(template, search)
        local, confidence = self.locate(response, soft)
        return local, confidence, response

    def training_loss(self, frames, labels):
        b, length = frames.shape[:2]
        scale = frames.new_tensor([80.,60.])
        rows = torch.arange(b, device=frames.device)
        gap = torch.randint(1, min(8,length-1)+1, (b,), device=frames.device) if length>1 else torch.zeros(b, device=frames.device,dtype=torch.long)
        previous = length-1-gap
        if labels.shape[-1] >= 4:
            valid = (labels[rows,previous,3]>0)&(labels[:,-1,3]>0)
            if not valid.any():
                return sum(p.sum()*0 for p in self.parameters())
            rows=rows[valid]; previous=previous[valid]
            frames=frames[valid]; labels=labels[valid]
            b=len(frames); rows=torch.arange(b,device=frames.device)
        target = labels[:,-1,:2]*scale
        template = crop_rectangle(frames[rows,previous],labels[rows,previous,:2]*scale,TEMPLATE)
        jitter = (torch.rand(b,2,device=frames.device)-.5)*frames.new_tensor([5.,5.])
        center = target+jitter
        search = crop_rectangle(frames[:,-1],center,SEARCH)
        local, _, _ = self(template,search,soft=True)
        return F.mse_loss(response_to_pixel(local,center),target)
