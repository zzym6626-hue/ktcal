"""Rectangular feature-space cosine matching, implementing paper equations 3–4."""
import torch
from torch import nn
from torch.nn import functional as F

TEMPLATE = (30, 25)  # width, height throughout this package
SEARCH = (35, 30)

# Coarse proposal pass. It runs on a factor-2 downsample so the effective
# displacement range grows to +-8 px while the paper's fine 30x25/35x30
# geometry still performs the final +-2.5 px refinement.
COARSE_FACTOR = 2
COARSE_TEMPLATE = (16, 12)
COARSE_SEARCH = (24, 20)


def downsample(images, factor=COARSE_FACTOR):
    if factor != 2:
        raise ValueError('Only factor-2 coarse matching is implemented')
    return F.avg_pool2d(images, factor)


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


def parabolic_offsets(response, row, column):
    """Fit a parabola through each peak and its two neighbours.

    The response map of the paper geometry is only 6x6, so a hard argmax snaps
    every match onto a 1 px lattice and feeds that quantisation straight into
    the Kalman update. The vertex of the parabola is a standard sub-cell
    refinement; it never moves further than half a cell and is skipped where
    the peak sits on the border or the fit is not concave.
    """
    batch, height, width = response.shape
    zero = torch.zeros(batch, dtype=response.dtype, device=response.device)

    def axis(peak, lower, upper):
        denominator = lower-2*peak+upper
        concave = denominator < -1e-9
        safe = torch.where(concave, denominator, torch.ones_like(denominator))
        return torch.where(
            concave, .5*(lower-upper)/safe, torch.zeros_like(peak)).clamp(-.5, .5)

    rows = torch.arange(batch, device=response.device)
    inner_row = (row > 0) & (row < height-1)
    inner_column = (column > 0) & (column < width-1)
    peak = response[rows, row, column]
    dy = torch.where(inner_row, axis(
        peak, response[rows, (row-1).clamp_min(0), column],
        response[rows, (row+1).clamp_max(height-1), column]), zero)
    dx = torch.where(inner_column, axis(
        peak, response[rows, row, (column-1).clamp_min(0)],
        response[rows, row, (column+1).clamp_max(width-1)]), zero)
    return dy, dx


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

    def locate(self, response, soft=False, subpixel=False):
        b, h, w = response.shape
        flat = response.flatten(1)
        yy, xx = torch.meshgrid(torch.arange(h, device=response.device),
                                 torch.arange(w, device=response.device), indexing='ij')
        coordinates = torch.stack([xx.flatten()/max(w-1,1), yy.flatten()/max(h-1,1)], -1).to(response.dtype)
        if soft:
            local = torch.softmax(flat/self.temperature, -1) @ coordinates
        elif subpixel:
            _, index = flat.max(-1)
            row, column = index // w, index % w
            dy, dx = parabolic_offsets(response, row, column)
            local = torch.stack([(column+dx)/max(w-1,1), (row+dy)/max(h-1,1)], -1)
        else:
            local = coordinates[flat.argmax(-1)]
        return local, flat.max(-1).values

    def locate_diagnostics(self, response, soft=False, subpixel=False):
        """Expose capture-range and peak-significance signals for gating.

        A peak on the response border means the true displacement may lie
        outside the +-2.5 px search range, so the match cannot be trusted as a
        Kalman observation. PSR is the peak-to-sidelobe ratio over the other
        response cells.
        """
        local, confidence = self.locate(response, soft, subpixel)
        batch, height, width = response.shape
        flat = response.reshape(batch, -1)
        peak, index = flat.max(-1)
        row, column = index // width, index % width
        mask = F.one_hot(index, flat.shape[-1]).bool()
        others = flat.masked_fill(mask, 0.)
        count = max(1, flat.shape[-1] - 1)
        mean = others.sum(-1) / count
        variance = ((others - mean[:, None]).pow(2).masked_fill(mask, 0.)).sum(-1) / count
        psr = (peak - mean) / variance.clamp_min(1e-12).sqrt()
        border = ((column == 0) | (column == width - 1)
                  | (row == 0) | (row == height - 1))
        return local, confidence, {'border_peak': border, 'psr': psr,
                                   'row': row, 'column': column}

    def match_with_diagnostics(self, template, search, soft=False, subpixel=False):
        response = self.ncc_response(template, search)
        local, confidence, details = self.locate_diagnostics(response, soft, subpixel)
        return local, confidence, response, details

    def forward(self, template, search, soft=False, subpixel=False):
        response = self.ncc_response(template, search)
        local, confidence = self.locate(response, soft, subpixel)
        return local, confidence, response

    def predicted_center_loss(self, previous, current, template_centers,
                              search_centers, targets):
        """Train on the same predicted-center crops used by deployed KTCal.

        A target outside the fine search's axis-aligned capture range cannot be
        represented by the response map, so it is excluded instead of pulling
        the soft peak toward a border cell.
        """
        half_span = targets.new_tensor([
            (SEARCH[0]-TEMPLATE[0])/2,
            (SEARCH[1]-TEMPLATE[1])/2,
        ])
        captured = ((targets-search_centers).abs() <= half_span).all(-1)
        count = int(captured.sum())
        if not count:
            return sum(parameter.sum()*0 for parameter in self.parameters()), 0
        previous = previous[captured]
        current = current[captured]
        template_centers = template_centers[captured]
        search_centers = search_centers[captured]
        targets = targets[captured]
        template = crop_rectangle(previous, template_centers, TEMPLATE)
        search = crop_rectangle(current, search_centers, SEARCH)
        local, _, _ = self(template, search, soft=True)
        predicted = response_to_pixel(local, search_centers)
        return F.mse_loss(predicted, targets), count

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
