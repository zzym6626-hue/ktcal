"""
NCC template matching branch (KTCal paper, Sec 3.2.2 / review reply W4).

Pipeline:
  1. A lightweight feature extractor encodes event frames.
  2. A template (pupil neighborhood, cropped around the previous position) is
     correlated against a local search region centered on the Kalman prediction.
  3. Normalized cross-correlation (NCC) produces a response map. The peak gives
     the matched pupil coordinate and a confidence score.

Differentiability (W4):
  - Training : Soft-Argmax with temperature T=0.1 -> differentiable, gradients
               flow back to the feature extractor via L_match.
  - Inference: Hard Argmax -> minimal latency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """Lightweight shared encoder for template / search patches (P0: kept small so
    NCC stays far cheaper than STNet, which is the premise of the closed loop)."""

    def __init__(self, in_ch=2, feat_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def soft_argmax_2d(response, temperature=0.1):
    """
    response: (B, H, W) correlation map.
    Returns normalized (x, y) in [0, 1] via differentiable soft-argmax,
    plus the peak confidence (softmax mass at the arg-max cell).
    """
    b, h, w = response.shape
    flat = response.reshape(b, -1)
    soft = F.softmax(flat / temperature, dim=-1)
    soft_map = soft.reshape(b, h, w)

    grid_y = torch.arange(h, device=response.device, dtype=response.dtype)
    grid_x = torch.arange(w, device=response.device, dtype=response.dtype)
    py = soft_map.sum(dim=2)  # (B, H)
    px = soft_map.sum(dim=1)  # (B, W)
    cy = (py * grid_y).sum(-1) / max(h - 1, 1)
    cx = (px * grid_x).sum(-1) / max(w - 1, 1)
    confidence = soft.max(dim=-1).values
    return torch.stack([cx, cy], dim=-1), confidence


def hard_argmax_2d(response):
    """response: (B, H, W). Returns normalized (x, y) in [0,1] and peak value."""
    b, h, w = response.shape
    flat = response.reshape(b, -1)
    peak_val, idx = flat.max(dim=-1)
    iy = (idx // w).to(response.dtype) / max(h - 1, 1)
    ix = (idx % w).to(response.dtype) / max(w - 1, 1)
    return torch.stack([ix, iy], dim=-1), peak_val


class NCCMatcher(nn.Module):
    """
    Normalized cross-correlation matcher.

    forward(template, search) computes a per-sample NCC response map between a
    template patch and a search patch, then localizes the peak.
    """

    def __init__(self, in_ch=2, feat_ch=16, temperature=0.1):
        super().__init__()
        self.extractor = FeatureExtractor(in_ch, feat_ch)
        self.temperature = temperature

    def ncc_response(self, template, search):
        """
        template: (B, C_in, th, tw)
        search  : (B, C_in, sh, sw)
        Returns a TRUE normalized cross-correlation map (B, oh, ow), values in
        [-1, 1], with oh=sh-th+1, ow=sw-tw+1.

        Each sliding window is normalized by ITS OWN local statistics (not by a
        single global norm over the whole search region), so the peak reaches ~1
        for a good match. Per window:
            NCC = <S_win - mean(S_win), T_hat> / |S_win - mean(S_win)|
        where T_hat is the zero-mean unit-norm template. Since T_hat is
        zero-mean, <S_win, T_hat> == <S_win - mean(S_win), T_hat>, so the
        numerator is a plain cross-correlation with the normalized template.
        """
        tf = self.extractor(template)
        sf = self.extractor(search)
        b, c, th, tw = tf.shape
        n = c * th * tw
        eps = 1e-6
        responses = []
        for i in range(b):
            t = tf[i:i + 1]
            t = t - t.mean()
            t = t / t.norm().clamp_min(eps)      # zero-mean, unit-norm template
            s = sf[i:i + 1]
            ones = torch.ones_like(t)
            num = F.conv2d(s, t)                  # cross-correlation numerator
            sum_win = F.conv2d(s, ones)           # sum of S over each window
            sumsq_win = F.conv2d(s * s, ones)     # sum of S^2 over each window
            var = (sumsq_win - sum_win * sum_win / n).clamp_min(eps)
            ncc = (num / var.sqrt())[0, 0]        # (oh, ow) in [-1, 1]
            responses.append(ncc)
        return torch.stack(responses, dim=0)

    def forward(self, template, search, training=None):
        if training is None:
            training = self.training
        response = self.ncc_response(template, search)  # (B, oh, ow)
        if training:
            coord, conf = soft_argmax_2d(response, self.temperature)
        else:
            coord, conf = hard_argmax_2d(response)
        return coord, conf, response


if __name__ == '__main__':
    matcher = NCCMatcher()
    tmpl = torch.rand(4, 2, 11, 11)
    srch = torch.rand(4, 2, 21, 21)
    matcher.train()
    c, conf, r = matcher(tmpl, srch)
    print('train coord', c.shape, 'conf', conf.shape, 'resp', r.shape)
    matcher.eval()
    c, conf, r = matcher(tmpl, srch)
    print('eval  coord', c.shape, 'conf', conf.shape)
