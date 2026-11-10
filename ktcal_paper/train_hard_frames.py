"""Training-set-only difficulty weights for offline 3ET STNet."""

import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def hard_frame_weights(frames, logits_x, logits_y, labels, valid,
                       persistent_boost=0., sparse_boost=0., min_run=5):
    """Upweight sustained train errors and within-record sparse event frames.

    The labels and predictions passed here must originate from the training
    batch, never from validation or test data. One batch row is one recording
    window, so no run or density quantile crosses a row boundary.
    """
    if (frames.ndim != 5 or logits_x.ndim != 3 or logits_y.ndim != 3
            or frames.shape[:2] != logits_x.shape[:2]
            or logits_x.shape[:2] != logits_y.shape[:2]
            or labels.shape != (*frames.shape[:2], 2)
            or valid.shape != frames.shape[:2]):
        raise ValueError('Expected aligned [B,T] frames, logits, labels and valid')
    if (not math.isfinite(persistent_boost) or persistent_boost < 0
            or not math.isfinite(sparse_boost) or sparse_boost < 0
            or isinstance(min_run, bool) or not isinstance(min_run, int)
            or min_run < 1):
        raise ValueError('Boosts must be finite/nonnegative; min_run positive')
    valid = valid.bool()
    weights = torch.ones_like(valid, dtype=logits_x.dtype)
    if persistent_boost and frames.shape[1] >= min_run:
        scale = labels.new_tensor((logits_x.shape[-1], logits_y.shape[-1]))
        prediction = torch.stack((logits_x.argmax(-1), logits_y.argmax(-1)), -1)
        bad = (torch.linalg.vector_norm(
            prediction.to(labels.dtype) - labels * scale, dim=-1) >= 3) & valid
        run_starts = F.avg_pool1d(
            bad[:, None].to(weights.dtype), min_run, stride=1) == 1
        persistent = F.max_pool1d(
            F.pad(run_starts.to(weights.dtype), (min_run-1, min_run-1)),
            min_run, stride=1).squeeze(1).bool()
        weights += persistent_boost * persistent.to(weights.dtype)
    if sparse_boost:
        counts = torch.count_nonzero(frames, dim=(2, 3, 4)).to(weights.dtype)
        for row in range(len(frames)):
            if bool(valid[row].any()):
                cutoff = torch.quantile(counts[row, valid[row]], .25)
                weights[row] += sparse_boost * (counts[row] <= cutoff).to(weights.dtype)
    return weights * valid.to(weights.dtype)
