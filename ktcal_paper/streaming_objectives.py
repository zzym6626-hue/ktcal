"""Training-only soft-coordinate difficulty weights and masked pixel loss."""
import math

import torch
from torch.nn import functional as F

from .stnet_sequence import decode_soft_expectation


def _validate(logits_x, logits_y, targets, valid, temperature):
    if (logits_x.ndim != 3 or logits_y.ndim != 3
            or logits_x.shape[:2] != logits_y.shape[:2]
            or targets.shape != logits_x.shape[:2] + (2,)
            or valid.shape != logits_x.shape[:2]):
        raise ValueError('Expected aligned [B,T] logits, targets and valid mask')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite soft decode temperature required')


@torch.no_grad()
def streaming_frame_weights(logits_x, logits_y, labels, valid, *, temperature=1.,
                            persistent_boost=0., min_run=5,
                            cold_boost=0., cold_frames=10):
    """Weights for real training timesteps; never use this for validation."""
    _validate(logits_x, logits_y, labels, valid, temperature)
    if (any(not math.isfinite(v) or v < 0 for v in (persistent_boost, cold_boost))
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in (min_run, cold_frames))):
        raise ValueError('Finite nonnegative boosts and positive integer frame counts required')
    valid = valid.bool()
    weights = torch.ones_like(valid, dtype=logits_x.dtype)
    if cold_boost:
        weights[:, :cold_frames] += cold_boost
    if persistent_boost and logits_x.shape[1] >= min_run:
        prediction = decode_soft_expectation(logits_x / temperature, logits_y / temperature)
        scale = labels.new_tensor((logits_x.shape[-1], logits_y.shape[-1]))
        bad = (torch.linalg.vector_norm((prediction - labels) * scale, dim=-1) >= 3.) & valid
        starts = F.avg_pool1d(bad[:, None].to(weights.dtype), min_run, stride=1) == 1
        persistent = F.max_pool1d(F.pad(starts.to(weights.dtype), (min_run - 1, min_run - 1)),
                                   min_run, stride=1).squeeze(1).bool()
        weights += persistent_boost * persistent.to(weights.dtype)
    return weights * valid.to(weights.dtype)


def weighted_coordinate_huber(logits_x, logits_y, targets, valid, *,
                              temperature=1., weights=None):
    _validate(logits_x, logits_y, targets, valid, temperature)
    valid = valid.bool()
    prediction = decode_soft_expectation(logits_x / temperature, logits_y / temperature)
    if not bool(valid.any()):
        return prediction.sum() * 0
    if not bool(torch.isfinite(targets[valid]).all()):
        raise ValueError('Valid pixel targets must be finite')
    scale = prediction.new_tensor((logits_x.shape[-1], logits_y.shape[-1]))
    per_frame = F.smooth_l1_loss(prediction[valid] * scale, targets.detach()[valid] * scale,
                                reduction='none').mean(-1)
    if weights is None:
        return per_frame.mean()
    if weights.shape != valid.shape:
        raise ValueError('Training weights must match valid mask')
    selected = weights.detach().to(device=per_frame.device, dtype=per_frame.dtype)[valid]
    if not bool(torch.isfinite(selected).all()) or bool((selected <= 0).any()):
        raise ValueError('Weights must be finite and positive on valid frames')
    return (per_frame * selected).sum() / selected.sum()
