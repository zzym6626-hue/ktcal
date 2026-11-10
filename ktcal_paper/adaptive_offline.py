"""Label-free, offline temporal correction and validation-only diagnostics."""

import numpy as np

from .combined_3et import SCALE, moving_average_sequence_predictions


def motion_speed(guidance):
    """Estimate px/frame from a three-frame-smoothed model trajectory."""
    values = np.asarray(guidance, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 1:
        raise ValueError('guidance must have shape [T,2] with T >= 1')
    if not np.isfinite(values).all():
        raise ValueError('guidance contains non-finite predictions')
    smooth = moving_average_sequence_predictions([values], 3)[0]
    if len(values) == 1:
        return np.zeros(1, dtype=np.float32)
    return np.linalg.norm(np.gradient(smooth * SCALE, axis=0), axis=1)


def adaptive_smooth(predictions, guidance, slow=.5, fast=1.5):
    """Blend MA11 at slow steps toward MA3 on fast steps, without labels."""
    values = np.asarray(predictions, dtype=np.float32)
    guide = np.asarray(guidance, dtype=np.float32)
    if values.shape != guide.shape or values.ndim != 2 or values.shape[1] != 2:
        raise ValueError('predictions and guidance must share shape [T,2]')
    if not np.isfinite(values).all() or not 0 <= slow < fast:
        raise ValueError('predictions must be finite and 0 <= slow < fast')
    speed = motion_speed(guide)
    short = moving_average_sequence_predictions([values], 3)[0]
    long = moving_average_sequence_predictions([values], 11)[0]
    weight = np.clip((speed - slow) / (fast - slow), 0., 1.)[:, None]
    return ((1. - weight) * long + weight * short).astype(np.float32), speed


def select_reliable_match(stnet, matched, confidence, *,
                          min_confidence=.7, max_disagreement_px=2.):
    """Keep STNet when a low-confidence matched output disagrees materially."""
    raw = np.asarray(stnet, dtype=np.float32)
    corrected = np.asarray(matched, dtype=np.float32)
    certainty = np.asarray(confidence, dtype=np.float32)
    if (raw.shape != corrected.shape or raw.ndim != 2 or raw.shape[1] != 2
            or certainty.shape != (len(raw),)):
        raise ValueError('stnet/matched must share [T,2], confidence [T]')
    if (not np.isfinite(raw).all() or not np.isfinite(corrected).all()
            or not 0 <= min_confidence <= 1 or max_disagreement_px <= 0):
        raise ValueError('invalid predictions or gate thresholds')
    disagreement = np.linalg.norm((corrected - raw) * SCALE, axis=1)
    fallback = (certainty < min_confidence) & (
        disagreement > max_disagreement_px)
    selected = np.where(fallback[:, None], raw, corrected)
    return selected, fallback


def diagnose_frame_groups(guidance, predicted, labels, event_counts):
    """Use predictions/events for strata; labels only for aggregate scores."""
    guide = np.asarray(guidance, dtype=np.float32)
    observed = np.asarray(predicted, dtype=np.float32)
    truth = np.asarray(labels, dtype=np.float32)
    events = np.asarray(event_counts)
    if (guide.shape != observed.shape or guide.shape != truth.shape
            or events.shape != (len(guide),)):
        raise ValueError('per-frame guidance, predictions, labels and events must align')
    speed = motion_speed(guide)
    sparse_cutoff = float(np.quantile(events, .25))
    masks = {
        'speed': {'stable': speed < .5,
                  'transition': (speed >= .5) & (speed < 1.5),
                  'moving': speed >= 1.5},
        'event_density': {'sparse': events <= sparse_cutoff,
                          'dense': events > sparse_cutoff},
        'edge': {'first_or_last_5': np.minimum(
            np.arange(len(guide)), len(guide) - 1 - np.arange(len(guide))) < 5},
    }
    errors = np.linalg.norm((observed - truth) * SCALE, axis=1)
    return {group: {name: {'frames': int(mask.sum()),
                           'mean_error_px': float(errors[mask].mean()) if mask.any() else None,
                           'p3': float(np.mean(errors[mask] < 3)) if mask.any() else None}
                    for name, mask in subgroups.items()}
            for group, subgroups in masks.items()}
