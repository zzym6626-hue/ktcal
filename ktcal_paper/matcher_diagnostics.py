"""Small, reusable measurements for template-matching diagnostics."""

import numpy as np
import torch
from torch.nn import functional as F


def response_map(template, search, metric="cosine"):
    """Return a sliding response map for already extracted feature patches."""
    if metric not in ("cosine", "zero_mean_ncc"):
        raise ValueError(f"unsupported matching metric: {metric}")
    if template.ndim != 4 or search.ndim != 4:
        raise ValueError("template and search must be B x C x H x W tensors")
    if template.shape[:2] != search.shape[:2]:
        raise ValueError("template and search batch/channel dimensions must match")
    height, width = template.shape[-2:]
    if search.shape[-2] < height or search.shape[-1] < width:
        raise ValueError("search patch is smaller than template")

    windows = F.unfold(search, (height, width))
    reference = template.flatten(1).unsqueeze(-1)
    if metric == "zero_mean_ncc":
        reference = reference - reference.mean(dim=1, keepdim=True)
        windows = windows - windows.mean(dim=1, keepdim=True)
    numerator = (windows * reference).sum(dim=1)
    denominator = (
        windows.square().sum(dim=1) * reference.square().sum(dim=1)
    ).clamp_min(1e-12).sqrt()
    return (numerator / denominator).reshape(
        search.shape[0], search.shape[-2] - height + 1,
        search.shape[-1] - width + 1)


def summarize_matches(errors, target_offsets, border, confidence, psr,
                      capture_half_span_px=(2.5, 2.5)):
    """Separate matcher error from failures caused by the finite search range."""
    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    offsets = np.asarray(target_offsets, dtype=np.float64)
    border = np.asarray(border, dtype=bool).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    psr = np.asarray(psr, dtype=np.float64).reshape(-1)
    if offsets.shape != (len(errors), 2):
        raise ValueError("target_offsets must have shape [frames, 2]")
    if any(len(values) != len(errors) for values in (border, confidence, psr)):
        raise ValueError("all diagnostic arrays must have the same length")
    if not len(errors):
        raise ValueError("cannot summarize an empty diagnostic case")

    capture = np.asarray(capture_half_span_px, dtype=np.float64)
    if capture.shape == ():
        capture = np.repeat(capture, 2)
    if capture.shape != (2,) or not np.isfinite(capture).all() or (capture < 0).any():
        raise ValueError("capture_half_span_px must be a non-negative x/y pair")
    inside = (np.abs(offsets) <= capture[None] + 1e-9).all(axis=1)

    def mean(mask):
        return float(errors[mask].mean()) if mask.any() else None

    return {
        "frames": int(len(errors)),
        "mean_error_px": float(errors.mean()),
        "median_error_px": float(np.median(errors)),
        "p1": float(np.mean(errors < 1.0)),
        "p3": float(np.mean(errors < 3.0)),
        "p5": float(np.mean(errors < 5.0)),
        "geometry_capture_rate": float(inside.mean()),
        "mean_error_inside_capture_px": mean(inside),
        "mean_error_outside_capture_px": mean(~inside),
        "border_peak_rate": float(border.mean()),
        "low_confidence_rate": float((confidence < 0.5).mean()),
        "confidence_mean": float(confidence.mean()),
        "psr_median": float(np.median(psr)),
    }
