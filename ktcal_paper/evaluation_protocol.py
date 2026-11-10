"""Shared STNet coordinate decoding, pixel scoring and postprocessing."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.ndimage import binary_dilation


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def hard_argmax_decode(logits_x, logits_y) -> np.ndarray:
    """Decode SimDR logits to normalized ``(x, y)`` coordinates."""
    x = _as_numpy(logits_x)
    y = _as_numpy(logits_y)
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, bins]")
    if x.shape[:2] != y.shape[:2]:
        raise ValueError("x and y logits must share batch and sequence shapes")
    if x.shape[-1] != 80 or y.shape[-1] != 60:
        raise ValueError("official protocol requires 80 x-bins and 60 y-bins")
    return np.stack(
        (np.argmax(x, axis=-1) / 80.0, np.argmax(y, axis=-1) / 60.0),
        axis=-1,
    )


def soft_expectation_decode(logits_x, logits_y) -> np.ndarray:
    """Decode SimDR logits by expectation of the softmax distribution.

    Validation on 3ET+2025 shows this reduces quantization error versus
    hard argmax; it is an inference-time-only change.
    """
    x = _as_numpy(logits_x)
    y = _as_numpy(logits_y)
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, bins]")
    if x.shape[:2] != y.shape[:2]:
        raise ValueError("x and y logits must share batch and sequence shapes")
    if x.shape[-1] != 80 or y.shape[-1] != 60:
        raise ValueError("official protocol requires 80 x-bins and 60 y-bins")
    shifted_x = x - x.max(axis=-1, keepdims=True)
    shifted_y = y - y.max(axis=-1, keepdims=True)
    weights_x = np.exp(shifted_x)
    weights_y = np.exp(shifted_y)
    probability_x = weights_x / weights_x.sum(axis=-1, keepdims=True)
    probability_y = weights_y / weights_y.sum(axis=-1, keepdims=True)
    bins_x = np.arange(80, dtype=x.dtype)
    bins_y = np.arange(60, dtype=y.dtype)
    return np.stack(
        (probability_x @ bins_x / 80.0, probability_y @ bins_y / 60.0),
        axis=-1,
    )


def summarize_pixel_errors(
    predictions,
    labels,
    *,
    width: int = 80,
    height: int = 60,
    pixel_tolerances: Iterable[int] = (1, 3, 5, 10, 15),
) -> dict:
    """Summarize Euclidean errors for normalized gaze coordinates."""
    predicted = _as_numpy(predictions).astype(np.float64, copy=False)
    expected = _as_numpy(labels).astype(np.float64, copy=False)
    if predicted.shape != expected.shape or predicted.shape[-1] != 2:
        raise ValueError("predictions and labels must have the same [..., 2] shape")
    if predicted.size == 0:
        raise ValueError("at least one prediction is required")

    delta = predicted.reshape(-1, 2) - expected.reshape(-1, 2)
    delta *= np.array([width, height], dtype=np.float64)
    errors = np.linalg.norm(delta, axis=-1)
    result = {
        "frames": int(errors.size),
        "mean_error_px": float(errors.mean()),
        "median_error_px": float(np.median(errors)),
    }
    for tolerance in pixel_tolerances:
        result[f"p{tolerance}"] = float(np.mean(errors < tolerance))
    return result


def trim_record_chunks(
    chunks,
    recording_lengths: Iterable[int],
    *,
    chunk_size: int = 200,
) -> np.ndarray:
    """Remove per-record padding without flattening across boundaries."""
    values = _as_numpy(chunks)
    lengths = [int(length) for length in recording_lengths]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if values.ndim < 2 or values.shape[1] != chunk_size:
        raise ValueError("chunks must have shape [chunks, chunk_size, ...]")
    if any(length < 0 for length in lengths):
        raise ValueError("recording lengths must be non-negative")

    chunks_per_record = [
        (length + chunk_size - 1) // chunk_size for length in lengths
    ]
    expected_chunks = sum(chunks_per_record)
    if values.shape[0] != expected_chunks:
        raise ValueError(
            f"chunk count mismatch: got {values.shape[0]}, expected {expected_chunks}"
        )

    trimmed = []
    offset = 0
    for length, record_chunks in zip(lengths, chunks_per_record):
        record = values[offset : offset + record_chunks]
        offset += record_chunks
        flattened = record.reshape((-1,) + values.shape[2:])
        trimmed.append(flattened[:length])
    if not trimmed:
        return np.empty((0,) + values.shape[2:], dtype=values.dtype)
    return np.concatenate(trimmed, axis=0)


def _replace_from_record_history(values: np.ndarray, mask: np.ndarray) -> None:
    previous_valid = None
    for index, masked in enumerate(mask):
        if not masked:
            previous_valid = index
        elif previous_valid is not None:
            values[index] = values[previous_valid]


def official_postprocess(
    predictions,
    *,
    confidence,
    num_events_y,
    recording_lengths: Iterable[int],
    confidence_threshold: float = 0.5,
    event_threshold: float = 0.09,
    dilation_iterations: int = 2,
) -> np.ndarray:
    """Apply official replacement rules without crossing recording boundaries."""
    output = _as_numpy(predictions).copy()
    confidence_values = _as_numpy(confidence).reshape(-1)
    event_values = _as_numpy(num_events_y).reshape(-1)
    lengths = [int(length) for length in recording_lengths]
    if output.ndim != 2 or output.shape[1] != 2:
        raise ValueError("predictions must have shape [frames, 2]")
    if any(length < 0 for length in lengths) or sum(lengths) != len(output):
        raise ValueError("recording lengths must sum to the prediction count")
    if len(confidence_values) != len(output) or len(event_values) != len(output):
        raise ValueError("confidence and event arrays must match the prediction count")
    if dilation_iterations < 0:
        raise ValueError("dilation_iterations must be non-negative")

    offset = 0
    for length in lengths:
        end = offset + length
        record = output[offset:end]
        confidence_mask = confidence_values[offset:end] < confidence_threshold
        _replace_from_record_history(record, confidence_mask)

        event_mask = event_values[offset:end] < event_threshold
        if dilation_iterations:
            event_mask = binary_dilation(
                event_mask,
                iterations=dilation_iterations,
            )
        _replace_from_record_history(record, event_mask)
        offset = end
    return output
