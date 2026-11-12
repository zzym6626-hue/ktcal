"""Isolated validation experiment helpers; targets never enter inference."""
import time
import numpy as np
import torch
from .combined_3et import SCALE, moving_average_sequence_predictions
from .evaluation_protocol import soft_expectation_decode


def validate_candidate_metrics(expected, actual, stats):
    """Fail closed on inconsistent evidence, including under ``python -O``."""
    if stats['small_fusion_calls'] != 0:
        raise ValueError('Low-tier fusion is forbidden')
    for key in ('p3', 'p5', 'p10'):
        if (not np.isfinite(expected[key]) or not np.isfinite(actual[key])
                or not 0 <= actual[key] <= 1 or actual[key] != expected[key]):
            raise ValueError('Candidate metric mismatch: ' + key)
    if (not np.isfinite(expected['mean_error_px'])
            or not np.isfinite(actual['mean_error_px'])
            or abs(actual['mean_error_px'] - expected['mean_error_px']) >= 2e-5):
        raise ValueError('Candidate metric mismatch: mean_error_px')
    if 'frames' in expected and actual.get('frames') != expected['frames']:
        raise ValueError('Candidate metric mismatch: frames')


def spatial_view(frames, view):
    dx, dy, flip = view
    if abs(dx) >= frames.shape[-1] or abs(dy) >= frames.shape[-2]:
        raise ValueError('Shift must be smaller than image')
    if dx or dy:
        result = torch.zeros_like(frames)
        height, width = frames.shape[-2:]
        sy, sx = max(0,-dy), max(0,-dx)
        ty, tx = max(0,dy), max(0,dx)
        ny, nx = height-abs(dy), width-abs(dx)
        result[...,ty:ty+ny,tx:tx+nx] = frames[...,sy:sy+ny,sx:sx+nx]
    else:
        result = frames
    return result.flip(-1) if flip else result


def undo_view(predicted_px, view):
    dx, dy, flip = view
    result = np.asarray(predicted_px).copy()
    if flip:
        result[...,0] = 79. - result[...,0]
    result -= np.array([dx,dy], dtype=result.dtype)
    return result


def average_views(cache, indices, weights=None, method="mean"):
    if not indices or min(indices) < 0 or max(indices) >= len(cache):
        raise ValueError('Nonempty valid view indices required')
    selected = np.asarray(cache)[indices]
    if method == "median" and weights is None:
        return np.median(selected,axis=0)
    if method == "trimmed" and weights is None:
        trim = len(selected)//5
        ordered = np.sort(selected,axis=0)
        return ordered[trim:len(selected)-trim].mean(axis=0)
    if method != "mean":
        raise ValueError("Invalid ensemble method/weight combination")
    if weights is None:
        return np.mean(selected, axis=0)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(indices),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Finite nonnegative ensemble weights required')
    return np.average(np.asarray(cache)[indices], axis=0, weights=weights).astype(np.float32)


def smooth_pixels(predictions_px, kind='ma11'):
    if kind == 'none':
        return np.asarray(predictions_px).copy()
    if not kind.startswith('ma'):
        from scipy.ndimage import gaussian_filter1d, median_filter
        from scipy.signal import savgol_filter
        if kind.startswith('gauss'):
            fn=lambda x:gaussian_filter1d(x,float(kind[5:]),axis=0,mode='nearest')
        elif kind.startswith('median'):
            fn=lambda x:median_filter(x,size=(int(kind[6:]),1),mode='nearest')
        elif kind.startswith('savgol'):
            fn=lambda x:savgol_filter(x,int(kind[6:]),2,axis=0,mode='interp')
        elif kind.startswith('rts_q'):
            from scripts.evaluate_stnet_temporal_filters import rts_smooth
            q,r=map(float,kind[5:].split('_r'))
            fn=lambda x:rts_smooth(x,q,r)
        else:
            raise ValueError('Unknown record filter')
        return np.stack([fn(np.asarray(record,dtype=np.float64)) for record in predictions_px])
    # Preserve checkpoint selection roundtrip/record boundaries.
    normalized = np.asarray(predictions_px).astype(np.float32) / SCALE
    return np.stack(moving_average_sequence_predictions(list(normalized), int(kind[2:]))) * SCALE


@torch.inference_mode()
def predict_views(model, frames, views, device='cuda', divisor=1., batch_size=8, temperature=1.):
    if batch_size < 1 or divisor <= 0 or temperature <= 0:
        raise ValueError('Positive inference configuration required')
    outputs = [[] for _ in views]
    for begin in range(0,len(frames),batch_size):
        inputs = frames[begin:begin+batch_size].to(device, dtype=torch.float32) / divisor
        for index, view in enumerate(views):
            lx, ly = model(spatial_view(inputs, view))
            decoded = soft_expectation_decode(lx/temperature, ly/temperature).astype(np.float32)*SCALE
            outputs[index].append(undo_view(decoded, view))
    return np.stack([np.concatenate(values) for values in outputs])


def synchronize(device='cuda'):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
