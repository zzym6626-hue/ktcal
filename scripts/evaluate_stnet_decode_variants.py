#!/usr/bin/env python3
"""Evaluate STNet decode and multi-view inference variants on flat 3ET HDF5.

All variants use the same frozen checkpoint; only inference-time decoding and
temporal view aggregation change.  Validation-only usage is the default; the
final holdout must only be evaluated once after all choices are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch


def summarize(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    return {
        'frames': int(errors.size),
        'mean_error_px': float(errors.mean()),
        'median_error_px': float(np.median(errors)),
        'p1': float(np.mean(errors <= 1)),
        'p3': float(np.mean(errors <= 3)),
        'p5': float(np.mean(errors <= 5)),
        'p10': float(np.mean(errors <= 10)),
    }


def _axis_decode(logits: torch.Tensor, mode: str) -> torch.Tensor:
    """Decode one axis with shape [..., bins] to bin coordinates."""
    if mode == 'hard':
        return logits.argmax(dim=-1).to(logits.dtype)
    if mode == 'soft':
        probability = torch.softmax(logits, dim=-1)
        grid = torch.arange(
            logits.shape[-1], device=logits.device, dtype=logits.dtype)
        return (probability * grid).sum(dim=-1)
    if mode == 'parabolic':
        probability = torch.softmax(logits, dim=-1)
        bins = probability.shape[-1]
        flat = probability.reshape(-1, bins)
        index = flat.argmax(dim=-1)
        delta = torch.zeros_like(index, dtype=flat.dtype)
        inner = (index > 0) & (index < bins - 1)
        if inner.any():
            positions = torch.nonzero(inner).squeeze(-1)
            centres = index[positions]
            lower = flat[positions, centres - 1]
            centre = flat[positions, centres]
            upper = flat[positions, centres + 1]
            denominator = lower - 2 * centre + upper
            safe = torch.where(
                denominator.abs() > 1e-8, denominator,
                torch.ones_like(denominator))
            delta[positions] = (0.5 * (lower - upper) / safe).clamp(-0.5, 0.5)
        return (index.to(flat.dtype) + delta).reshape(probability.shape[:-1])
    raise ValueError(f'unsupported decode mode: {mode}')


def decode_pair(logits_x: torch.Tensor, logits_y: torch.Tensor,
                mode: str) -> torch.Tensor:
    x = _axis_decode(logits_x, mode) / logits_x.shape[-1]
    y = _axis_decode(logits_y, mode) / logits_y.shape[-1]
    return torch.stack((x, y), dim=-1)


def _window_starts(length: int, window: int, stride: int) -> list:
    if window < 1 or stride < 1:
        raise ValueError('window and stride must be positive')
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def view_prediction(model, frames: torch.Tensor, *, window: int, stride: int,
                    decode_mode: str, reverse: bool) -> torch.Tensor:
    """Aggregate sliding-window predictions of one temporal view."""
    length = frames.shape[0]
    source = torch.flip(frames, dims=(0,)) if reverse else frames
    sums = torch.zeros(length, 2, device=frames.device)
    counts = torch.zeros(length, device=frames.device)
    for start in _window_starts(length, window, stride):
        stop = start + window
        chunk = source[start:stop].unsqueeze(0)
        logits_x, logits_y = model(chunk)
        prediction = decode_pair(logits_x[0], logits_y[0], decode_mode)
        if reverse:
            prediction = torch.flip(prediction, dims=(0,))
            target_start, target_stop = length - stop, length - start
        else:
            target_start, target_stop = start, stop
        sums[target_start:target_stop] += prediction
        counts[target_start:target_stop] += 1
    return sums / counts.unsqueeze(-1)


def build_model(checkpoint):
    from models.STNet import STNet
    config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    model_args = SimpleNamespace(
        stnet_block3_kernel=int(config.get('block3_kernel', 5)),
        stnet_unidirectional=bool(config.get('unidirectional', False)),
        stnet_decode='argmax',
        stnet_readout=config.get('readout', 'last'),
        stnet_temporal_spatial_kernel=int(
            config.get('temporal_spatial_kernel', 1)),
    )
    model = STNet(model_args)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model, config


def main():
    parser = argparse.ArgumentParser('STNet decode-variant evaluation')
    parser.add_argument('--h5', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--stride', type=int, default=50)
    parser.add_argument('--limit', type=int, default=0,
                        help='Evaluate only the first N chunks (0 = all)')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    h5_path = Path(args.h5).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    model, config = build_model(checkpoint)
    model.to(device).eval()

    variants = [
        ('hard_forward', 'hard', False),
        ('soft_forward', 'soft', False),
        ('parabolic_forward', 'parabolic', False),
        ('soft_bidir_avg', 'soft', True),
        ('parabolic_bidir_avg', 'parabolic', True),
        ('hard_bidir_avg', 'hard', True),
    ]
    errors = {name: [] for name, _, _ in variants}
    scale = torch.tensor([80.0, 60.0])
    with h5py.File(h5_path, 'r') as handle, torch.inference_mode():
        label_key = 'label' if 'label' in handle else 'labels'
        count = len(handle['frames'])
        selected = count if args.limit <= 0 else min(count, args.limit)
        for sequence in range(selected):
            frames = torch.from_numpy(
                handle['frames'][sequence].astype(np.float32))
            divisor = float(config.get('input_divisor', 1.0))
            if divisor != 1.0:
                frames = frames / divisor
            labels = torch.from_numpy(
                handle[label_key][sequence, :, :2].astype(np.float32))
            frames = frames.to(device)
            cache = {}

            def get_view(decode_mode, reverse, cache=cache, frames=frames):
                key = (decode_mode, reverse)
                if key not in cache:
                    cache[key] = view_prediction(
                        model, frames, window=args.window, stride=args.stride,
                        decode_mode=decode_mode, reverse=reverse)
                return cache[key]

            for name, decode_mode, use_reverse in variants:
                prediction = get_view(decode_mode, False)
                if use_reverse:
                    prediction = 0.5 * (
                        prediction + get_view(decode_mode, True))
                error = torch.linalg.vector_norm(
                    (prediction.cpu() - labels) * scale, dim=-1)
                errors[name].append(error.numpy())

    report = {
        'schema': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'h5': str(h5_path),
        'h5_sha256': hashlib.sha256(h5_path.read_bytes()).hexdigest(),
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': hashlib.sha256(
            checkpoint_path.read_bytes()).hexdigest(),
        'window': args.window,
        'stride': args.stride,
        'chunks_evaluated': selected,
        'metrics': {name: summarize(np.concatenate(values))
                    for name, values in errors.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
