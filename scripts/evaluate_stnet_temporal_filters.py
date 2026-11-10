#!/usr/bin/env python3
"""Evaluate offline temporal filters on frozen STNet predictions (validation only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter

from ktcal_paper.evaluation_protocol import soft_expectation_decode


def summarize(errors: np.ndarray) -> dict:
    errors = np.asarray(errors, dtype=np.float64)
    return {
        'frames': int(errors.size),
        'mean_error_px': float(errors.mean()),
        'median_error_px': float(np.median(errors)),
        'p1': float(np.mean(errors < 1)),
        'p3': float(np.mean(errors < 3)),
        'p5': float(np.mean(errors < 5)),
        'p10': float(np.mean(errors < 10)),
    }


def build_model(checkpoint):
    from models.STNet import STNet
    config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    args = SimpleNamespace(
        stnet_block3_kernel=int(config.get('block3_kernel', 5)),
        stnet_unidirectional=bool(config.get('unidirectional', False)),
        stnet_decode='argmax',
        stnet_readout=config.get('readout', 'last'),
        stnet_temporal_spatial_kernel=int(
            config.get('temporal_spatial_kernel', 1)),
    )
    model = STNet(args)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model, config


def moving_average(values, size):
    kernel = np.ones(size, dtype=np.float64) / size
    return np.stack([
        np.convolve(np.pad(values[:, axis], (size // 2, size // 2), mode='edge'),
                    kernel, mode='valid')
        for axis in range(values.shape[1])
    ], axis=1)


def rts_smooth(values, process_noise, measurement_noise):
    """Constant-velocity RTS smoother for [T, 2] pixel observations."""
    values = np.asarray(values, dtype=np.float64)
    length, dims = values.shape
    state_dim = dims * 2
    transition = np.eye(state_dim)
    for dim in range(dims):
        transition[dim, dims + dim] = 1.0
    observation = np.zeros((dims, state_dim))
    for dim in range(dims):
        observation[dim, dim] = 1.0
    process = np.eye(state_dim) * process_noise
    measurement = np.eye(dims) * measurement_noise
    means = np.zeros((length, state_dim))
    covariances = np.zeros((length, state_dim, state_dim))
    predicted_means = np.zeros_like(means)
    predicted_covariances = np.zeros_like(covariances)
    means[0, :dims] = values[0]
    covariances[0] = np.eye(state_dim) * 100.0
    for index in range(1, length):
        predicted_means[index] = transition @ means[index - 1]
        predicted_covariances[index] = (
            transition @ covariances[index - 1] @ transition.T + process)
        innovation_cov = (
            observation @ predicted_covariances[index] @ observation.T
            + measurement)
        gain = (predicted_covariances[index] @ observation.T
                @ np.linalg.inv(innovation_cov))
        means[index] = predicted_means[index] + gain @ (
            values[index] - observation @ predicted_means[index])
        covariances[index] = (
            (np.eye(state_dim) - gain @ observation)
            @ predicted_covariances[index])
    smoothed_means = means.copy()
    smoothed_covariances = covariances.copy()
    for index in range(length - 2, -1, -1):
        gain = (covariances[index] @ transition.T
                @ np.linalg.inv(predicted_covariances[index + 1]))
        smoothed_means[index] = means[index] + gain @ (
            smoothed_means[index + 1] - predicted_means[index + 1])
        smoothed_covariances[index] = covariances[index] + gain @ (
            smoothed_covariances[index + 1] - predicted_covariances[index + 1]
            ) @ gain.T
    return smoothed_means[:, :dims]


def build_filter_variants():
    variants = {
        'raw': lambda x: x,
        'ma3': lambda x: moving_average(x, 3),
        'ma5': lambda x: moving_average(x, 5),
        'ma7': lambda x: moving_average(x, 7),
        'ma9': lambda x: moving_average(x, 9),
        'ma11': lambda x: moving_average(x, 11),
        'ma13': lambda x: moving_average(x, 13),
        'ma15': lambda x: moving_average(x, 15),
        'median3': lambda x: median_filter(x, size=(3, 1), mode='nearest'),
        'median5': lambda x: median_filter(x, size=(5, 1), mode='nearest'),
        'median7': lambda x: median_filter(x, size=(7, 1), mode='nearest'),
        'gauss1': lambda x: gaussian_filter1d(x, 1.0, axis=0, mode='nearest'),
        'gauss2': lambda x: gaussian_filter1d(x, 2.0, axis=0, mode='nearest'),
        'savgol5': lambda x: savgol_filter(x, 5, 2, axis=0, mode='interp'),
        'savgol7': lambda x: savgol_filter(x, 7, 2, axis=0, mode='interp'),
        'savgol9': lambda x: savgol_filter(x, 9, 2, axis=0, mode='interp'),
    }
    for process_noise in (0.01, 0.1, 1.0, 10.0, 100.0):
        for measurement_noise in (1.0, 4.0, 9.0):
            name = f'rts_q{process_noise:g}_r{measurement_noise:g}'
            variants[name] = (
                lambda x, q=process_noise, r=measurement_noise:
                rts_smooth(x, q, r))
    return variants


def main():
    parser = argparse.ArgumentParser('STNet offline temporal filter evaluation')
    parser.add_argument('--h5', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    h5_path = Path(args.h5).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model, config = build_model(checkpoint)
    model.to(device).eval()
    divisor = float(config.get('input_divisor', 1.0))
    scale = np.array([80.0, 60.0], dtype=np.float64)

    variants = build_filter_variants()

    errors = {name: [] for name in variants}
    with h5py.File(h5_path, 'r') as handle, torch.inference_mode():
        label_key = 'label' if 'label' in handle else 'labels'
        count = len(handle['frames'])
        selected = count if args.limit <= 0 else min(count, args.limit)
        for sequence in range(selected):
            frames = torch.from_numpy(
                handle['frames'][sequence].astype(np.float32)).to(device)
            if divisor != 1.0:
                frames = frames / divisor
            logits_x, logits_y = model(frames.unsqueeze(0))
            prediction = soft_expectation_decode(
                logits_x, logits_y)[0].astype(np.float64)
            labels = handle[label_key][sequence, :, :2].astype(np.float64)
            for name, filter_fn in variants.items():
                filtered = filter_fn(prediction)
                error = np.linalg.norm((filtered - labels) * scale, axis=-1)
                errors[name].append(error)

    report = {
        'schema': 1,
        'h5': str(h5_path),
        'h5_sha256': hashlib.sha256(h5_path.read_bytes()).hexdigest(),
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': hashlib.sha256(
            checkpoint_path.read_bytes()).hexdigest(),
        'chunks_evaluated': selected,
        'metrics': {name: summarize(np.concatenate(values))
                    for name, values in errors.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    ranked = sorted(
        report['metrics'].items(),
        key=lambda item: (-item[1]['p3'], item[1]['mean_error_px']))
    for name, metrics in ranked:
        print(f"{name:24s} p3={metrics['p3']:.4f} "
              f"mean={metrics['mean_error_px']:.4f} "
              f"median={metrics['median_error_px']:.4f}", flush=True)


if __name__ == '__main__':
    main()
