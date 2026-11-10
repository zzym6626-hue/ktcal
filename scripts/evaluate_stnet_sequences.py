#!/usr/bin/env python3
"""Evaluate an STNet checkpoint on whole recordings (full temporal context)."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from ktcal_paper.data import load_manifest
from models.STNet import STNet


def soft_decode(logits, bins):
    probability = torch.softmax(logits, dim=-1)
    grid = torch.arange(bins, dtype=probability.dtype)
    return (probability * grid).sum(-1) / bins


def summarize(errors):
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


def window_last_steps(length, size):
    """Last index of each non-overlapping window of ``size`` in a sequence."""
    if size < 1:
        raise ValueError('Window size must be positive')
    return list(range(size - 1, length, size))


def main():
    parser = argparse.ArgumentParser('Full-sequence STNet evaluation')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--split', default='val',
                        choices=('train', 'val', 'holdout', 'all'))
    parser.add_argument('--window-sizes', default='1,2,4,8,16,32,64')
    parser.add_argument('--output', default=None)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = state.get('config', {})
    window_sizes = [int(item) for item in args.window_sizes.split(',') if item]

    if args.split == 'all':
        names = [name for split in ('train', 'val', 'holdout')
                 for name in manifest['splits'][split]]
    else:
        names = list(manifest['splits'][args.split])

    model_args = SimpleNamespace(
        stnet_block3_kernel=5, stnet_unidirectional=False, stnet_decode='soft',
        stnet_readout=config.get('readout', 'last'),
        stnet_temporal_spatial_kernel=int(
            config.get('temporal_spatial_kernel', 1)))
    model = STNet(model_args)
    model.load_state_dict(state['model_state_dict'])
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model.to(device).eval()

    h5_path = manifest_path.parent / manifest['h5']
    scale = torch.tensor([80.0, 60.0])
    step_errors = []
    last_step_errors = []
    window_errors = {size: [] for size in window_sizes}
    with h5py.File(h5_path, 'r') as handle, torch.inference_mode():
        for name in names:
            group = handle[name]
            frames = group['frames'][:].astype(np.float32) / 15
            labels = group['labels'][:].astype(np.float32)
            inputs = torch.from_numpy(frames).unsqueeze(0).to(device)
            predict_w, predict_h = model(inputs)
            prediction = torch.stack((
                soft_decode(predict_w[0], 80), soft_decode(predict_h[0], 60)),
                dim=-1).cpu()
            target = torch.from_numpy(labels[:, :2])
            errors = torch.linalg.vector_norm(
                (prediction - target) * scale, dim=-1).numpy()
            step_errors.append(errors)
            last_step_errors.append(errors[-1])
            for size in window_sizes:
                ends = window_last_steps(len(errors), size)
                if ends:
                    window_errors[size].append(errors[ends])
            print(json.dumps({
                'recording': name, 'frames': int(len(errors)),
                'mean_error_px': float(errors.mean())}), flush=True)

    all_steps = np.concatenate(step_errors)
    report = {
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': hashlib.sha256(
            checkpoint_path.read_bytes()).hexdigest(),
        'manifest': str(manifest_path),
        'split': args.split,
        'recordings': len(names),
        'window': config.get('window'),
        'training_split_mode': config.get('split_mode'),
        'checkpoint_best_validation_error_px': state.get(
            'best_validation_error_px'),
        'full_sequence_all_steps': summarize(all_steps),
        'full_sequence_last_step': summarize(last_step_errors),
        'window_last_step_error_px': {
            str(size): float(np.concatenate(values).mean())
            for size, values in window_errors.items() if values},
    }
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text, flush=True)


if __name__ == '__main__':
    main()
