#!/usr/bin/env python3
"""Score an STNet checkpoint on several split protocols."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from ktcal_paper.data import load_manifest
from ktcal_paper.stnet_sequence import (
    RecordChunkDataset,
    evaluate_stnet,
    random_chunk_split,
)
from models.STNet import STNet

def build_dataset(manifest_path, mode, window, val_ratio, seed):
    if mode == 'random':
        recordings, _, val_indices, _ = random_chunk_split(
            manifest_path, window, val_ratio=val_ratio, seed=seed)
        return RecordChunkDataset(
            manifest_path, recordings, window=window,
            chunk_indices=val_indices)
    if mode in ('record', 'all'):
        name = 'val' if mode == 'record' else 'train'
        return RecordChunkDataset(manifest_path, name, window=window)
    raise ValueError(f'Unknown mode: {mode}')


def main():
    parser = argparse.ArgumentParser('Score an STNet checkpoint')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--mode', required=True,
                        choices=('random', 'record', 'all'))
    parser.add_argument('--window', type=int, default=None)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = state.get('config', {}) if isinstance(state, dict) else {}
    window = args.window or int(config.get('window', 32))
    load_manifest(manifest_path)
    dataset = build_dataset(
        manifest_path, args.mode, window, args.val_ratio, args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model_args = SimpleNamespace(
        stnet_block3_kernel=5, stnet_unidirectional=False, stnet_decode='soft',
        stnet_readout=config.get('readout', 'last'),
        stnet_temporal_spatial_kernel=int(config.get('temporal_spatial_kernel', 1)))
    model = STNet(model_args)
    model.load_state_dict(state['model_state_dict'], strict=True)
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    model.to(device)
    metrics = evaluate_stnet(model, loader, device)
    dataset.close()
    report = {
        'checkpoint': str(checkpoint_path),
        'manifest': str(manifest_path),
        'mode': args.mode,
        'window': window,
        'val_ratio': args.val_ratio if args.mode == 'random' else None,
        'seed': args.seed if args.mode == 'random' else None,
        'checkpoint_best_validation_error_px': state.get(
            'best_validation_error_px') if isinstance(state, dict) else None,
        'architecture': 'stnet',
        'metrics': metrics,
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
