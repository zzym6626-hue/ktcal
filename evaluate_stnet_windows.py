"""Evaluate STNet by deployment window length on the fixed validation split."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import provider_data
from metrics import decode_soft_expectation
from models.STNet import STNet
from train_ktcal import load_model_weights, seed_everything, split_train_val


def parse_args():
    parser = argparse.ArgumentParser('STNet per-window validation')
    parser.add_argument('--train_h5_path', default='./data/seet/train_3et.h5')
    parser.add_argument('--stnet_ckpt', required=True)
    parser.add_argument('--stnet_block3_kernel', type=int, choices=[5, 7], default=5)
    parser.add_argument('--stnet_unidirectional', action='store_true', default=False)
    parser.add_argument('--window_sizes', default='1,2,4,8,16')
    parser.add_argument('--windows_per_sequence', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--output_json', default=None)
    return parser.parse_args()


def pixel_error(prediction, target):
    scale = prediction.new_tensor([80.0, 60.0])
    return torch.linalg.vector_norm((prediction - target) * scale, dim=-1)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cpu' if args.use_cpu else 'cuda')
    frame_array, label_array = provider_data.load_h5_f(args.train_h5_path)
    frames = torch.from_numpy(np.asarray(frame_array)).float()
    labels = torch.from_numpy(np.asarray(label_array)).float()
    _, _, val_frames, val_labels, _, _ = split_train_val(
        frames, labels, args.val_ratio, args.seed)
    loader = DataLoader(
        TensorDataset(val_frames, val_labels), batch_size=args.batch_size,
        shuffle=False, num_workers=0)
    model = STNet(args).to(device)
    load_model_weights(model, args.stnet_ckpt, device)
    model.eval()
    window_sizes = [int(value) for value in args.window_sizes.split(',')]
    generator = torch.Generator().manual_seed(args.seed + 20000)
    totals = {
        window: {'error': 0.0, 'count': 0, 'p3': 0, 'p5': 0}
        for window in window_sizes
    }
    full_metrics = {
        'all': {'error': 0.0, 'count': 0, 'p3': 0, 'p5': 0},
        'last': {'error': 0.0, 'count': 0, 'p3': 0, 'p5': 0},
    }

    with torch.no_grad():
        for batch_frames, batch_labels in loader:
            batch_frames = batch_frames.to(device)
            batch_labels = batch_labels.to(device)
            predict_w, predict_h = model(batch_frames)
            full_coords = decode_soft_expectation(predict_w, predict_h)
            full_all_error = pixel_error(full_coords, batch_labels[:, :, :2])
            full_last_error = full_all_error[:, -1]
            for name, error in (('all', full_all_error), ('last', full_last_error)):
                full_metrics[name]['error'] += float(error.sum())
                full_metrics[name]['count'] += error.numel()
                full_metrics[name]['p3'] += int((error < 3.0).sum())
                full_metrics[name]['p5'] += int((error < 5.0).sum())

            batch_size, sequence_length = batch_frames.shape[:2]
            for window_size in window_sizes:
                for _ in range(args.windows_per_sequence):
                    starts = torch.randint(
                        0, sequence_length - window_size + 1,
                        (batch_size,), generator=generator)
                    offsets = torch.arange(window_size).unsqueeze(0)
                    indices = starts.unsqueeze(1) + offsets
                    batch_indices = torch.arange(batch_size).unsqueeze(1)
                    window_frames = batch_frames[batch_indices, indices]
                    target = batch_labels[batch_indices[:, 0], indices[:, -1], :2]
                    coord = model.last_step(window_frames)
                    error = pixel_error(coord, target)
                    metric = totals[window_size]
                    metric['error'] += float(error.sum())
                    metric['count'] += error.numel()
                    metric['p3'] += int((error < 3.0).sum())
                    metric['p5'] += int((error < 5.0).sum())

    results = {
        'checkpoint': str(Path(args.stnet_ckpt).resolve()),
        'full_sequence_all_steps_error_px': (
            full_metrics['all']['error'] / full_metrics['all']['count']),
        'full_sequence_last_step_error_px': (
            full_metrics['last']['error'] / full_metrics['last']['count']),
        'full_sequence_accuracy': {
            name: {
                'p3': metric['p3'] / metric['count'],
                'p5': metric['p5'] / metric['count'],
            }
            for name, metric in full_metrics.items()
        },
        'window_last_step_error_px': {
            str(window): metric['error'] / metric['count']
            for window, metric in totals.items()
        },
        'window_last_step_accuracy': {
            str(window): {
                'p3': metric['p3'] / metric['count'],
                'p5': metric['p5'] / metric['count'],
            }
            for window, metric in totals.items()
        },
        'config': vars(args),
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as handle:
            json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
