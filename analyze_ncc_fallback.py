"""Analyze whether a larger NCC search can safely rescue local-match failures."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.ncc_matcher import NCCMatcher, hard_argmax_2d
from train_ktcal import load_model_weights


class FallbackDataset(Dataset):
    def __init__(self, h5_path, cache_path, template_size, fallback_search_size):
        with h5py.File(h5_path, 'r') as handle:
            self.frames = torch.from_numpy(handle['frames'][:]).float()
        cache = np.load(cache_path, allow_pickle=False)
        self.sequence_index = torch.from_numpy(cache['sequence_index'].astype(np.int64))
        self.frame_id = torch.from_numpy(cache['frame_id'].astype(np.int64))
        self.template_frame_id = torch.from_numpy(cache['template_frame_id'].astype(np.int64))
        self.template_center = torch.from_numpy(cache['template_center'].astype(np.float32))
        self.search_center = torch.from_numpy(cache['search_center'].astype(np.float32))
        self.ground_truth = torch.from_numpy(cache['ground_truth'].astype(np.float32))
        self.local_coord = torch.from_numpy(cache['ncc_coord'].astype(np.float32))
        self.local_confidence = torch.from_numpy(cache['confidence'].astype(np.float32))
        self.local_boundary = torch.from_numpy(cache['boundary_peak'].astype(np.bool_))
        self.template_size = template_size
        self.fallback_search_size = fallback_search_size

    def __len__(self):
        return len(self.frame_id)

    @staticmethod
    def _crop(frame, center, size):
        channels, height, width = frame.shape
        center_x = int(round(float(center[0].clamp(0, 1)) * (width - 1)))
        center_y = int(round(float(center[1].clamp(0, 1)) * (height - 1)))
        half = size // 2
        patch = torch.zeros(channels, size, size, dtype=frame.dtype)
        x0, x1 = center_x - half, center_x + half + 1
        y0, y1 = center_y - half, center_y + half + 1
        source_x0, source_x1 = max(x0, 0), min(x1, width)
        source_y0, source_y1 = max(y0, 0), min(y1, height)
        patch_x0, patch_y0 = source_x0 - x0, source_y0 - y0
        patch[:, patch_y0:patch_y0 + source_y1 - source_y0,
              patch_x0:patch_x0 + source_x1 - source_x0] = (
            frame[:, source_y0:source_y1, source_x0:source_x1]
        )
        return patch

    def __getitem__(self, index):
        sequence_index = int(self.sequence_index[index])
        template_frame = self.frames[sequence_index, int(self.template_frame_id[index])]
        current_frame = self.frames[sequence_index, int(self.frame_id[index])]
        return {
            'template': self._crop(
                template_frame, self.template_center[index], self.template_size),
            'search': self._crop(
                current_frame, self.search_center[index], self.fallback_search_size),
            'search_center': self.search_center[index],
            'ground_truth': self.ground_truth[index],
            'local_coord': self.local_coord[index],
            'local_confidence': self.local_confidence[index],
            'local_boundary': self.local_boundary[index],
        }


def parse_args():
    parser = argparse.ArgumentParser('Analyze NCC fallback gating')
    parser.add_argument('--h5_path', default='./data/seet/train_3et.h5')
    parser.add_argument('--cache_path', required=True)
    parser.add_argument('--ncc_ckpt', required=True)
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--template_size', type=int, default=11)
    parser.add_argument('--fallback_search_size', type=int, default=31)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cpu' if args.use_cpu else 'cuda')
    dataset = FallbackDataset(
        args.h5_path, args.cache_path, args.template_size, args.fallback_search_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    matcher = NCCMatcher().to(device)
    load_model_weights(matcher, args.ncc_ckpt, device)
    matcher.eval()
    fallback_coords = []
    fallback_confidences = []
    fallback_boundaries = []
    local_coords = []
    local_confidences = []
    local_boundaries = []
    search_centers = []
    ground_truth = []
    response_size = args.fallback_search_size - args.template_size + 1
    half_search = args.fallback_search_size // 2
    half_template = args.template_size // 2

    with torch.no_grad():
        for batch in loader:
            response = matcher.ncc_response(
                batch['template'].to(device), batch['search'].to(device))
            coord, confidence = hard_argmax_2d(response)
            cell_x = coord[:, 0] * (response_size - 1)
            cell_y = coord[:, 1] * (response_size - 1)
            center = batch['search_center'].to(device)
            center_x = torch.round(center[:, 0] * 79.0)
            center_y = torch.round(center[:, 1] * 59.0)
            absolute_x = (center_x - half_search + cell_x + half_template) / 79.0
            absolute_y = (center_y - half_search + cell_y + half_template) / 59.0
            peak_x = cell_x.round().long()
            peak_y = cell_y.round().long()
            boundary = ((peak_x == 0) | (peak_x == response_size - 1)
                        | (peak_y == 0) | (peak_y == response_size - 1))
            fallback_coords.append(torch.stack([absolute_x, absolute_y], dim=-1).cpu())
            fallback_confidences.append(confidence.cpu())
            fallback_boundaries.append(boundary.cpu())
            local_coords.append(batch['local_coord'])
            local_confidences.append(batch['local_confidence'])
            local_boundaries.append(batch['local_boundary'])
            search_centers.append(batch['search_center'])
            ground_truth.append(batch['ground_truth'])

    fallback_coords = torch.cat(fallback_coords).numpy()
    fallback_confidences = torch.cat(fallback_confidences).numpy()
    fallback_boundaries = torch.cat(fallback_boundaries).numpy()
    local_coords = torch.cat(local_coords).numpy()
    local_confidences = torch.cat(local_confidences).numpy()
    local_boundaries = torch.cat(local_boundaries).numpy()
    search_centers = torch.cat(search_centers).numpy()
    ground_truth = torch.cat(ground_truth).numpy()
    pixel_scale = np.array([80.0, 60.0])
    fallback_error = np.linalg.norm((fallback_coords - ground_truth) * pixel_scale, axis=-1)
    local_error = np.linalg.norm((local_coords - ground_truth) * pixel_scale, axis=-1)
    kalman_error = np.linalg.norm((search_centers - ground_truth) * pixel_scale, axis=-1)
    fallback_epsilon = np.linalg.norm((fallback_coords - search_centers) * pixel_scale, axis=-1)
    attempted = (local_confidences < 0.5) | local_boundaries

    results = {
        'samples': len(dataset),
        'attempted': int(attempted.sum()),
        'attempted_ratio': float(attempted.mean()),
        'attempted_baseline': {
            'local_error_px': float(local_error[attempted].mean()),
            'kalman_error_px': float(kalman_error[attempted].mean()),
            'fallback_error_px': float(fallback_error[attempted].mean()),
            'fallback_better_than_local': float(
                (fallback_error[attempted] < local_error[attempted]).mean()),
            'fallback_better_than_kalman': float(
                (fallback_error[attempted] < kalman_error[attempted]).mean()),
        },
        'gates': [],
    }
    for threshold in (0.5, 0.6, 0.7, 0.8):
        for margin in (0.0, 0.05, 0.10, 0.15):
            accepted = (attempted
                        & ~fallback_boundaries
                        & (fallback_confidences >= threshold)
                        & (fallback_confidences >= local_confidences + margin))
            count = int(accepted.sum())
            if count == 0:
                continue
            results['gates'].append({
                'confidence_threshold': threshold,
                'confidence_margin': margin,
                'accepted': count,
                'accepted_ratio': float(accepted.mean()),
                'fallback_error_px': float(fallback_error[accepted].mean()),
                'local_error_px': float(local_error[accepted].mean()),
                'kalman_error_px': float(kalman_error[accepted].mean()),
                'fallback_epsilon_px': float(fallback_epsilon[accepted].mean()),
                'fallback_better_than_local': float(
                    (fallback_error[accepted] < local_error[accepted]).mean()),
                'fallback_better_than_kalman': float(
                    (fallback_error[accepted] < kalman_error[accepted]).mean()),
            })
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
