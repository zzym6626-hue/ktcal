"""Train NCC on cached closed-loop Kalman/template deployment states."""
import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from models.ncc_matcher import NCCMatcher, hard_argmax_2d, soft_argmax_2d
from train_ktcal import load_model_weights, seed_everything


class RolloutNCCDataset(Dataset):
    def __init__(self, h5_path, cache_path, template_size=11, search_size=21,
                 template_mode='historical'):
        with h5py.File(h5_path, 'r') as handle:
            self.frames = torch.from_numpy(handle['frames'][:]).float()
            self.labels = torch.from_numpy(handle['label'][:, :, :2]).float()
        cache = np.load(cache_path, allow_pickle=False)
        self.metadata = json.loads(str(cache['metadata_json'].item()))
        self.sequence_index = torch.from_numpy(cache['sequence_index'].astype(np.int64))
        self.frame_id = torch.from_numpy(cache['frame_id'].astype(np.int64))
        self.template_frame_id = torch.from_numpy(cache['template_frame_id'].astype(np.int64))
        self.template_center = torch.from_numpy(cache['template_center'].astype(np.float32))
        self.search_center = torch.from_numpy(cache['search_center'].astype(np.float32))
        self.target_cell = torch.from_numpy(cache['target_cell'].astype(np.float32))
        self.matchable = torch.from_numpy(cache['matchable'].astype(np.bool_))
        self.boundary_peak = torch.from_numpy(cache['boundary_peak'].astype(np.bool_))
        self.tier = torch.from_numpy(cache['tier'].astype(np.int64))
        self.template_age = torch.from_numpy(cache['template_age'].astype(np.int64))
        self.template_size = template_size
        self.search_size = search_size
        self.template_mode = template_mode
        if template_mode not in ('historical', 'ground_truth'):
            raise ValueError('template_mode must be historical or ground_truth')
        expected = self.metadata['tracker_config']
        if expected['template_size'] != template_size or expected['search_size'] != search_size:
            raise ValueError('Cache and training template/search sizes do not match')

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
        current_frame = self.frames[sequence_index, int(self.frame_id[index])]
        template_frame_id = int(self.template_frame_id[index])
        template_frame = self.frames[sequence_index, template_frame_id]
        template_center = self.template_center[index]
        if self.template_mode == 'ground_truth':
            template_center = self.labels[sequence_index, template_frame_id]
        template = self._crop(template_frame, template_center, self.template_size)
        search = self._crop(current_frame, self.search_center[index], self.search_size)
        failure = bool(self.boundary_peak[index] or self.tier[index] >= 2)
        return {
            'template': template,
            'search': search,
            'target_cell': self.target_cell[index],
            'matchable': self.matchable[index],
            'failure': failure,
            'tier': self.tier[index],
            'template_age': self.template_age[index],
        }

    def sampling_weights(self, failure_weight, unmatchable_weight, stale_template_weight):
        failure = self.boundary_peak | (self.tier >= 2)
        weights = torch.ones(len(self), dtype=torch.double)
        weights[failure] *= failure_weight
        weights[~self.matchable] *= unmatchable_weight
        weights[self.template_age >= 8] *= stale_template_weight
        return weights


def validate_cache_pair(args, train_dataset, val_dataset):
    train_meta = train_dataset.metadata
    val_meta = val_dataset.metadata
    source = str(Path(args.train_h5_path).resolve())
    if train_meta['source'] != source or val_meta['source'] != source:
        raise ValueError('Rollout cache source does not match --train_h5_path')
    if train_meta['split'] != 'train' or val_meta['split'] != 'val':
        raise ValueError('Expected a train cache and a val cache')
    train_indices = set(train_meta['selected_indices'])
    val_indices = set(val_meta['selected_indices'])
    if train_indices & val_indices:
        raise ValueError('Train and validation rollout caches overlap')
    if not set(train_dataset.sequence_index.tolist()).issubset(train_indices):
        raise ValueError('Train cache contains sequences outside its metadata split')
    if not set(val_dataset.sequence_index.tolist()).issubset(val_indices):
        raise ValueError('Validation cache contains sequences outside its metadata split')
    for key in ('seed', 'val_ratio', 'stnet_ckpt', 'ncc_ckpt', 'tracker_config'):
        if train_meta[key] != val_meta[key]:
            raise ValueError(f'Train/validation rollout metadata mismatch: {key}')


def parse_args():
    parser = argparse.ArgumentParser('Deployment-aware NCC training')
    parser.add_argument('--train_h5_path', default='./data/seet/train_3et.h5')
    parser.add_argument('--train_cache', required=True)
    parser.add_argument('--val_cache', required=True)
    parser.add_argument('--ncc_init_ckpt', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--samples_per_epoch', type=int, default=0,
                        help='Weighted rollout samples drawn per epoch; 0 uses cache size')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--min_delta', type=float, default=0.005)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--template_size', type=int, default=11)
    parser.add_argument('--search_size', type=int, default=21)
    parser.add_argument('--template_mode', choices=['historical', 'ground_truth'],
                        default='historical')
    parser.add_argument('--failure_weight', type=float, default=3.0)
    parser.add_argument('--unmatchable_sample_weight', type=float, default=1.0)
    parser.add_argument('--stale_template_weight', type=float, default=1.5)
    parser.add_argument('--hard_negative_weight', type=float, default=0.1)
    parser.add_argument('--boundary_weight', type=float, default=0.1)
    parser.add_argument('--peak_ce_weight', type=float, default=0.01)
    parser.add_argument('--unmatchable_conf_weight', type=float, default=0.1)
    parser.add_argument('--positive_conf_weight', type=float, default=0.05)
    parser.add_argument('--hard_negative_margin', type=float, default=0.15)
    parser.add_argument('--boundary_margin', type=float, default=0.10)
    parser.add_argument('--positive_conf_margin', type=float, default=0.5)
    parser.add_argument('--unmatchable_conf_margin', type=float, default=0.4)
    parser.add_argument('--positive_radius', type=int, default=1)
    return parser.parse_args()


def rollout_loss(matcher, batch, args):
    template = batch['template']
    search = batch['search']
    target_cell = batch['target_cell']
    matchable = batch['matchable'].bool()
    response = matcher.ncc_response(template, search)
    batch_size, response_h, response_w = response.shape
    target_x_float = target_cell[:, 0]
    target_y_float = target_cell[:, 1]
    target_x = target_x_float.round().long().clamp(0, response_w - 1)
    target_y = target_y_float.round().long().clamp(0, response_h - 1)
    target_normalized = torch.stack([
        target_x_float / max(response_w - 1, 1),
        target_y_float / max(response_h - 1, 1),
    ], dim=-1)

    zero = response.sum() * 0.0
    if matchable.any():
        valid_response = response[matchable]
        valid_target = target_normalized[matchable]
        coord, _ = soft_argmax_2d(valid_response, matcher.temperature)
        coordinate_loss = F.mse_loss(coord, valid_target)
        valid_x = target_x[matchable]
        valid_y = target_y[matchable]
        valid_count = valid_response.shape[0]
        yy = torch.arange(response_h, device=response.device).view(1, response_h, 1)
        xx = torch.arange(response_w, device=response.device).view(1, 1, response_w)
        positive_mask = (
            (xx - valid_x.view(-1, 1, 1)).abs() <= args.positive_radius
        ) & (
            (yy - valid_y.view(-1, 1, 1)).abs() <= args.positive_radius
        )
        rows = torch.arange(valid_count, device=response.device)
        positive_score = valid_response[rows, valid_y, valid_x]
        negative_score = valid_response.masked_fill(
            positive_mask, float('-inf')).flatten(1).max(1).values
        hard_negative_loss = F.relu(
            args.hard_negative_margin - positive_score + negative_score).mean()
        boundary_mask = torch.zeros_like(valid_response, dtype=torch.bool)
        boundary_mask[:, 0, :] = True
        boundary_mask[:, -1, :] = True
        boundary_mask[:, :, 0] = True
        boundary_mask[:, :, -1] = True
        boundary_mask &= ~positive_mask
        boundary_score = valid_response.masked_fill(
            ~boundary_mask, float('-inf')).flatten(1).max(1).values
        boundary_loss = F.relu(
            args.boundary_margin - positive_score + boundary_score).mean()
        target_index = valid_y * response_w + valid_x
        peak_ce_loss = F.cross_entropy(
            valid_response.flatten(1) / matcher.temperature, target_index)
        positive_conf_loss = F.relu(
            args.positive_conf_margin - positive_score).mean()
    else:
        coordinate_loss = hard_negative_loss = boundary_loss = zero
        peak_ce_loss = positive_conf_loss = zero

    if (~matchable).any():
        unmatchable_peak = response[~matchable].flatten(1).max(1).values
        unmatchable_conf_loss = F.relu(
            unmatchable_peak - args.unmatchable_conf_margin).mean()
    else:
        unmatchable_conf_loss = zero

    total = (
        coordinate_loss
        + args.hard_negative_weight * hard_negative_loss
        + args.boundary_weight * boundary_loss
        + args.peak_ce_weight * peak_ce_loss
        + args.positive_conf_weight * positive_conf_loss
        + args.unmatchable_conf_weight * unmatchable_conf_loss
    )
    metrics = {
        'coordinate': coordinate_loss.detach(),
        'hard_negative': hard_negative_loss.detach(),
        'boundary': boundary_loss.detach(),
        'peak_ce': peak_ce_loss.detach(),
        'positive_conf': positive_conf_loss.detach(),
        'unmatchable_conf': unmatchable_conf_loss.detach(),
    }
    return total, metrics


@torch.no_grad()
def evaluate(matcher, loader, args, device):
    matcher.eval()
    total_error = 0.0
    valid_count = 0
    boundary_count = 0
    false_confident = 0
    unmatchable_count = 0
    low_conf_valid = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        response = matcher.ncc_response(batch['template'], batch['search'])
        coord, confidence = hard_argmax_2d(response)
        matchable = batch['matchable'].bool()
        response_h, response_w = response.shape[1:]
        peak_x = (coord[:, 0] * (response_w - 1)).round().long()
        peak_y = (coord[:, 1] * (response_h - 1)).round().long()
        boundary_count += int(((peak_x == 0) | (peak_x == response_w - 1)
                               | (peak_y == 0) | (peak_y == response_h - 1)).sum())
        if matchable.any():
            target = torch.stack([
                batch['target_cell'][:, 0] / max(response_w - 1, 1),
                batch['target_cell'][:, 1] / max(response_h - 1, 1),
            ], dim=-1)
            error = torch.linalg.vector_norm(
                (coord[matchable] - target[matchable])
                * response.new_tensor([response_w - 1, response_h - 1]), dim=-1)
            total_error += float(error.sum())
            valid_count += int(matchable.sum())
            low_conf_valid += int((confidence[matchable] < 0.5).sum())
        if (~matchable).any():
            false_confident += int((confidence[~matchable] >= 0.5).sum())
            unmatchable_count += int((~matchable).sum())
    sample_count = max(len(loader.dataset), 1)
    metrics = {
        'matchable_hard_error_cells': total_error / max(valid_count, 1),
        'boundary_rate': boundary_count / sample_count,
        'matchable_low_conf_rate': low_conf_valid / max(valid_count, 1),
        'unmatchable_false_conf_rate': false_confident / max(unmatchable_count, 1),
        'matchable_samples': valid_count,
        'unmatchable_samples': unmatchable_count,
    }
    metrics['selection_score'] = (
        metrics['matchable_hard_error_cells']
        + metrics['boundary_rate']
        + metrics['matchable_low_conf_rate']
        + metrics['unmatchable_false_conf_rate']
    )
    return metrics


def checkpoint_payload(matcher, args, epoch, metrics, train_metadata, val_metadata):
    return {
        'format_version': 2,
        'model_state_dict': matcher.state_dict(),
        'epoch': epoch,
        'selection_metric': 'rollout_error_plus_boundary_plus_false_confidence',
        'best_metrics': metrics,
        'seed': args.seed,
        'config': vars(args),
        'rollout_metadata': {'train': train_metadata, 'val': val_metadata},
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cpu' if args.use_cpu else 'cuda')
    train_dataset = RolloutNCCDataset(
        args.train_h5_path, args.train_cache, args.template_size, args.search_size,
        args.template_mode)
    val_dataset = RolloutNCCDataset(
        args.train_h5_path, args.val_cache, args.template_size, args.search_size,
        args.template_mode)
    validate_cache_pair(args, train_dataset, val_dataset)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        train_dataset.sampling_weights(
            args.failure_weight, args.unmatchable_sample_weight,
            args.stale_template_weight),
        num_samples=args.samples_per_epoch or len(train_dataset),
        replacement=True, generator=generator)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    matcher = NCCMatcher().to(device)
    load_model_weights(matcher, args.ncc_init_ckpt, device)
    optimizer = torch.optim.AdamW(matcher.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epoch, 1), eta_min=1e-6)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    epoch_path = save_path / 'epochs'
    epoch_path.mkdir(parents=True, exist_ok=True)
    with (save_path / 'run_config.json').open('w', encoding='utf-8') as handle:
        json.dump({
            'config': vars(args),
            'train_rollout': train_dataset.metadata,
            'val_rollout': val_dataset.metadata,
        }, handle, indent=2)

    baseline = evaluate(matcher, val_loader, args, device)
    print(f"[val] epoch 0 baseline: {json.dumps(baseline, sort_keys=True)}")
    best_score = baseline['selection_score']
    best_metrics = baseline
    baseline_payload = checkpoint_payload(
        matcher, args, 0, baseline, train_dataset.metadata, val_dataset.metadata)
    torch.save(baseline_payload, save_path / 'ncc_baseline.pth')
    torch.save(baseline_payload, save_path / 'ncc_best.pth')
    torch.save(baseline_payload, epoch_path / 'epoch_000.pth')
    epochs_no_improve = 0

    for epoch in range(args.epoch):
        matcher.train()
        running = {key: 0.0 for key in (
            'total', 'coordinate', 'hard_negative', 'boundary', 'peak_ce',
            'positive_conf', 'unmatchable_conf')}
        batch_count = 0
        for batch in tqdm(train_loader, desc=f'rollout NCC epoch {epoch + 1}'):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss, metrics = rollout_loss(matcher, batch, args)
            loss.backward()
            optimizer.step()
            running['total'] += float(loss.detach())
            for key in metrics:
                running[key] += float(metrics[key])
            batch_count += 1
        scheduler.step()
        summary = {key: value / max(batch_count, 1) for key, value in running.items()}
        val_metrics = evaluate(matcher, val_loader, args, device)
        print(f"[train] epoch {epoch + 1}: {json.dumps(summary, sort_keys=True)}")
        print(f"[val] epoch {epoch + 1}: {json.dumps(val_metrics, sort_keys=True)}")
        epoch_payload = checkpoint_payload(
            matcher, args, epoch + 1, val_metrics,
            train_dataset.metadata, val_dataset.metadata)
        torch.save(epoch_payload, save_path / 'ncc_last.pth')
        torch.save(epoch_payload, epoch_path / f'epoch_{epoch + 1:03d}.pth')
        if val_metrics['selection_score'] < best_score - args.min_delta:
            best_score = val_metrics['selection_score']
            best_metrics = val_metrics
            epochs_no_improve = 0
            torch.save(
                checkpoint_payload(matcher, args, epoch + 1, val_metrics,
                                   train_dataset.metadata, val_dataset.metadata),
                save_path / 'ncc_best.pth')
            print(f'  best rollout NCC saved (score={best_score:.4f})')
        else:
            epochs_no_improve += 1
            print(f'  no improve for {epochs_no_improve}/{args.patience} (best={best_score:.4f})')
            if epochs_no_improve >= args.patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    print(f'best validation metrics: {json.dumps(best_metrics, sort_keys=True)}')


if __name__ == '__main__':
    main()
