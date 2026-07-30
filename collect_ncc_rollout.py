"""Collect deployment-distribution NCC samples from KTCal closed-loop rollouts."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import provider_data
from ktcal_tracker import KTCalTracker
from models.STNet import STNet
from models.ncc_matcher import NCCMatcher
from train_ktcal import load_model_weights, seed_everything, split_train_val


TIER_IDS = {
    'tier1': 0,
    'tier2': 1,
    'tier3_low_conf': 2,
    'tier3_large_error': 3,
}


def parse_args():
    parser = argparse.ArgumentParser('Collect NCC closed-loop rollout samples')
    parser.add_argument('--train_h5_path', default='./data/seet/train_3et.h5')
    parser.add_argument('--stnet_ckpt', required=True)
    parser.add_argument('--stnet_block3_kernel', type=int, choices=[5, 7], default=5)
    parser.add_argument('--ncc_ckpt', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_sequences', type=int, default=0)
    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--tau1', type=float, default=3.0)
    parser.add_argument('--tau2', type=float, default=5.0)
    parser.add_argument('--tau3', type=float, default=1.0)
    parser.add_argument('--ncc_thresh', type=float, default=0.5)
    parser.add_argument('--template_size', type=int, default=11)
    parser.add_argument('--search_size', type=int, default=21)
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--reset_velocity_mode',
                        choices=['legacy', 'previous_position', 'zero', 'preserve'],
                        default='legacy')
    parser.add_argument('--low_conf_policy',
                        choices=['sync_stnet', 'kalman', 'async_stnet', 'hold', 'hold_async'],
                        default='sync_stnet')
    parser.add_argument('--kalman_process_noise', type=float, default=1e-3)
    parser.add_argument('--kalman_measurement_noise', type=float, default=1e-2)
    parser.add_argument('--kalman_velocity_decay', type=float, default=1.0)
    parser.add_argument('--soft_fusion', action='store_true', default=False,
                        help='Disable the deployed tau3 hard reset during rollout')
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cpu' if args.use_cpu else 'cuda')
    frames_np, labels_np = provider_data.load_h5_f(args.train_h5_path)
    frames = torch.from_numpy(np.asarray(frames_np)).float()
    labels = torch.from_numpy(np.asarray(labels_np)).float()
    (_, _, _, _, train_indices, val_indices) = split_train_val(
        frames, labels, args.val_ratio, args.seed)
    selected_indices = train_indices if args.split == 'train' else val_indices
    if args.max_sequences:
        selected_indices = selected_indices[:args.max_sequences]
    if not selected_indices:
        raise ValueError('No sequences selected for rollout collection')

    stnet = STNet(args).to(device)
    matcher = NCCMatcher().to(device)
    load_model_weights(stnet, args.stnet_ckpt, device)
    load_model_weights(matcher, args.ncc_ckpt, device)
    stnet.eval()
    matcher.eval()
    tracker = KTCalTracker(
        stnet, matcher,
        tau1=args.tau1, tau2=args.tau2, tau3=args.tau3,
        ncc_thresh=args.ncc_thresh,
        template_size=args.template_size, search_size=args.search_size,
        window_size=args.window_size, hard_reset=not args.soft_fusion,
        collect_rollout=True, device=device,
        reset_velocity_mode=args.reset_velocity_mode,
        low_conf_policy=args.low_conf_policy,
        kalman_process_noise=args.kalman_process_noise,
        kalman_measurement_noise=args.kalman_measurement_noise,
        kalman_velocity_decay=args.kalman_velocity_decay,
    )

    records = {
        'sequence_index': [],
        'frame_id': [],
        'template_frame_id': [],
        'template_center': [],
        'search_center': [],
        'ncc_coord': [],
        'ground_truth': [],
        'target_cell': [],
        'confidence': [],
        'epsilon': [],
        'boundary_peak': [],
        'matchable': [],
        'tier': [],
        'template_age': [],
        'kalman_error_px': [],
        'ncc_error_px': [],
    }
    width, height = frames.shape[-1], frames.shape[-2]
    half_search = args.search_size // 2
    half_template = args.template_size // 2
    response_size = args.search_size - args.template_size + 1
    pixel_scale = np.array([float(width), float(height)])

    aggregate = {
        'sequences': len(selected_indices), 'samples': 0, 'matchable': 0,
        'boundary': 0, 'low_conf': 0, 'large_error': 0,
    }
    with torch.no_grad():
        for sequence_index in tqdm(selected_indices, desc=f'rollout {args.split}'):
            _, stats = tracker.track_sequence(frames[sequence_index])
            sequence_labels = labels[sequence_index, :, :2].numpy()
            for sample in stats['rollout_samples']:
                frame_id = sample['frame_id']
                gt = sequence_labels[frame_id].astype(np.float64)
                search_center = np.asarray(sample['search_center'], dtype=np.float64)
                search_x = int(round(np.clip(search_center[0], 0.0, 1.0) * (width - 1)))
                search_y = int(round(np.clip(search_center[1], 0.0, 1.0) * (height - 1)))
                gt_x = gt[0] * (width - 1)
                gt_y = gt[1] * (height - 1)
                target_cell = np.array([
                    gt_x - (search_x - half_search) - half_template,
                    gt_y - (search_y - half_search) - half_template,
                ])
                matchable = bool(
                    0.0 <= target_cell[0] <= response_size - 1
                    and 0.0 <= target_cell[1] <= response_size - 1
                )
                epsilon = sample['epsilon']
                tier = TIER_IDS[sample['tier']]
                ncc_coord = np.asarray(sample['ncc_coord'], dtype=np.float64)
                kalman_error = float(np.linalg.norm((search_center - gt) * pixel_scale))
                ncc_error = float(np.linalg.norm((ncc_coord - gt) * pixel_scale))

                records['sequence_index'].append(sequence_index)
                records['frame_id'].append(frame_id)
                records['template_frame_id'].append(sample['template_frame_id'])
                records['template_center'].append(sample['template_center'])
                records['search_center'].append(search_center)
                records['ncc_coord'].append(ncc_coord)
                records['ground_truth'].append(gt)
                records['target_cell'].append(target_cell)
                records['confidence'].append(sample['confidence'])
                records['epsilon'].append(np.nan if epsilon is None else epsilon)
                records['boundary_peak'].append(sample['boundary_peak'])
                records['matchable'].append(matchable)
                records['tier'].append(tier)
                records['template_age'].append(frame_id - sample['template_frame_id'])
                records['kalman_error_px'].append(kalman_error)
                records['ncc_error_px'].append(ncc_error)

                aggregate['samples'] += 1
                aggregate['matchable'] += int(matchable)
                aggregate['boundary'] += int(sample['boundary_peak'])
                aggregate['low_conf'] += int(tier == TIER_IDS['tier3_low_conf'])
                aggregate['large_error'] += int(tier == TIER_IDS['tier3_large_error'])

    arrays = {
        'sequence_index': np.asarray(records['sequence_index'], dtype=np.int32),
        'frame_id': np.asarray(records['frame_id'], dtype=np.int16),
        'template_frame_id': np.asarray(records['template_frame_id'], dtype=np.int16),
        'template_center': np.asarray(records['template_center'], dtype=np.float32),
        'search_center': np.asarray(records['search_center'], dtype=np.float32),
        'ncc_coord': np.asarray(records['ncc_coord'], dtype=np.float32),
        'ground_truth': np.asarray(records['ground_truth'], dtype=np.float32),
        'target_cell': np.asarray(records['target_cell'], dtype=np.float32),
        'confidence': np.asarray(records['confidence'], dtype=np.float32),
        'epsilon': np.asarray(records['epsilon'], dtype=np.float32),
        'boundary_peak': np.asarray(records['boundary_peak'], dtype=np.bool_),
        'matchable': np.asarray(records['matchable'], dtype=np.bool_),
        'tier': np.asarray(records['tier'], dtype=np.int8),
        'template_age': np.asarray(records['template_age'], dtype=np.int16),
        'kalman_error_px': np.asarray(records['kalman_error_px'], dtype=np.float32),
        'ncc_error_px': np.asarray(records['ncc_error_px'], dtype=np.float32),
    }
    metadata = {
        'format_version': 1,
        'source': os.path.abspath(args.train_h5_path),
        'split': args.split,
        'seed': args.seed,
        'val_ratio': args.val_ratio,
        'selected_indices': selected_indices,
        'stnet_ckpt': os.path.abspath(args.stnet_ckpt),
        'ncc_ckpt': os.path.abspath(args.ncc_ckpt),
        'tracker_config': {
            'tau1': args.tau1, 'tau2': args.tau2, 'tau3': args.tau3,
            'ncc_thresh': args.ncc_thresh, 'template_size': args.template_size,
            'search_size': args.search_size, 'window_size': args.window_size,
            'hard_reset': not args.soft_fusion,
            'reset_velocity_mode': args.reset_velocity_mode,
            'low_conf_policy': args.low_conf_policy,
            'kalman_process_noise': args.kalman_process_noise,
            'kalman_measurement_noise': args.kalman_measurement_noise,
            'kalman_velocity_decay': args.kalman_velocity_decay,
        },
        'aggregate': aggregate,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, metadata_json=np.asarray(json.dumps(metadata)), **arrays)

    sample_count = max(aggregate['samples'], 1)
    print(f'saved rollout cache: {output_path}')
    print(f"samples={aggregate['samples']} matchable={aggregate['matchable']/sample_count:.1%} "
          f"boundary={aggregate['boundary']/sample_count:.1%} "
          f"low_conf={aggregate['low_conf']/sample_count:.1%} "
          f"large_error={aggregate['large_error']/sample_count:.1%}")


if __name__ == '__main__':
    main()
