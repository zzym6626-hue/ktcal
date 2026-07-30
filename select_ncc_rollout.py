"""Select rollout-trained NCC checkpoints with full closed-loop validation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter
from tqdm import tqdm

import provider_data
from ktcal_tracker import KTCalTracker
from models.STNet import STNet
from models.ncc_matcher import NCCMatcher
from train_ktcal import load_model_weights, seed_everything, split_train_val


def parse_args():
    parser = argparse.ArgumentParser('Closed-loop selection for rollout NCC checkpoints')
    parser.add_argument('--train_h5_path', default='./data/seet/train_3et.h5')
    parser.add_argument('--stnet_ckpt', required=True)
    parser.add_argument('--stnet_block3_kernel', type=int, choices=[5, 7], default=5)
    parser.add_argument('--stnet_unidirectional', action='store_true', default=False)
    parser.add_argument('--stnet_decode', choices=['soft', 'argmax'], default='soft')
    parser.add_argument('--ncc_ckpt', default=None,
                        help='Evaluate one fixed NCC checkpoint directly')
    parser.add_argument('--checkpoint_dir', default=None,
                        help='Directory containing rollout NCC epoch checkpoints')
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--output_ckpt', default=None)
    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--tau1', type=float, default=3.0)
    parser.add_argument('--tau2', type=float, default=5.0)
    parser.add_argument('--tau3', type=float, default=1.0)
    parser.add_argument('--ncc_thresh', type=float, default=0.5)
    parser.add_argument('--stnet_cooldown', type=int, default=0)
    parser.add_argument('--disable_tier2', action='store_true', default=False)
    parser.add_argument('--severe_conf_threshold', type=float, default=0.2)
    parser.add_argument('--severe_epsilon_threshold', type=float, default=9.0)
    parser.add_argument('--cooldown_low_conf_mode',
                        choices=['hold', 'ncc', 'anchor'], default='ncc')
    parser.add_argument('--trigger_classifier', type=str, default=None)
    parser.add_argument('--trigger_learn_threshold', type=float, default=0.5)
    parser.add_argument('--reset_velocity_mode',
                        choices=['legacy', 'previous_position', 'zero', 'preserve'],
                        default='legacy')
    parser.add_argument('--low_conf_policy',
                        choices=['sync_stnet', 'kalman', 'async_stnet', 'hold', 'hold_async'],
                        default='sync_stnet')
    parser.add_argument('--kalman_process_noise', type=float, default=1e-3)
    parser.add_argument('--kalman_measurement_noise', type=float, default=1e-2)
    parser.add_argument('--kalman_velocity_decay', type=float, default=1.0)
    parser.add_argument('--pp_window', type=int, default=15)
    parser.add_argument('--pp_poly', type=int, default=3)
    return parser.parse_args()


def load_trigger_classifier(path):
    if not path:
        return None
    cache = np.load(path)
    return {
        'weights': cache['weights'].astype(np.float64),
        'bias': float(cache['bias']),
        'mean': cache['mean'].astype(np.float64),
        'std': cache['std'].astype(np.float64),
        'epsilon_fill': float(cache['epsilon_fill']),
    }


@torch.no_grad()
def evaluate_checkpoint(path, stnet, frames, labels, args, device):
    matcher = NCCMatcher().to(device)
    load_model_weights(matcher, path, device)
    matcher.eval()
    trigger_classifier = load_trigger_classifier(args.trigger_classifier)
    tracker = KTCalTracker(
        stnet, matcher, tau1=args.tau1, tau2=args.tau2, tau3=args.tau3,
        ncc_thresh=args.ncc_thresh, template_size=11, search_size=21,
        window_size=16, hard_reset=True, device=device,
        reset_velocity_mode=args.reset_velocity_mode,
        low_conf_policy=args.low_conf_policy,
        kalman_process_noise=args.kalman_process_noise,
        kalman_measurement_noise=args.kalman_measurement_noise,
        kalman_velocity_decay=args.kalman_velocity_decay,
        stnet_cooldown=args.stnet_cooldown,
        disable_tier2=args.disable_tier2,
        severe_conf_threshold=args.severe_conf_threshold,
        severe_epsilon_threshold=args.severe_epsilon_threshold,
        cooldown_low_conf_mode=args.cooldown_low_conf_mode,
        trigger_classifier=trigger_classifier,
        trigger_learn_threshold=args.trigger_learn_threshold,
    )
    pixel_scale = np.array([80.0, 60.0])
    total_error = total_online_error = total_pp_error = 0.0
    correct_pp = {3: 0, 5: 0}
    total_frames = stnet_calls = tier3 = boundary = resets = 0
    cooldown_skips = severe_bypasses = tier2_skips = 0
    for sequence_index in tqdm(range(len(frames)), desc=Path(path).stem, leave=False):
        coords, stats = tracker.track_sequence(frames[sequence_index])
        target = labels[sequence_index, :, :2].numpy()
        online = stats['online_coords']
        coords_pp = coords.copy()
        if len(coords_pp) >= args.pp_window:
            coords_pp[:, 0] = savgol_filter(coords[:, 0], args.pp_window, args.pp_poly)
            coords_pp[:, 1] = savgol_filter(coords[:, 1], args.pp_window, args.pp_poly)
        total_error += np.linalg.norm((coords - target) * pixel_scale, axis=-1).sum()
        total_online_error += np.linalg.norm((online - target) * pixel_scale, axis=-1).sum()
        dist_pp = np.linalg.norm((coords_pp - target) * pixel_scale, axis=-1)
        total_pp_error += dist_pp.sum()
        for tolerance in correct_pp:
            correct_pp[tolerance] += int((dist_pp < tolerance).sum())
        total_frames += len(target)
        stnet_calls += stats['stnet_calls']
        tier3 += stats['tier3']
        boundary += stats['boundary_peaks']
        resets += stats['resets']
        cooldown_skips += stats['cooldown_skips']
        severe_bypasses += stats['severe_cooldown_bypasses']
        tier2_skips += stats['tier2_skips']
    return {
        'checkpoint': str(path),
        'online_error_px': total_online_error / total_frames,
        'final_error_px': total_error / total_frames,
        'post_processed_error_px': total_pp_error / total_frames,
        'post_processed_p3': correct_pp[3] / total_frames,
        'post_processed_p5': correct_pp[5] / total_frames,
        'trigger_ratio': stnet_calls / total_frames,
        'tier3_ratio': tier3 / total_frames,
        'boundary_rate': boundary / total_frames,
        'reset_ratio': resets / total_frames,
        'cooldown_skip_ratio': cooldown_skips / total_frames,
        'severe_bypass_ratio': severe_bypasses / total_frames,
        'tier2_skip_ratio': tier2_skips / total_frames,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cpu' if args.use_cpu else 'cuda')
    frame_array, label_array = provider_data.load_h5_f(args.train_h5_path)
    frames = torch.from_numpy(np.asarray(frame_array)).float()
    labels = torch.from_numpy(np.asarray(label_array)).float()
    (_, _, val_frames, val_labels, _, _) = split_train_val(
        frames, labels, args.val_ratio, args.seed)

    stnet = STNet(args).to(device)
    load_model_weights(stnet, args.stnet_ckpt, device)
    stnet.eval()
    if args.ncc_ckpt:
        result = evaluate_checkpoint(
            args.ncc_ckpt, stnet, val_frames, val_labels, args, device)
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open('w', encoding='utf-8') as handle:
            json.dump({'config': vars(args), 'result': result}, handle, indent=2)
        print(json.dumps(result, indent=2))
        return
    if not args.checkpoint_dir:
        raise ValueError('Provide --ncc_ckpt or --checkpoint_dir')
    checkpoint_dir = Path(args.checkpoint_dir)
    epoch_paths = sorted((checkpoint_dir / 'epochs').glob('epoch_*.pth'))
    if not epoch_paths:
        raise ValueError('No epoch checkpoints found')
    ranked = []
    for path in epoch_paths:
        payload = torch.load(path, map_location='cpu')
        score = payload.get('best_metrics', {}).get('selection_score', float('inf'))
        ranked.append((score, path))
    baseline_path = next((path for _, path in ranked if path.stem == 'epoch_000'), None)
    candidate_paths = [path for _, path in sorted(ranked)[:args.top_k]]
    if baseline_path is not None and baseline_path not in candidate_paths:
        candidate_paths.append(baseline_path)

    results = []
    for path in candidate_paths:
        results.append(evaluate_checkpoint(path, stnet, val_frames, val_labels, args, device))
    baseline = next(result for result in results if Path(result['checkpoint']).stem == 'epoch_000')
    eligible = [
        result for result in results
        if result['trigger_ratio'] <= baseline['trigger_ratio']
        and result['boundary_rate'] <= baseline['boundary_rate']
    ]
    selected = min(
        eligible or [baseline],
        key=lambda item: (item['final_error_px'], item['trigger_ratio']))
    selected_payload = torch.load(selected['checkpoint'], map_location='cpu')
    selected_payload['closed_loop_validation'] = selected
    selected_payload['closed_loop_baseline'] = baseline
    if args.output_ckpt:
        output_ckpt = Path(args.output_ckpt)
        output_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(selected_payload, output_ckpt)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open('w', encoding='utf-8') as handle:
        json.dump({
            'config': vars(args), 'baseline': baseline,
            'selected': selected, 'candidates': results,
        }, handle, indent=2)
    print(f'selected checkpoint: {selected["checkpoint"]}')
    print(json.dumps(selected, indent=2))


if __name__ == '__main__':
    main()
