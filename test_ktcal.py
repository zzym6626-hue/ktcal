"""
KTCal closed-loop evaluation (KTCal paper, Fig 1).

Runs the predict-and-correct tracker over each test sequence and reports:
  - pupil localization accuracy (p1/p3/p5/... within pixel tolerances),
  - mean pixel Euclidean error and mean squared pixel error,
  - tier-trigger statistics, accounted component latency, and actual sequential
    wall-clock latency. The STNet trigger ratio is also reported against the
    baseline that runs STNet on every frame.

Usage:
    python3 test_ktcal.py --test_h5_path ./data/seet/test_3et.h5 \
            --stnet_ckpt ./checkpoint_ktcal/stnet_best.pth \
            --ncc_ckpt   ./checkpoint_ktcal/ncc_best.pth --use_cpu
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.signal import savgol_filter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

import provider_data
from metrics import decode_soft_expectation
from models.STNet import STNet
from models.ncc_matcher import NCCMatcher
from ktcal_tracker import KTCalTracker


def parse_args():
    p = argparse.ArgumentParser('KTCal evaluation')
    p.add_argument('--use_cpu', action='store_true', default=False)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--test_h5_path', type=str, default='./data/seet/test_3et.h5')
    p.add_argument('--stnet_ckpt', type=str, default='./checkpoint_ktcal/stnet_best.pth')
    p.add_argument('--stnet_block3_kernel', type=int, choices=[5, 7], default=5)
    p.add_argument('--stnet_unidirectional', action='store_true', default=False,
                   help='Load a unidirectional STNet variant')
    p.add_argument('--stnet_decode', choices=['soft', 'argmax'], default='soft',
                   help='Decode STNet coordinate heatmaps by expectation or peak location')
    p.add_argument('--ncc_ckpt', type=str, default='./checkpoint_ktcal/ncc_best.pth')
    p.add_argument('--sensor_width', type=int, default=640)
    p.add_argument('--sensor_height', type=int, default=480)
    p.add_argument('--spatial_factor', type=float, default=0.125)
    p.add_argument('--pixel_tolerances', default=[1, 3, 5, 10, 15], type=int)
    p.add_argument('--tau1', type=float, default=3.0)
    p.add_argument('--tau2', type=float, default=5.0)
    p.add_argument('--tau3', type=float, default=1.0)
    p.add_argument('--hard_reset', action='store_true', default=False,
                   help='Use tau3-triggered hard Kalman reset instead of trusted fusion')
    p.add_argument('--reset_velocity_mode',
                   choices=['legacy', 'previous_position', 'zero', 'preserve'],
                   default='legacy', help='Velocity policy applied during hard reset')
    p.add_argument('--ncc_thresh', type=float, default=0.5)
    p.add_argument('--low_conf_policy',
                   choices=['sync_stnet', 'kalman', 'async_stnet', 'hold', 'hold_async'],
                   default='sync_stnet')
    p.add_argument('--kalman_process_noise', type=float, default=1e-3)
    p.add_argument('--kalman_measurement_noise', type=float, default=1e-2)
    p.add_argument('--kalman_velocity_decay', type=float, default=1.0)
    p.add_argument('--stnet_cooldown', type=int, default=0)
    p.add_argument('--disable_tier2', action='store_true', default=False)
    p.add_argument('--severe_conf_threshold', type=float, default=0.2)
    p.add_argument('--severe_epsilon_threshold', type=float, default=9.0)
    p.add_argument('--cooldown_low_conf_mode',
                   choices=['hold', 'ncc', 'anchor'], default='ncc')
    p.add_argument('--trigger_classifier', type=str, default=None,
                   help='Optional learned trigger classifier (npz) that can bypass cooldown')
    p.add_argument('--trigger_learn_threshold', type=float, default=0.5,
                   help='Probability threshold above which the learned trigger fires')
    p.add_argument('--template_size', type=int, default=11,
                   help='Odd NCC template side length')
    p.add_argument('--search_size', type=int, default=21,
                   help='Odd NCC search-window side length')
    p.add_argument('--ncc_center_penalty', type=float, default=0.0,
                   help='Penalty per normalized squared distance from the NCC center')
    p.add_argument('--ncc_boundary_penalty', type=float, default=0.0,
                   help='Additional score penalty on the NCC response boundary')
    p.add_argument('--pp_window', type=int, default=15, help='Savitzky-Golay window')
    p.add_argument('--pp_poly', type=int, default=3, help='Savitzky-Golay poly order')
    p.add_argument('--batch_mode', action='store_true', default=False,
                   help='TDTracker-style offline batch: STNet.forward outputs all frames '
                        'in one pass (no closed loop). No post-processing applied.')
    p.add_argument('--output_json', type=str, default=None,
                   help='Optional path for structured evaluation results')
    return p.parse_args()


def load_model_checkpoint(model, path, device, name):
    if not os.path.exists(path):
        print(f'[warn] {name} checkpoint not found ({path}); using random init')
        return {}
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        metadata = {k: checkpoint[k] for k in ('format_version', 'epoch', 'seed',
                                                'best_val_mean_pixel_error')
                    if k in checkpoint}
    else:
        model.load_state_dict(checkpoint, strict=False)
        metadata = {'format_version': 0, 'legacy_weights_only': True}
    print(f'loaded {name} weights: {path}')
    return metadata


def save_results(path, results):
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'results saved: {output_path}')


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


def synchronize_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    if args.template_size <= 0 or args.template_size % 2 == 0:
        raise ValueError('--template_size must be a positive odd integer')
    if args.search_size <= 0 or args.search_size % 2 == 0:
        raise ValueError('--search_size must be a positive odd integer')
    if args.search_size < args.template_size:
        raise ValueError('--search_size must be greater than or equal to --template_size')
    if args.ncc_center_penalty < 0.0 or args.ncc_boundary_penalty < 0.0:
        raise ValueError('NCC penalties must be non-negative')
    device = torch.device('cpu' if args.use_cpu else 'cuda')

    F_test, L_test = provider_data.load_h5_f(args.test_h5_path)
    F_test = torch.from_numpy(np.array(F_test)).float()
    L_test = torch.from_numpy(np.array(L_test)).float()
    print('data=', F_test.shape)

    stnet = STNet(args).to(device)
    matcher = NCCMatcher().to(device)
    stnet_metadata = load_model_checkpoint(stnet, args.stnet_ckpt, device, 'STNet')
    ncc_metadata = load_model_checkpoint(matcher, args.ncc_ckpt, device, 'NCC')
    stnet.eval()
    matcher.eval()
    trigger_classifier = load_trigger_classifier(args.trigger_classifier)
    if trigger_classifier is not None:
        print(f'loaded trigger classifier: {args.trigger_classifier} '
              f'(threshold={args.trigger_learn_threshold})')

    tracker = KTCalTracker(stnet, matcher,
                           tau1=args.tau1, tau2=args.tau2, tau3=args.tau3,
                           ncc_thresh=args.ncc_thresh,
                           template_size=args.template_size,
                           search_size=args.search_size,
                           ncc_center_penalty=args.ncc_center_penalty,
                           ncc_boundary_penalty=args.ncc_boundary_penalty,
                           hard_reset=args.hard_reset, device=device,
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
                           trigger_learn_threshold=args.trigger_learn_threshold)

    w_scale = args.sensor_width * args.spatial_factor   # 80
    h_scale = args.sensor_height * args.spatial_factor  # 60

    # ---- TDTracker-style offline batch mode: one STNet.forward per sequence ----
    if args.batch_mode:
        import time as _time
        total_dist_b, total_squared_dist_b, total_frames_b = 0.0, 0.0, 0
        correct_b = {p: 0 for p in args.pixel_tolerances}
        t_total = 0.0
        with torch.no_grad():
            for i in tqdm(range(F_test.shape[0]), desc='STNet batch'):
                seq = F_test[i].unsqueeze(0).to(device)   # (1, S, 2, 60, 80)
                labels = L_test[i][:, :2].numpy()
                synchronize_device(device)
                t0 = _time.perf_counter()
                pw, ph = stnet(seq)                        # (1,S,80),(1,S,60)
                if args.stnet_decode == 'argmax':
                    coords = torch.stack((pw.argmax(-1).to(pw.dtype) / 80.0,
                                          ph.argmax(-1).to(ph.dtype) / 60.0), dim=-1)[0]
                else:
                    coords = decode_soft_expectation(pw, ph)[0]
                synchronize_device(device)
                t_total += _time.perf_counter() - t0
                coords = coords.cpu().numpy()
                dist = np.linalg.norm((coords - labels) * np.array([w_scale, h_scale]), axis=-1)
                total_dist_b += dist.sum()
                total_squared_dist_b += np.square(dist).sum()
                total_frames_b += dist.shape[0]
                for p in args.pixel_tolerances:
                    correct_b[p] += int((dist < p).sum())
        accuracies_b = {f'p{p}': correct_b[p] / total_frames_b for p in args.pixel_tolerances}
        mean_error_b = total_dist_b / total_frames_b
        squared_error_b = total_squared_dist_b / total_frames_b
        latency_b = t_total / total_frames_b * 1000
        print('\n===== STNet BATCH mode (TDTracker-style, no closed loop) =====')
        for name, value in accuracies_b.items():
            print(f'  {name} accuracy: {value:.4f}')
        print(f'  mean pixel error         : {mean_error_b:.4f} px')
        print(f'  mean squared pixel error : {squared_error_b:.4f} px^2')
        print(f'  avg latency              : {latency_b:.4f} ms/frame '
              f'(batch forward total / frames)')
        save_results(args.output_json, {
            'mode': 'offline_stnet_batch',
            'test_h5_path': os.path.abspath(args.test_h5_path),
            'frames': total_frames_b,
            'accuracy': accuracies_b,
            'mean_pixel_error_px': mean_error_b,
            'mean_squared_pixel_error_px2': squared_error_b,
            'average_model_latency_ms_per_frame': latency_b,
            'checkpoint_metadata': {'stnet': stnet_metadata, 'ncc': ncc_metadata},
            'config': vars(args),
        })
        return


    total_dist = 0.0
    total_squared_dist = 0.0
    total_online_dist = 0.0
    total_online_squared_dist = 0.0
    total_dist_pp = 0.0
    total_squared_dist_pp = 0.0
    total_frames = 0
    correct = {p: 0 for p in args.pixel_tolerances}
    correct_online = {p: 0 for p in args.pixel_tolerances}
    correct_pp = {p: 0 for p in args.pixel_tolerances}
    agg = {
        'tier1': 0, 'tier2': 0, 'tier3': 0,
        'tier3_bootstrap': 0, 'tier3_low_conf': 0,
        'tier3_large_error': 0, 'boundary_peaks': 0,
        'low_conf_boundary_peaks': 0, 'large_error_boundary_peaks': 0,
        'cooldown_skips': 0, 'low_conf_cooldown_skips': 0,
        'large_error_cooldown_skips': 0, 'tier2_skips': 0,
        'severe_cooldown_bypasses': 0,
        'stnet_calls': 0, 'resets': 0,
    }
    confidence_bins = {
        'lt_0.3': 0, '0.3_to_0.4': 0, '0.4_to_0.5': 0,
        '0.5_to_0.6': 0, 'ge_0.6': 0,
    }
    epsilon_bins = {
        'lt_3': 0, '3_to_5': 0, '5_to_7': 0,
        '7_to_10': 0, 'ge_10': 0,
    }
    cause_quality = {
        'low_conf': {'count': 0, 'prediction_error': 0.0,
                     'ncc_error': 0.0, 'ncc_better': 0},
        'large_error': {'count': 0, 'prediction_error': 0.0,
                        'ncc_error': 0.0, 'ncc_better': 0},
    }
    critical_total_ms = 0.0
    async_total_ms = 0.0
    sync_stnet_total_ms = 0.0
    wall_total_ms = 0.0
    lag_frames_weighted = 0.0
    lag_ms_weighted = 0.0
    async_results = 0

    n_seq = F_test.shape[0]
    for i in tqdm(range(n_seq), desc='KTCal tracking'):
        frames = F_test[i]           # (S, 2, 60, 80)
        labels = L_test[i][:, :2].numpy()  # (S, 2) normalized
        coords, stats = tracker.track_sequence(frames)
        online_coords = stats['online_coords']
        pixel_scale = np.array([w_scale, h_scale])
        for record in stats['tier3_records']:
            gt = labels[record['frame_id']]
            prediction_error = np.linalg.norm((record['prediction'] - gt) * pixel_scale)
            ncc_error = np.linalg.norm((record['ncc'] - gt) * pixel_scale)
            quality = cause_quality[record['cause']]
            quality['count'] += 1
            quality['prediction_error'] += prediction_error
            quality['ncc_error'] += ncc_error
            quality['ncc_better'] += int(ncc_error < prediction_error)

        diff = (coords - labels) * pixel_scale
        dist = np.linalg.norm(diff, axis=-1)   # (S,) pixel error
        total_dist += dist.sum()
        total_squared_dist += np.square(dist).sum()
        total_frames += dist.shape[0]
        for p in args.pixel_tolerances:
            correct[p] += int((dist < p).sum())

        online_dist = np.linalg.norm(
            (online_coords - labels) * np.array([w_scale, h_scale]), axis=-1
        )
        total_online_dist += online_dist.sum()
        total_online_squared_dist += np.square(online_dist).sum()
        for p in args.pixel_tolerances:
            correct_online[p] += int((online_dist < p).sum())

        # Post-processing: Savitzky-Golay trajectory smoothing (like TDTracker)
        if coords.shape[0] >= args.pp_window:
            coords_pp = coords.copy()
            coords_pp[:, 0] = savgol_filter(coords[:, 0], args.pp_window, args.pp_poly)
            coords_pp[:, 1] = savgol_filter(coords[:, 1], args.pp_window, args.pp_poly)
        else:
            coords_pp = coords
        dist_pp = np.linalg.norm((coords_pp - labels) * np.array([w_scale, h_scale]), axis=-1)
        total_dist_pp += dist_pp.sum()
        total_squared_dist_pp += np.square(dist_pp).sum()
        for p in args.pixel_tolerances:
            correct_pp[p] += int((dist_pp < p).sum())

        for key in agg:
            agg[key] += stats[key]
        for key in confidence_bins:
            confidence_bins[key] += stats['confidence_bins'][key]
        for key in epsilon_bins:
            epsilon_bins[key] += stats['epsilon_bins'][key]
        critical_total_ms += stats['critical_ms']
        async_total_ms += stats['async_ms']
        sync_stnet_total_ms += stats['sync_stnet_ms']
        wall_total_ms += stats['wall_ms']
        result_count = stats['async_results']
        async_results += result_count
        lag_frames_weighted += stats['avg_delivery_lag_frames'] * result_count
        lag_ms_weighted += stats['avg_delivery_lag_ms'] * result_count

    accuracy = {f'p{p}': correct[p] / total_frames for p in args.pixel_tolerances}
    accuracy_online = {
        f'p{p}': correct_online[p] / total_frames
        for p in args.pixel_tolerances
    }
    accuracy_pp = {f'p{p}': correct_pp[p] / total_frames for p in args.pixel_tolerances}
    mean_error = total_dist / total_frames
    mean_squared_error = total_squared_dist / total_frames
    mean_online_error = total_online_dist / total_frames
    mean_online_squared_error = total_online_squared_dist / total_frames
    mean_error_pp = total_dist_pp / total_frames
    mean_squared_error_pp = total_squared_dist_pp / total_frames
    print('\n===== KTCal Immediate Online Output =====')
    for name, value in accuracy_online.items():
        print(f'  {name} accuracy: {value:.4f}')
    print(f'  mean pixel error         : {mean_online_error:.4f} px')
    print(f'  mean squared pixel error : {mean_online_squared_error:.4f} px^2')
    print('\n===== KTCal Final Closed-Loop Corrected Output =====')
    for name, value in accuracy.items():
        print(f'  {name} accuracy: {value:.4f}')
    print(f'  mean pixel error         : {mean_error:.4f} px')
    print(f'  mean squared pixel error : {mean_squared_error:.4f} px^2')
    print(f'\n----- After post-processing (Savitzky-Golay w={args.pp_window},poly={args.pp_poly}) -----')
    for name, value in accuracy_pp.items():
        print(f'  {name} accuracy: {value:.4f}')
    print(f'  mean pixel error         : {mean_error_pp:.4f} px')
    print(f'  mean squared pixel error : {mean_squared_error_pp:.4f} px^2')
    print('\n----- Scheduling / Latency (on-demand STNet) -----')
    total_tier = agg['tier1'] + agg['tier2'] + agg['tier3']
    tier_ratios = {
        'tier1': agg['tier1'] / max(total_tier, 1),
        'tier2': agg['tier2'] / max(total_tier, 1),
        'tier3': agg['tier3'] / max(total_tier, 1),
    }
    if total_tier > 0:
        print(f"  tier1 (Kalman only) : {agg['tier1']} ({tier_ratios['tier1']*100:.1f}%)")
        print(f"  tier2 (async STNet) : {agg['tier2']} ({tier_ratios['tier2']*100:.1f}%)")
        print(f"  tier3 (forced STNet): {agg['tier3']} ({tier_ratios['tier3']*100:.1f}%)")
    print('\n----- Tier3 Cause Diagnostics -----')
    print(f"  bootstrap             : {agg['tier3_bootstrap']}")
    print(f"  NCC confidence < 0.5  : {agg['tier3_low_conf']} ")
    print(f"  epsilon >= 5px        : {agg['tier3_large_error']}")
    print(f"  all NCC boundary peaks: {agg['boundary_peaks']}")
    print(f"  low-conf at boundary  : {agg['low_conf_boundary_peaks']}")
    print(f"  large-error boundary  : {agg['large_error_boundary_peaks']}")
    print(f"  cooldown skips        : {agg['cooldown_skips']} "
          f"(low-conf={agg['low_conf_cooldown_skips']}, "
          f"large-error={agg['large_error_cooldown_skips']}, "
          f"tier2={agg['tier2_skips']})")
    print(f"  severe bypasses       : {agg['severe_cooldown_bypasses']}")
    print(f"  confidence bins       : {confidence_bins}")
    print(f"  epsilon bins (conf ok): {epsilon_bins}")
    recover_conf_04 = confidence_bins['0.4_to_0.5']
    recover_epsilon_7 = epsilon_bins['5_to_7']
    print(f"  threshold-only candidates: conf 0.4-0.5={recover_conf_04}, "
          f"epsilon 5-7px={recover_epsilon_7}")
    print('  ground-truth diagnosis:')
    cause_quality_summary = {}
    for cause, quality in cause_quality.items():
        count = max(quality['count'], 1)
        cause_quality_summary[cause] = {
            'count': quality['count'],
            'mean_kalman_error_px': float(quality['prediction_error'] / count),
            'mean_ncc_error_px': float(quality['ncc_error'] / count),
            'ncc_better_ratio': quality['ncc_better'] / count,
        }
        summary = cause_quality_summary[cause]
        print(f"    {cause}: Kalman={summary['mean_kalman_error_px']:.2f}px, "
              f"NCC={summary['mean_ncc_error_px']:.2f}px, "
              f"NCC better={summary['ncc_better_ratio']*100:.1f}%")
    stnet_ratio = agg['stnet_calls'] / total_frames
    print(f"  STNet calls total   : {agg['stnet_calls']} / {total_frames} frames "
          f"({stnet_ratio*100:.1f}%)")
    print(f"  Kalman resets       : {agg['resets']}")
    crit = critical_total_ms / total_frames
    asyncl = async_total_ms / total_frames
    sync_stnet_ms = sync_stnet_total_ms / total_frames
    wall_ms = wall_total_ms / total_frames
    lag_frames = lag_frames_weighted / max(async_results, 1)
    lag_ms = lag_ms_weighted / max(async_results, 1)
    print(f"  online fast-path latency    : {crit:.4f} ms/frame")
    print(f"  async STNet compute work    : {asyncl:.4f} ms/frame (tier2 worker share)")
    print(f"  sync STNet critical work    : {sync_stnet_ms:.4f} ms/frame (tier3)")
    print(f"  concurrent end-to-end wall  : {wall_ms:.4f} ms/frame (includes final flush)")
    print(f"  correction delivery lag     : {lag_frames:.2f} frames / {lag_ms:.2f} ms")
    print(f"  (final accuracy is available after asynchronous correction and replay)")
    save_results(args.output_json, {
        'mode': 'ktcal_closed_loop_async',
        'test_h5_path': os.path.abspath(args.test_h5_path),
        'frames': total_frames,
        'immediate_online': {
            'accuracy': accuracy_online,
            'mean_pixel_error_px': mean_online_error,
            'mean_squared_pixel_error_px2': mean_online_squared_error,
        },
        'final_closed_loop_corrected': {
            'accuracy': accuracy,
            'mean_pixel_error_px': mean_error,
            'mean_squared_pixel_error_px2': mean_squared_error,
        },
        'post_processed': {
            'method': 'savgol',
            'window': args.pp_window,
            'poly_order': args.pp_poly,
            'accuracy': accuracy_pp,
            'mean_pixel_error_px': mean_error_pp,
            'mean_squared_pixel_error_px2': mean_squared_error_pp,
        },
        'scheduling': {
            'counts': agg,
            'tier_ratios': tier_ratios,
            'stnet_ratio': stnet_ratio,
            'confidence_bins': confidence_bins,
            'epsilon_bins': epsilon_bins,
            'threshold_only_candidates': {
                'confidence_0.4_to_0.5': recover_conf_04,
                'epsilon_5_to_7': recover_epsilon_7,
            },
            'ground_truth_diagnosis': cause_quality_summary,
        },
        'latency_ms_per_frame': {
            'online_fast_path': crit,
            'async_stnet_compute_work': asyncl,
            'sync_stnet_critical_work': sync_stnet_ms,
            'concurrent_end_to_end_wall': wall_ms,
        },
        'correction_delivery_lag': {
            'frames': lag_frames,
            'milliseconds': lag_ms,
            'results': async_results,
        },
        'checkpoint_metadata': {'stnet': stnet_metadata, 'ncc': ncc_metadata},
        'config': vars(args),
    })


if __name__ == '__main__':
    main()
