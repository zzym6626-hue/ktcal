"""
KTCal training (KTCal paper Sec 3 / review reply W4).

Joint end-to-end objective:

    L_total = L_ST + lambda * L_match ,   lambda = 0.3

  - L_ST    : STNet SA-SimDR heatmap loss (reuses KLDiscretLoss from metrics.py).
  - L_match : differentiable NCC template-matching loss. A template is cropped
              around the ground-truth pupil in an earlier frame and correlated
              against a jittered local search window in the current frame; the
              Soft-Argmax peak (T=0.1) is supervised by the current pupil location.

Both branches share supervision from ground-truth pupil coordinates only
(no extra matching labels), matching the paper's description.

Usage:
    python3 train_ktcal.py --train_h5_path ./data/seet/train_3et.h5 \
                           --val_ratio 0.1 --seed 42 --use_cpu
"""
import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))

import provider_data
from metrics import KLDiscretLoss, px_euclidean_dist, decode_soft_expectation
from models.STNet import STNet
from models.ncc_matcher import NCCMatcher, hard_argmax_2d, soft_argmax_2d
from ktcal_tracker import KTCalTracker

TEMPLATE_SIZE = 11
SEARCH_SIZE = 21
LAMBDA_MATCH = 0.3


def parse_args():
    p = argparse.ArgumentParser('KTCal training')
    p.add_argument('--use_cpu', action='store_true', default=False)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--train_h5_path', type=str, default='./data/seet/train_3et.h5')
    p.add_argument('--test_h5_path', type=str, default=None,
                   help='Deprecated and ignored: the test set is never used during training')
    p.add_argument('--save_path', type=str, default='./checkpoint_ktcal')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--val_ratio', type=float, default=0.1,
                   help='Fraction of train sequences reserved for validation')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_train_batches', type=int, default=0,
                   help='Limit train batches per epoch for smoke tests; 0 uses all batches')
    p.add_argument('--max_val_batches', type=int, default=0,
                   help='Limit validation batches for smoke tests; 0 uses all batches')
    p.add_argument('--no_save', action='store_true', default=False,
                   help='Do not write checkpoints or run metadata')
    p.add_argument('--sensor_width', type=int, default=640)
    p.add_argument('--sensor_height', type=int, default=480)
    p.add_argument('--spatial_factor', type=float, default=0.125)
    p.add_argument('--pixel_tolerances', default=[1, 3, 5, 10, 15], type=int)
    p.add_argument('--epoch', default=200, type=int)
    p.add_argument('--patience', default=20, type=int, help='early-stopping patience (epochs)')
    p.add_argument('--min_delta', default=0.01, type=float,
                   help='Minimum validation mean-pixel-error improvement to reset patience')
    p.add_argument('--learning_rate', default=0.002, type=float)
    p.add_argument('--frames_per_seq', default=8, type=int,
                   help='frames sampled per sequence for the NCC match loss')
    p.add_argument('--ncc_only', action='store_true', default=False,
                   help='Freeze/skip STNet and select checkpoints using NCC validation error')
    p.add_argument('--ncc_init_ckpt', type=str, default=None,
                   help='Optional NCC checkpoint used to initialize NCC-only training')
    p.add_argument('--hard_negative_margin', type=float, default=0.15)
    p.add_argument('--boundary_margin', type=float, default=0.10)
    p.add_argument('--hard_negative_weight', type=float, default=0.0)
    p.add_argument('--boundary_weight', type=float, default=0.0)
    p.add_argument('--peak_ce_weight', type=float, default=0.0)
    p.add_argument('--positive_radius', type=int, default=1)
    p.add_argument('--ncc_closed_loop_val', action='store_true', default=False,
                   help='Select NCC checkpoints using closed-loop validation sequences')
    p.add_argument('--stnet_val_ckpt', type=str, default=None,
                   help='STNet checkpoint required by --ncc_closed_loop_val')
    p.add_argument('--closed_val_sequences', type=int, default=8)
    p.add_argument('--closed_val_trigger_weight', type=float, default=1.0,
                   help='Pixels added to selection score per unit STNet trigger ratio')
    p.add_argument('--closed_val_hard_reset', action='store_true', default=False)
    p.add_argument('--closed_val_tau3', type=float, default=1.0)
    p.add_argument('--stnet_only', action='store_true', default=False,
                   help='Train STNet without updating NCC')
    p.add_argument('--stnet_init_ckpt', type=str, default=None)
    p.add_argument('--short_window_sizes', type=str, default='',
                   help='Comma-separated STNet training windows, e.g. 4,8,16')
    p.add_argument('--full_sequence_weight', type=float, default=1.0)
    p.add_argument('--full_sequence_probability', type=float, default=1.0)
    p.add_argument('--short_window_weight', type=float, default=1.0)
    p.add_argument('--short_val_windows_per_seq', type=int, default=4)
    p.add_argument('--stnet_last_step_only', action='store_true', default=False,
                   help='Supervise only the final output of each STNet window')
    p.add_argument('--stnet_coord_loss_weight', type=float, default=0.0,
                   help='Weight for normalized coordinate L2 loss in STNet training')
    p.add_argument('--stnet_teacher_ckpt', type=str, default=None,
                   help='Frozen full-sequence STNet teacher used for short-window distillation')
    p.add_argument('--stnet_distill_weight', type=float, default=0.0)
    p.add_argument('--stnet_distill_temperature', type=float, default=2.0)
    p.add_argument('--stnet_pixel_huber_weight', type=float, default=0.0,
                   help='Weight for last-step pixel Smooth-L1 refinement')
    p.add_argument('--stnet_pixel_huber_beta', type=float, default=3.0,
                   help='Pixel transition point for last-step Smooth-L1 refinement')
    p.add_argument('--stnet_block3_kernel', type=int, choices=[5, 7], default=5)
    p.add_argument('--stnet_unidirectional', action='store_true', default=False,
                   help='Use a unidirectional GRU (hidden=256) instead of Bi-GRU')
    p.add_argument('--allow_kernel_resize_init', action='store_true', default=False,
                   help='Center-pad compatible spatial kernels when loading STNet')
    p.add_argument('--augment_horizontal_flip', type=float, default=0.0,
                   help='Per-sequence horizontal flip probability')
    p.add_argument('--augment_rotation_degrees', type=float, default=0.0)
    p.add_argument('--augment_shift_pixels', type=float, default=0.0)
    p.add_argument('--augment_event_dropout', type=float, default=0.0,
                   help='Probability of dropping each event-frame value')
    return p.parse_args()


def crop_patch(frames, cx, cy, size):
    """
    frames: (N, C, H, W); cx, cy: (N,) pixel centers. Returns (N, C, size, size)
    zero-padded crops using grid_sample so the op is differentiable and batched.
    """
    n, c, h, w = frames.shape
    half = size // 2
    device = frames.device
    ys = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    xs = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')            # (size, size)
    gx = gx.unsqueeze(0) + cx.view(-1, 1, 1)                  # (N, size, size)
    gy = gy.unsqueeze(0) + cy.view(-1, 1, 1)
    # normalize to [-1, 1] for grid_sample
    norm_x = gx / (w - 1) * 2 - 1
    norm_y = gy / (h - 1) * 2 - 1
    grid = torch.stack([norm_x, norm_y], dim=-1)             # (N, size, size, 2)
    return F.grid_sample(frames, grid, align_corners=True, padding_mode='zeros')


def augment_stnet_batch(frames, labels, args):
    """Apply one label-synchronized spatial transform per sequence."""
    batch_size, sequence_length, channels, height, width = frames.shape
    output_frames = frames
    output_labels = labels.clone()
    rotation = args.augment_rotation_degrees
    shift = args.augment_shift_pixels
    if rotation > 0.0 or shift > 0.0:
        angles = (torch.rand(batch_size, device=frames.device) * 2.0 - 1.0)
        angles = angles * rotation * np.pi / 180.0
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)
        translation = torch.empty(batch_size, 2, device=frames.device)
        translation[:, 0] = (
            (torch.rand(batch_size, device=frames.device) * 2.0 - 1.0)
            * shift * 2.0 / max(width - 1, 1))
        translation[:, 1] = (
            (torch.rand(batch_size, device=frames.device) * 2.0 - 1.0)
            * shift * 2.0 / max(height - 1, 1))
        forward = torch.zeros(batch_size, 2, 2, device=frames.device)
        forward[:, 0, 0] = cos_a
        forward[:, 0, 1] = -sin_a
        forward[:, 1, 0] = sin_a
        forward[:, 1, 1] = cos_a
        inverse = forward.transpose(1, 2)
        inverse_translation = -torch.bmm(
            inverse, translation.unsqueeze(-1)).squeeze(-1)
        theta = torch.cat([inverse, inverse_translation.unsqueeze(-1)], dim=-1)
        theta = theta.repeat_interleave(sequence_length, dim=0)
        flattened = output_frames.reshape(
            batch_size * sequence_length, channels, height, width)
        grid = F.affine_grid(theta, flattened.shape, align_corners=True)
        output_frames = F.grid_sample(
            flattened, grid, align_corners=True, padding_mode='zeros')
        output_frames = output_frames.reshape(
            batch_size, sequence_length, channels, height, width)
        normalized = output_labels[:, :, :2] * 2.0 - 1.0
        normalized = torch.bmm(
            normalized, forward.transpose(1, 2)) + translation.unsqueeze(1)
        output_labels[:, :, :2] = (normalized + 1.0) / 2.0

    if args.augment_horizontal_flip > 0.0:
        flip = torch.rand(batch_size, device=frames.device) < args.augment_horizontal_flip
        if flip.any():
            output_frames = output_frames.clone()
            output_frames[flip] = torch.flip(output_frames[flip], dims=(-1,))
            output_labels[flip, :, 0] = 1.0 - output_labels[flip, :, 0]

    if args.augment_event_dropout > 0.0:
        keep = torch.rand_like(output_frames) >= args.augment_event_dropout
        output_frames = output_frames * keep.to(output_frames.dtype)
    output_labels[:, :, :2].clamp_(0.0, 1.0)
    return output_frames, output_labels


def match_loss(matcher, frames, labels, frames_per_seq, args,
               max_gap=8, generator=None):
    """Cross-frame NCC loss with hard-negative and boundary supervision."""
    b, s, c, h, w = frames.shape
    device = frames.device
    if frames_per_seq < 1 or s <= max_gap:
        raise ValueError('frames_per_seq must be positive and sequence length must exceed max_gap')

    gap = torch.randint(
        1, max_gap + 1, (b, frames_per_seq), device=device, generator=generator)
    t = torch.randint(
        max_gap, s, (b, frames_per_seq), device=device, generator=generator)
    tp = t - gap
    batch_idx = torch.arange(b, device=device).unsqueeze(1).expand(-1, frames_per_seq)

    frame_t = frames[batch_idx, t].reshape(-1, c, h, w)
    frame_tp = frames[batch_idx, tp].reshape(-1, c, h, w)
    gt_t = labels[batch_idx, t, :2].reshape(-1, 2)
    gt_tp = labels[batch_idx, tp, :2].reshape(-1, 2)
    n = frame_t.shape[0]

    tp_x = gt_tp[:, 0] * (w - 1)
    tp_y = gt_tp[:, 1] * (h - 1)
    template = crop_patch(frame_tp, tp_x, tp_y, TEMPLATE_SIZE)

    gt_x = gt_t[:, 0] * (w - 1)
    gt_y = gt_t[:, 1] * (h - 1)
    jitter = (
        torch.rand(n, 2, device=device, generator=generator) - 0.5
    ) * (SEARCH_SIZE - TEMPLATE_SIZE)
    sc_x = gt_x + jitter[:, 0]
    sc_y = gt_y + jitter[:, 1]
    search = crop_patch(frame_t, sc_x, sc_y, SEARCH_SIZE)

    response = matcher.ncc_response(template, search)
    coord, _ = soft_argmax_2d(response, matcher.temperature)
    oh, ow = response.shape[1], response.shape[2]
    half_s = SEARCH_SIZE // 2
    half_t = TEMPLATE_SIZE // 2
    tgt_x = (gt_x - (sc_x - half_s) - half_t) / (ow - 1)
    tgt_y = (gt_y - (sc_y - half_s) - half_t) / (oh - 1)
    target = torch.stack([tgt_x, tgt_y], dim=-1).clamp(0, 1)
    coordinate_loss = F.mse_loss(coord, target)

    target_x = (target[:, 0] * (ow - 1)).round().long()
    target_y = (target[:, 1] * (oh - 1)).round().long()
    yy = torch.arange(oh, device=device).view(1, oh, 1)
    xx = torch.arange(ow, device=device).view(1, 1, ow)
    positive_mask = (
        (xx - target_x.view(-1, 1, 1)).abs() <= args.positive_radius
    ) & (
        (yy - target_y.view(-1, 1, 1)).abs() <= args.positive_radius
    )
    positive_score = response[torch.arange(n, device=device), target_y, target_x]
    negative_score = response.masked_fill(positive_mask, float('-inf')).flatten(1).max(1).values
    hard_negative_loss = F.relu(
        args.hard_negative_margin - positive_score + negative_score).mean()

    boundary_mask = torch.zeros_like(response, dtype=torch.bool)
    boundary_mask[:, 0, :] = True
    boundary_mask[:, -1, :] = True
    boundary_mask[:, :, 0] = True
    boundary_mask[:, :, -1] = True
    boundary_mask &= ~positive_mask
    boundary_score = response.masked_fill(
        ~boundary_mask, float('-inf')).flatten(1).max(1).values
    boundary_loss = F.relu(
        args.boundary_margin - positive_score + boundary_score).mean()

    target_index = target_y * ow + target_x
    peak_ce_loss = F.cross_entropy(response.flatten(1) / matcher.temperature, target_index)
    total = (
        coordinate_loss
        + args.hard_negative_weight * hard_negative_loss
        + args.boundary_weight * boundary_loss
        + args.peak_ce_weight * peak_ce_loss
    )

    hard_coord, _ = hard_argmax_2d(response.detach())
    pixel_scale = response.new_tensor([ow - 1, oh - 1])
    hard_error = torch.linalg.vector_norm((hard_coord - target) * pixel_scale, dim=-1)
    peak_x = (hard_coord[:, 0] * (ow - 1)).round().long()
    peak_y = (hard_coord[:, 1] * (oh - 1)).round().long()
    boundary_rate = (
        (peak_x == 0) | (peak_x == ow - 1)
        | (peak_y == 0) | (peak_y == oh - 1)
    ).float().mean()
    metrics = {
        'coordinate': coordinate_loss.detach(),
        'hard_negative': hard_negative_loss.detach(),
        'boundary': boundary_loss.detach(),
        'peak_ce': peak_ce_loss.detach(),
        'hard_error': hard_error.mean(),
        'boundary_rate': boundary_rate,
        'samples': n,
    }
    return total, metrics


def parse_window_sizes(value):
    if not value.strip():
        return []
    sizes = sorted({int(item.strip()) for item in value.split(',') if item.strip()})
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError('--short_window_sizes must contain positive integers')
    return sizes


def stnet_supervised_loss(predict_w, predict_h, labels, criterion,
                          width_scale, height_scale, last_step_only=False,
                          coord_loss_weight=0.0):
    if last_step_only:
        predict_w = predict_w[:, -1:]
        predict_h = predict_h[:, -1:]
        labels = labels[:, -1:]
    loss = criterion(
        predict_w, predict_h, labels[:, :, 0], labels[:, :, 1])
    outputs = decode_soft_expectation(predict_w, predict_h)
    coordinate_loss = F.mse_loss(outputs, labels[:, :, :2])
    loss = loss + coord_loss_weight * coordinate_loss
    error, count = px_euclidean_dist(
        labels[:, :, :2], outputs,
        width_scale=width_scale, height_scale=height_scale)
    return loss, error / max(count, 1), count, coordinate_loss.detach()


def short_window_loss(stnet, frames, labels, criterion, window_size,
                      width_scale, height_scale, generator=None,
                      last_step_only=False, coord_loss_weight=0.0,
                      teacher_logits=None, distill_weight=0.0,
                      distill_temperature=2.0, pixel_huber_weight=0.0,
                      pixel_huber_beta=3.0):
    """Run STNet on one random fixed-length window per sequence."""
    batch_size, sequence_length = frames.shape[:2]
    if window_size > sequence_length:
        raise ValueError('short-window size exceeds the sequence length')
    starts = torch.randint(
        0, sequence_length - window_size + 1, (batch_size,),
        device=frames.device, generator=generator)
    offsets = torch.arange(window_size, device=frames.device).unsqueeze(0)
    indices = starts.unsqueeze(1) + offsets
    batch_indices = torch.arange(batch_size, device=frames.device).unsqueeze(1)
    window_frames = frames[batch_indices, indices]
    window_labels = labels[batch_indices, indices]
    predict_w, predict_h = stnet(window_frames)
    loss, error, count, coordinate_loss = stnet_supervised_loss(
        predict_w, predict_h, window_labels, criterion,
        width_scale, height_scale,
        last_step_only=last_step_only,
        coord_loss_weight=coord_loss_weight)

    zero = loss.new_zeros(())
    distill_loss = zero
    end_indices = starts + window_size - 1
    row_indices = torch.arange(batch_size, device=frames.device)
    if teacher_logits is not None and distill_weight > 0.0:
        teacher_w, teacher_h = teacher_logits
        target_w = teacher_w[row_indices, end_indices]
        target_h = teacher_h[row_indices, end_indices]
        student_w = predict_w[:, -1]
        student_h = predict_h[:, -1]
        temperature = distill_temperature
        distill_w = F.kl_div(
            F.log_softmax(student_w / temperature, dim=-1),
            F.softmax(target_w / temperature, dim=-1), reduction='batchmean')
        distill_h = F.kl_div(
            F.log_softmax(student_h / temperature, dim=-1),
            F.softmax(target_h / temperature, dim=-1), reduction='batchmean')
        distill_loss = 0.5 * (distill_w + distill_h) * temperature ** 2
        loss = loss + distill_weight * distill_loss

    pixel_huber_loss = zero
    if pixel_huber_weight > 0.0:
        student_coord = decode_soft_expectation(
            predict_w[:, -1:], predict_h[:, -1:])[:, 0]
        target_coord = labels[row_indices, end_indices, :2]
        pixel_scale = student_coord.new_tensor([width_scale, height_scale])
        pixel_huber_loss = F.smooth_l1_loss(
            student_coord * pixel_scale, target_coord * pixel_scale,
            beta=pixel_huber_beta)
        loss = loss + pixel_huber_weight * pixel_huber_loss

    auxiliary = {
        'distill': distill_loss.detach(),
        'pixel_huber': pixel_huber_loss.detach(),
    }
    return loss, error, count, coordinate_loss, auxiliary


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_train_val(frames, labels, val_ratio, seed):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f'val_ratio must be between 0 and 1, got {val_ratio}')
    n_sequences = len(frames)
    if n_sequences < 2:
        raise ValueError('At least two sequences are required for a train/val split')
    val_size = max(1, min(n_sequences - 1, int(round(n_sequences * val_ratio))))
    indices = np.random.default_rng(seed).permutation(n_sequences)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return (frames[train_indices], labels[train_indices],
            frames[val_indices], labels[val_indices],
            train_indices.tolist(), val_indices.tolist())


def checkpoint_payload(model, args, epoch, best_error, split):
    return {
        'format_version': 1,
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'best_val_mean_pixel_error': best_error,
        'selection_metric': (
            'closed_loop_error_plus_trigger'
            if args.ncc_closed_loop_val else
            'ncc_hard_argmax_pixel_error' if args.ncc_only else
            ('stnet_last_step_full_short_weighted_error'
             if args.stnet_last_step_only else 'stnet_full_short_weighted_error')
            if args.stnet_only else 'stnet_mean_pixel_error'
        ),
        'seed': args.seed,
        'config': vars(args),
        'split': split,
    }


def load_model_weights(model, path, device, allow_kernel_resize=False):
    checkpoint = torch.load(path, map_location=device)
    state_dict = (
        checkpoint['model_state_dict']
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
        else checkpoint
    )
    if allow_kernel_resize:
        target_state = model.state_dict()
        resized_state = dict(state_dict)
        for key, source in state_dict.items():
            target = target_state.get(key)
            if target is None or source.shape == target.shape:
                continue
            compatible = (
                source.ndim == 5 and target.ndim == 5
                and source.shape[:3] == target.shape[:3]
                and source.shape[-2] <= target.shape[-2]
                and source.shape[-1] <= target.shape[-1]
            )
            if not compatible:
                raise ValueError(
                    f'cannot resize checkpoint tensor {key}: '
                    f'{tuple(source.shape)} -> {tuple(target.shape)}')
            padded = torch.zeros_like(target)
            offset_y = (target.shape[-2] - source.shape[-2]) // 2
            offset_x = (target.shape[-1] - source.shape[-1]) // 2
            padded[..., offset_y:offset_y + source.shape[-2],
                   offset_x:offset_x + source.shape[-1]] = source
            resized_state[key] = padded
            print(f'center-padded checkpoint kernel {key}: '
                  f'{tuple(source.shape)} -> {tuple(target.shape)}')
        state_dict = resized_state
    model.load_state_dict(state_dict, strict=True)


def evaluate_closed_loop(stnet, matcher, frames, labels, args, device):
    """Evaluate fixed validation sequences with the deployed closed-loop tracker."""
    sequence_count = min(args.closed_val_sequences, len(frames))
    if sequence_count < 1:
        raise ValueError('--closed_val_sequences must select at least one sequence')
    tracker = KTCalTracker(
        stnet, matcher, tau1=3.0, tau2=5.0,
        tau3=args.closed_val_tau3, ncc_thresh=0.5,
        hard_reset=args.closed_val_hard_reset, device=device)
    pixel_scale = np.array([
        args.sensor_width * args.spatial_factor,
        args.sensor_height * args.spatial_factor,
    ])
    total_error = 0.0
    total_frames = 0
    stnet_calls = 0
    matcher.eval()
    stnet.eval()
    with torch.no_grad():
        for sequence_index in range(sequence_count):
            coords, stats = tracker.track_sequence(frames[sequence_index])
            target = labels[sequence_index, :, :2].cpu().numpy()
            distance = np.linalg.norm((coords - target) * pixel_scale, axis=-1)
            total_error += distance.sum()
            total_frames += len(distance)
            stnet_calls += stats['stnet_calls']
    mean_error = total_error / max(total_frames, 1)
    trigger_ratio = stnet_calls / max(total_frames, 1)
    score = mean_error + args.closed_val_trigger_weight * trigger_ratio
    return {
        'mean_error': mean_error,
        'trigger_ratio': trigger_ratio,
        'score': score,
        'sequences': sequence_count,
    }


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    seed_everything(args.seed)
    short_window_sizes = parse_window_sizes(args.short_window_sizes)
    if args.ncc_only and args.stnet_only:
        raise ValueError('--ncc_only and --stnet_only are mutually exclusive')
    if short_window_sizes and not args.stnet_only:
        raise ValueError('--short_window_sizes requires --stnet_only')
    if args.stnet_only and not short_window_sizes:
        raise ValueError('--stnet_only requires --short_window_sizes')
    if not 0.0 <= args.full_sequence_probability <= 1.0:
        raise ValueError('--full_sequence_probability must be in [0, 1]')
    if not 0.0 <= args.augment_horizontal_flip <= 1.0:
        raise ValueError('--augment_horizontal_flip must be in [0, 1]')
    if not 0.0 <= args.augment_event_dropout < 1.0:
        raise ValueError('--augment_event_dropout must be in [0, 1)')
    if args.augment_rotation_degrees < 0.0 or args.augment_shift_pixels < 0.0:
        raise ValueError('augmentation magnitudes must be non-negative')
    if args.positive_radius < 0:
        raise ValueError('--positive_radius must be non-negative')
    if min(args.hard_negative_margin, args.boundary_margin,
           args.hard_negative_weight, args.boundary_weight,
           args.peak_ce_weight, args.closed_val_trigger_weight,
           args.full_sequence_weight, args.short_window_weight,
           args.stnet_coord_loss_weight, args.stnet_distill_weight,
           args.stnet_pixel_huber_weight) < 0.0:
        raise ValueError('NCC margin and loss-weight arguments must be non-negative')
    if args.stnet_distill_temperature <= 0.0 or args.stnet_pixel_huber_beta <= 0.0:
        raise ValueError('distillation temperature and pixel Huber beta must be positive')
    if args.stnet_distill_weight > 0.0 and not args.stnet_teacher_ckpt:
        raise ValueError('--stnet_distill_weight requires --stnet_teacher_ckpt')
    if args.stnet_teacher_ckpt and not args.stnet_only:
        raise ValueError('--stnet_teacher_ckpt requires --stnet_only')
    if args.stnet_teacher_ckpt and (
            args.augment_horizontal_flip > 0.0
            or args.augment_rotation_degrees > 0.0
            or args.augment_shift_pixels > 0.0
            or args.augment_event_dropout > 0.0):
        raise ValueError('teacher distillation currently requires augmentation to be disabled')
    if args.ncc_closed_loop_val and (not args.ncc_only or not args.stnet_val_ckpt):
        raise ValueError('--ncc_closed_loop_val requires --ncc_only and --stnet_val_ckpt')
    device = torch.device('cpu' if args.use_cpu else 'cuda')
    if args.test_h5_path:
        print('[warn] --test_h5_path is deprecated and ignored; test data is evaluation-only')

    frames, labels = provider_data.load_h5_f(args.train_h5_path)
    frames = torch.from_numpy(np.array(frames))
    labels = torch.from_numpy(np.array(labels))
    (train_frames, train_labels, val_frames, val_labels,
     train_indices, val_indices) = split_train_val(frames, labels, args.val_ratio, args.seed)
    split = {
        'source': os.path.abspath(args.train_h5_path),
        'train_indices': train_indices,
        'val_indices': val_indices,
    }
    print(f'data split: train={len(train_indices)} val={len(val_indices)} seed={args.seed}')

    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_frames, train_labels),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=train_generator)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_frames, val_labels),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    stnet = None if args.ncc_only else STNet(args).to(device)
    matcher = None if args.stnet_only else NCCMatcher().to(device)
    if args.stnet_init_ckpt:
        if stnet is None:
            raise ValueError('--stnet_init_ckpt cannot be used with --ncc_only')
        load_model_weights(
            stnet, args.stnet_init_ckpt, device,
            allow_kernel_resize=args.allow_kernel_resize_init)
        print(f'loaded STNet initialization: {args.stnet_init_ckpt}')
    if args.ncc_init_ckpt:
        if matcher is None:
            raise ValueError('--ncc_init_ckpt cannot be used with --stnet_only')
        load_model_weights(matcher, args.ncc_init_ckpt, device)
        print(f'loaded NCC initialization: {args.ncc_init_ckpt}')
    teacher_stnet = None
    if args.stnet_teacher_ckpt:
        teacher_stnet = STNet(args).to(device)
        load_model_weights(teacher_stnet, args.stnet_teacher_ckpt, device)
        teacher_stnet.eval()
        for parameter in teacher_stnet.parameters():
            parameter.requires_grad_(False)
        print(f'loaded frozen STNet teacher: {args.stnet_teacher_ckpt}')
    validation_stnet = None
    if args.ncc_closed_loop_val:
        validation_stnet = STNet(args).to(device)
        load_model_weights(validation_stnet, args.stnet_val_ckpt, device)
        validation_stnet.eval()
        for parameter in validation_stnet.parameters():
            parameter.requires_grad_(False)
        print(f'loaded validation STNet: {args.stnet_val_ckpt}')
    criterion = None if args.ncc_only else KLDiscretLoss()

    params = []
    if matcher is not None:
        params += list(matcher.parameters())
    if stnet is not None:
        params += list(stnet.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=8, T_mult=2, eta_min=1e-6)

    if not args.no_save:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.save_path, 'run_config.json'), 'w', encoding='utf-8') as f:
            json.dump({'config': vars(args), 'split': split}, f, indent=2)
    best_err = float('inf')
    epochs_no_improve = 0
    if args.ncc_closed_loop_val:
        baseline = evaluate_closed_loop(
            validation_stnet, matcher, val_frames, val_labels, args, device)
        best_err = baseline['score']
        print(f"[closed-val] epoch 0: error={baseline['mean_error']:.4f}px "
              f"trigger={baseline['trigger_ratio']:.4f} score={best_err:.4f}")
        if not args.no_save:
            torch.save(
                checkpoint_payload(matcher, args, 0, best_err, split),
                os.path.join(args.save_path, 'ncc_best.pth'))
            print('  initial clean NCC saved as the closed-loop baseline')

    for epoch in range(args.epoch):
        if stnet is not None:
            stnet.train()
        if matcher is not None:
            matcher.train()
        running = {
            'total': 0.0, 'st': 0.0, 'match': 0.0,
            'hard_negative': 0.0, 'boundary': 0.0,
            'peak_ce': 0.0, 'hard_error': 0.0, 'short_error': 0.0,
            'coord_l2': 0.0, 'distill': 0.0, 'pixel_huber': 0.0,
        }
        for batch_idx, (frame, label) in enumerate(tqdm(train_loader, desc=f'epoch {epoch+1} train')):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            frame = frame.float().to(device)
            label = label.float().to(device)
            if args.stnet_only and (
                    args.augment_horizontal_flip > 0.0
                    or args.augment_rotation_degrees > 0.0
                    or args.augment_shift_pixels > 0.0
                    or args.augment_event_dropout > 0.0):
                frame, label = augment_stnet_batch(frame, label, args)
            optimizer.zero_grad()

            if args.stnet_only:
                teacher_logits = None
                if teacher_stnet is not None:
                    with torch.no_grad():
                        teacher_logits = teacher_stnet(frame)
                if random.random() < args.full_sequence_probability:
                    predict_w, predict_h = stnet(frame)
                    loss_st, _, _, full_coord_loss = stnet_supervised_loss(
                        predict_w, predict_h, label, criterion,
                        args.sensor_width * args.spatial_factor,
                        args.sensor_height * args.spatial_factor,
                        last_step_only=args.stnet_last_step_only,
                        coord_loss_weight=args.stnet_coord_loss_weight)
                else:
                    loss_st = frame.new_zeros(())
                    full_coord_loss = frame.new_zeros(())
                selected_window = random.choice(short_window_sizes)
                loss_short, short_error, _, short_coord_loss, short_aux = short_window_loss(
                    stnet, frame, label, criterion, selected_window,
                    args.sensor_width * args.spatial_factor,
                    args.sensor_height * args.spatial_factor,
                    last_step_only=args.stnet_last_step_only,
                    coord_loss_weight=args.stnet_coord_loss_weight,
                    teacher_logits=teacher_logits,
                    distill_weight=args.stnet_distill_weight,
                    distill_temperature=args.stnet_distill_temperature,
                    pixel_huber_weight=args.stnet_pixel_huber_weight,
                    pixel_huber_beta=args.stnet_pixel_huber_beta)
                loss_match = frame.new_zeros(())
                loss = (
                    args.full_sequence_weight * loss_st
                    + args.short_window_weight * loss_short
                )
                zero = frame.new_zeros(())
                match_metrics = {
                    'hard_negative': zero, 'boundary': zero,
                    'peak_ce': zero, 'hard_error': zero,
                }
                coord_metric = full_coord_loss + short_coord_loss
            else:
                coord_metric = frame.new_zeros(())
                short_aux = {'distill': coord_metric, 'pixel_huber': coord_metric}
                loss_match, match_metrics = match_loss(
                    matcher, frame, label, args.frames_per_seq, args)
                short_error = frame.new_zeros(())
                if stnet is None:
                    loss_st = frame.new_zeros(())
                    loss = loss_match
                else:
                    predict_w, predict_h = stnet(frame)
                    loss_st = criterion(
                        predict_w, predict_h, label[:, :, 0], label[:, :, 1])
                    loss = loss_st + LAMBDA_MATCH * loss_match

            loss.backward()
            optimizer.step()
            running['total'] += loss.item()
            running['st'] += loss_st.item()
            running['match'] += loss_match.item()
            running['hard_negative'] += match_metrics['hard_negative'].item()
            running['boundary'] += match_metrics['boundary'].item()
            running['peak_ce'] += match_metrics['peak_ce'].item()
            running['hard_error'] += match_metrics['hard_error'].item()
            running['short_error'] += short_error.item()
            running['coord_l2'] += coord_metric.item()
            running['distill'] += short_aux['distill'].item()
            running['pixel_huber'] += short_aux['pixel_huber'].item()
        scheduler.step(epoch)

        n_batches = args.max_train_batches or len(train_loader)
        n_batches = min(n_batches, len(train_loader))
        print(f"[train] epoch {epoch+1}: total={running['total']/n_batches:.4f} "
              f"L_ST={running['st']/n_batches:.4f} L_match={running['match']/n_batches:.4f} "
              f"hard_neg={running['hard_negative']/n_batches:.4f} "
              f"boundary={running['boundary']/n_batches:.4f} "
              f"peak_ce={running['peak_ce']/n_batches:.4f} "
              f"hard_err={running['hard_error']/n_batches:.4f}px "
              f"short_err={running['short_error']/n_batches:.4f}px "
              f"coord_l2={running['coord_l2']/n_batches:.6f} "
              f"distill={running['distill']/n_batches:.4f} "
              f"pixel_huber={running['pixel_huber']/n_batches:.4f}")

        if stnet is not None:
            stnet.eval()
        if matcher is not None:
            matcher.eval()
        total_err, total_n = 0.0, 0
        total_short_error, total_short_n = 0.0, 0
        total_boundary = 0.0
        val_generator = torch.Generator(device=device).manual_seed(args.seed + 10000)
        with torch.no_grad():
            for batch_idx, (frame, label) in enumerate(val_loader):
                if args.max_val_batches and batch_idx >= args.max_val_batches:
                    break
                frame = frame.float().to(device)
                label = label.float().to(device)
                if stnet is None:
                    _, val_metrics = match_loss(
                        matcher, frame, label, args.frames_per_seq, args,
                        generator=val_generator)
                    n = val_metrics['samples']
                    total_err += val_metrics['hard_error'].item() * n
                    total_boundary += val_metrics['boundary_rate'].item() * n
                    total_n += n
                else:
                    pw, ph = stnet(frame)
                    if args.stnet_only:
                        _, full_error, n, _ = stnet_supervised_loss(
                            pw, ph, label, criterion,
                            args.sensor_width * args.spatial_factor,
                            args.sensor_height * args.spatial_factor,
                            last_step_only=args.stnet_last_step_only,
                            coord_loss_weight=0.0)
                        total_err += full_error.item() * n
                        total_n += n
                    else:
                        outputs = decode_soft_expectation(pw, ph)
                        err, n = px_euclidean_dist(
                            label[:, :, :2], outputs[:, :],
                            width_scale=args.sensor_width * args.spatial_factor,
                            height_scale=args.sensor_height * args.spatial_factor)
                        total_err += err.item()
                        total_n += n
                    if args.stnet_only:
                        for window_size in short_window_sizes:
                            for _ in range(args.short_val_windows_per_seq):
                                _, short_error, short_n, _, _ = short_window_loss(
                                    stnet, frame, label, criterion, window_size,
                                    args.sensor_width * args.spatial_factor,
                                    args.sensor_height * args.spatial_factor,
                                    generator=val_generator,
                                    last_step_only=args.stnet_last_step_only,
                                    coord_loss_weight=0.0)
                                total_short_error += short_error.item() * short_n
                                total_short_n += short_n
        val_err = total_err / max(total_n, 1)
        if stnet is None:
            print(f"[val]   epoch {epoch+1}: NCC hard error={val_err:.4f}px "
                  f"boundary rate={total_boundary/max(total_n, 1):.4f}")
        else:
            print(f"[val]   epoch {epoch+1}: mean pixel error = {val_err:.4f} px")

        selection_err = val_err
        if args.stnet_only:
            short_val_err = total_short_error / max(total_short_n, 1)
            weight_sum = args.full_sequence_weight + args.short_window_weight
            selection_err = (
                args.full_sequence_weight * val_err
                + args.short_window_weight * short_val_err
            ) / max(weight_sum, 1e-12)
            print(f"[short-val] epoch {epoch+1}: error={short_val_err:.4f}px "
                  f"weighted score={selection_err:.4f}")
        if args.ncc_closed_loop_val:
            closed_val = evaluate_closed_loop(
                validation_stnet, matcher, val_frames, val_labels, args, device)
            selection_err = closed_val['score']
            print(f"[closed-val] epoch {epoch+1}: "
                  f"error={closed_val['mean_error']:.4f}px "
                  f"trigger={closed_val['trigger_ratio']:.4f} "
                  f"score={selection_err:.4f}")

        if selection_err < best_err - args.min_delta:
            best_err = selection_err
            epochs_no_improve = 0
            if not args.no_save:
                if stnet is not None:
                    torch.save(
                        checkpoint_payload(stnet, args, epoch + 1, best_err, split),
                        os.path.join(args.save_path, 'stnet_best.pth'))
                if matcher is not None:
                    torch.save(
                        checkpoint_payload(matcher, args, epoch + 1, best_err, split),
                        os.path.join(args.save_path, 'ncc_best.pth'))
                metric_name = (
                    'closed-loop score' if args.ncc_closed_loop_val else
                    'NCC hard error' if stnet is None else
                    'full/short weighted error' if args.stnet_only else
                    'mean pixel error'
                )
                print(f"  best model saved ({metric_name} {best_err:.4f})")
            else:
                print(f"  new best selection score {best_err:.4f} (saving disabled)")
        else:
            epochs_no_improve += 1
            print(f"  no improve for {epochs_no_improve}/{args.patience} epochs (best {best_err:.4f})")
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1}: no improvement for {args.patience} epochs.")
                break


if __name__ == '__main__':
    main()
