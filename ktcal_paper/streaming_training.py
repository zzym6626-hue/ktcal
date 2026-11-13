"""Full-record causal supervision and optional train-only teacher targets."""
import json
import math
from pathlib import Path
import platform
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .combined_3et import SCALE, _sha256, load_stnet_checkpoint, validate_validation_h5
from .stnet_sequence import decode_soft_expectation
from .stnet_sequence import (
    _configure, _save_checkpoint, _write_json,
    masked_official_simdr_loss,
)
from .evaluation_protocol import summarize_pixel_errors
from .streaming_objectives import streaming_frame_weights, weighted_coordinate_huber


def _training_path(path):
    path = Path(path).resolve()
    if path.name.lower() != 'train_aug.h5':
        raise ValueError('Training and teacher targets require train_aug.h5')
    return validate_validation_h5(path)


class StreamingSequenceDataset(Dataset):
    def __init__(self, path, split='train', teacher_cache=None, preload=False):
        if split == 'train':
            self.path, self.label_key, self.shape, _ = _training_path(path)
        elif split == 'validation':
            path = Path(path).resolve()
            if path.name.lower() != 'test_aug.h5' or teacher_cache is not None:
                raise ValueError('Validation requires test_aug.h5 without a teacher')
            self.path, self.label_key, self.shape, _ = validate_validation_h5(path)
        else:
            raise ValueError('Unknown streaming dataset split')
        self.split, self._handle = split, None
        self._frames = self._labels = self.teacher = None
        if teacher_cache is not None:
            with np.load(teacher_cache, allow_pickle=False) as cache:
                metadata = json.loads(str(cache['metadata']))
                coordinates = cache['coordinates'].astype(np.float32)
            if (metadata.get('split') != 'train'
                    or metadata.get('source_h5_sha256') != _sha256(self.path)
                    or len(metadata.get('teacher_checkpoint_sha256', '')) != 64
                    or coordinates.shape != self.shape[:2] + (2,)
                    or not np.isfinite(coordinates).all()):
                raise ValueError('Invalid teacher cache source, split or coordinates')
            self.teacher = coordinates
        if preload:
            with h5py.File(self.path, 'r') as handle:
                self._frames = handle['frames'][:]
                self._labels = handle[self.label_key][:, :, :2]

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if self._frames is None:
            if self._handle is None:
                self._handle = h5py.File(self.path, 'r', rdcc_nbytes=32 * 1024 * 1024)
            frames = self._handle['frames'][index].astype(np.float32)
            labels = self._handle[self.label_key][index, :, :2].astype(np.float32)
        else:
            frames = self._frames[index].astype(np.float32)
            labels = self._labels[index].astype(np.float32)
        teacher = labels.copy() if self.teacher is None else self.teacher[index].copy()
        return torch.from_numpy(frames), torch.from_numpy(labels), torch.from_numpy(teacher)

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        if getattr(self, '_handle', None) is not None:
            self.close()


def transform_coordinates(coordinates, view, flip_axis_px=79.):
    if not math.isfinite(flip_axis_px) or not 79. <= flip_axis_px <= 80.:
        raise ValueError('Finite training flip axis in[79,80] required')
    dx, dy, flip = view
    shifted = coordinates + coordinates.new_tensor([dx / 80., dy / 60.])
    if flip:
        shifted = torch.stack((flip_axis_px / 80. - shifted[..., 0], shifted[..., 1]), -1)
    return shifted


def teacher_coordinate_loss(logits_x, logits_y, teacher, labels, max_error_px=5., *,
                            weights=None, temperature=1., valid=None):
    if not np.isfinite(max_error_px) or max_error_px <= 0:
        raise ValueError('Positive finite teacher trust radius required')
    scale = labels.new_tensor([80., 60.])
    teacher = teacher.detach()
    trusted = (torch.linalg.vector_norm((teacher - labels) * scale, dim=-1) < max_error_px
               ) & ((labels >= 0) & (labels <= 1)).all(-1)
    if valid is not None:
        if valid.shape != trusted.shape or valid.dtype != torch.bool:
            raise ValueError('Teacher validity must align with received target timesteps')
        trusted = trusted & valid
    return weighted_coordinate_huber(logits_x, logits_y, teacher, trusted,
                                      weights=weights, temperature=temperature)


def spatial_heatmap_loss(heatmap, labels, valid, sigma_px=3.):
    if not math.isfinite(sigma_px) or sigma_px <= 0 or heatmap.shape[-2:] != (20, 26):
        raise ValueError('Expected20x26 heatmaps and positive spatial target sigma')
    if labels.shape != heatmap.shape[:-2] + (2,) or valid.shape != heatmap.shape[:-2]:
        raise ValueError('Spatial labels/masks must match heatmaps')
    grid_x = 1. + 3. * torch.arange(26, device=heatmap.device, dtype=heatmap.dtype)
    grid_y = 1. + 3. * torch.arange(20, device=heatmap.device, dtype=heatmap.dtype)
    dx = grid_x - labels[..., 0, None] * 80.
    dy = grid_y - labels[..., 1, None] * 60.
    target = (-(dx[..., None, :].square() + dy[..., :, None].square())
              / (2 * sigma_px ** 2)).flatten(-2).softmax(-1)
    loss = F.kl_div(heatmap.flatten(-2).log_softmax(-1), target, reduction='none').sum(-1)
    return (loss * valid).sum() / valid.sum().clamp_min(1)


@torch.inference_mode()
def build_teacher_cache(train_h5, checkpoint, output, batch_size=4,
                        device='cuda', temperature=.65):
    path, _, _, _ = _training_path(train_h5)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Existing teacher targets cannot be overwritten')
    if batch_size < 1 or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive teacher batch size and temperature required')
    from .offline_experiment import predict_views, smooth_pixels
    model, _, divisor = load_stnet_checkpoint(checkpoint, device)
    views = [(dx, dy, flip) for dx, dy in ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2))
             for flip in (False, True)]
    with h5py.File(path, 'r') as handle:
        frames = torch.from_numpy(handle['frames'][:].astype(np.float32))
    predictions = []
    for begin in range(0, len(frames), 64):
        predictions.append(predict_views(model, frames[begin:begin + 64], views, device=device,
                                         divisor=divisor, batch_size=batch_size,
                                         temperature=temperature))
        print(json.dumps(dict(teacher_records=min(begin + 64, len(frames)),
                              total_training_records=len(frames))), flush=True)
    predictions = np.concatenate(predictions, axis=1)
    coordinates = (smooth_pixels(np.median(predictions, axis=0), 'ma7') / SCALE).astype(np.float32)
    metadata = dict(schema=1, split='train', source_h5_sha256=_sha256(path),
                    teacher_checkpoint_sha256=_sha256(checkpoint),
                    views=views, temperature=temperature, filter='offline_ma7',
                    privileged_future_during_training=True, final_holdout_used=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        np.savez_compressed(handle, coordinates=coordinates, metadata=json.dumps(metadata))
    return metadata


def _startup_frames(frames, warm_start_steps):
    if warm_start_steps:
        return torch.cat((frames[:, :1].expand(-1, warm_start_steps, -1, -1, -1), frames), 1)
    return frames


@torch.inference_mode()
def evaluate_streaming(model, loader, device, warm_start_steps=0, decode_temperature=1.):
    if not math.isfinite(decode_temperature) or decode_temperature <= 0:
        raise ValueError('Positive finite soft decode temperature required')
    model.eval()
    predictions, labels = [], []
    for frames, coordinates, _ in loader:
        x, y = model(_startup_frames(frames.to(device), warm_start_steps))
        predictions.append(decode_soft_expectation(x[:, warm_start_steps:] / decode_temperature,
                            y[:, warm_start_steps:] / decode_temperature).cpu().numpy())
        labels.append(coordinates.numpy())
    predictions, labels = np.concatenate(predictions), np.concatenate(labels)
    result = summarize_pixel_errors(predictions, labels)
    result['first_10_p3'] = summarize_pixel_errors(predictions[:, :10], labels[:, :10])['p3']
    return result


def train_streaming_3et(train_h5, val_h5, *, output, initialize_from=None,
                        teacher_cache=None, teacher_weight=0., teacher_max_error_px=5.,
                        epochs=30, patience=8, batch_size=16, hidden_size=256,
                        lr=1e-4, min_lr=1e-5, weight_decay=1e-4,
                        sigma=1., pixel_loss_weight=.1, dropout=.2,
                        augment=False, preload=False, seed=42, threads=4,
                        device='auto', gradient_clip=5., spatial_localizer=False,
                        spatial_loss_weight=.1, warm_start_steps=0,
                        current_readout=False, current_loss_weight=.25, flip_axis_px=79.,
                        decode_temperature=1., persistent_boost=0., persistent_min_run=5,
                        cold_boost=0., cold_frames=10, learned_initial_state=False,
                        rotation_degrees=0., scale_jitter=0., count_gain_jitter=0.,
                        raw_current_readout=False, raw_current_loss_weight=.25,
                        memory_readout=False, memory_window=8, memory_loss_weight=.25):
    _training_path(train_h5)
    val_h5 = Path(val_h5).resolve()
    if val_h5.name.lower() != 'test_aug.h5':
        raise ValueError('Streaming selection requires test_aug.h5')
    validate_validation_h5(val_h5)
    if spatial_localizer and current_readout:
        raise ValueError('Spatial localization and current readout are mutually exclusive')
    if learned_initial_state and not current_readout:
        raise ValueError('Learned initial state requires current readout')
    if raw_current_readout and not current_readout:
        raise ValueError('Raw current readout requires current readout')
    if raw_current_readout and learned_initial_state:
        raise ValueError('Raw current readout and learned initial state are mutually exclusive')
    if memory_readout and not current_readout:
        raise ValueError('Memory readout requires current readout')
    if memory_readout and (raw_current_readout or learned_initial_state):
        raise ValueError('Memory readout and other auxiliary architectures are mutually exclusive')
    if isinstance(memory_window, bool) or not isinstance(memory_window, int) or not 2 <= memory_window <= 32:
        raise ValueError('Integer memory window in[2,32] required')
    augmentation_values = (rotation_degrees, scale_jitter, count_gain_jitter)
    if (any(not math.isfinite(value) or not 0. <= value <= limit
            for value, limit in zip(augmentation_values, (15., .3, .3)))
            or any(augmentation_values) and not augment):
        raise ValueError('Finite bounded record augmentation requires augment=True')
    if not math.isfinite(flip_axis_px) or not 79. <= flip_axis_px <= 80.:
        raise ValueError('Finite training flip axis in[79,80] required')
    if (isinstance(warm_start_steps, bool) or not isinstance(warm_start_steps, int)
            or warm_start_steps < 0):
        raise ValueError('Nonnegative integer training warm-start steps required')
    if min(epochs, patience, batch_size) < 1 or hidden_size not in (128, 256):
        raise ValueError('Positive training sizes and 128/256 hidden units required')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in (persistent_min_run, cold_frames)):
        raise ValueError('Positive integer difficulty frame counts required')
    if (any(not np.isfinite(v) or v <= 0 for v in
            (lr, min_lr, sigma, teacher_max_error_px, gradient_clip, decode_temperature))
            or min_lr > lr
            or any(not np.isfinite(v) or v < 0 for v in
                   (weight_decay, pixel_loss_weight, teacher_weight, spatial_loss_weight,
                    current_loss_weight, raw_current_loss_weight, memory_loss_weight,
                    persistent_boost, cold_boost))
            or not np.isfinite(dropout) or not 0 <= dropout < 1):
        raise ValueError('Invalid streaming training configuration')
    if teacher_weight and teacher_cache is None:
        raise ValueError('Teacher loss requires a verified train-only teacher cache')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Existing streaming experiments cannot be overwritten')
    resolved_device = _configure(seed, threads, device)
    from models.streaming_stnet import StreamingSTNet
    if memory_readout:
        from models.memory_readout_streaming_stnet import MemoryReadoutStreamingSTNet
        model = MemoryReadoutStreamingSTNet(hidden_size=hidden_size, memory_window=memory_window)
    elif raw_current_readout:
        from models.raw_current_streaming_stnet import RawCurrentStreamingSTNet
        model = RawCurrentStreamingSTNet(hidden_size=hidden_size)
    elif learned_initial_state:
        from models.initialized_streaming_stnet import InitializedCurrentReadoutStreamingSTNet
        model = InitializedCurrentReadoutStreamingSTNet(hidden_size=hidden_size)
    elif current_readout:
        from models.current_readout_streaming_stnet import CurrentReadoutStreamingSTNet
        model = CurrentReadoutStreamingSTNet(hidden_size=hidden_size)
    elif spatial_localizer:
        from models.localization_streaming_stnet import LocalizationStreamingSTNet
        model = LocalizationStreamingSTNet(hidden_size=hidden_size)
    else:
        model = StreamingSTNet(hidden_size=hidden_size)
    initialization = None
    if initialize_from is not None:
        source = torch.load(initialize_from, map_location='cpu', weights_only=False)
        if source.get('config', {}).get('temporal_mode') == 'causal_streaming':
            source_config = source.get('config', {})
            if (source_config.get('spatial_localizer') and not spatial_localizer
                    or source_config.get('current_readout') and not current_readout
                    or source_config.get('learned_initial_state') and not learned_initial_state
                    or source_config.get('raw_current_readout') and not raw_current_readout
                    or source_config.get('memory_readout') and not memory_readout):
                raise ValueError('Cannot initialize from an incompatible auxiliary architecture')
            if memory_readout and not source_config.get('memory_readout'):
                if not source_config.get('current_readout'):
                    raise ValueError('Memory readout resume requires current readout checkpoint')
                initialization = model.initialize_from_current_readout(source['model_state_dict'])
            elif raw_current_readout and not source_config.get('raw_current_readout'):
                if not source_config.get('current_readout'):
                    raise ValueError('Raw current resume requires current readout checkpoint')
                initialization = model.initialize_from_current_readout(source['model_state_dict'])
            elif learned_initial_state and not source_config.get('learned_initial_state'):
                if not source_config.get('current_readout'):
                    raise ValueError('Learned initial state resume requires current readout checkpoint')
                initialization = model.initialize_from_current_readout(source['model_state_dict'])
            elif current_readout and not source_config.get('current_readout', False):
                initialization = model.initialize_from_streaming(source['model_state_dict'])
            elif spatial_localizer and not source_config.get('spatial_localizer', False):
                initialization = model.initialize_from_streaming(source['model_state_dict'])
            else:
                model.load_state_dict(source['model_state_dict'], strict=True)
                initialization = dict(recurrent_initialization='streaming_checkpoint_resume')
        else:
            if memory_readout:
                raise ValueError('Memory readout resume requires causal current readout checkpoint')
            if raw_current_readout:
                raise ValueError('Raw current resume requires causal current readout checkpoint')
            if learned_initial_state:
                raise ValueError('Learned initial state resume requires current readout checkpoint')
            original, _, _ = load_stnet_checkpoint(initialize_from, 'cpu')
            initialization = model.initialize_from_bidirectional(original)
        initialization.update(source=str(Path(initialize_from).resolve()),
                              source_sha256=_sha256(initialize_from))
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = dropout
    model.spatialdropout.drop = dropout
    model.args.stnet_flip_axis_px = float(flip_axis_px)
    model.to(resolved_device)
    train_data = StreamingSequenceDataset(train_h5, split='train',
                                          teacher_cache=teacher_cache, preload=preload)
    val_data = StreamingSequenceDataset(val_h5, split='validation', preload=preload)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=generator)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    if spatial_localizer or current_readout:
        special_parameters = (list(model.current_readout.parameters()) if current_readout else
                              list(model.localizer.parameters()) + list(model.localizer_condition.parameters()))
        scale_parameter = model.current_readout_scale if current_readout else model.localizer_scale
        scale_parameters = [scale_parameter]
        if learned_initial_state:
            special_parameters += list(model.initial_hidden_projection.parameters())
            scale_parameters.append(model.initial_hidden_scale)
        if raw_current_readout:
            special_parameters += list(model.raw_current_encoder.parameters()) + list(model.raw_current_head.parameters())
            scale_parameters.append(model.raw_current_scale)
        if memory_readout:
            special_parameters += (list(model.memory_query.parameters()) + list(model.memory_key.parameters())
                                   + list(model.memory_value.parameters()) + list(model.memory_head.parameters())
                                   + [model.memory_relative_bias])
            scale_parameters.append(model.memory_scale)
        special_ids = {id(parameter) for parameter in special_parameters + scale_parameters}
        groups = [dict(params=[p for p in model.parameters() if id(p) not in special_ids], lr_multiplier=1.),
                  dict(params=special_parameters, lr_multiplier=10.),
                  dict(params=scale_parameters, lr_multiplier=20.)]
    else:
        groups = model.parameters()
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    config = dict(
        schema=1, dataset='3ET+2025', temporal_mode='causal_streaming',
        architecture=type(model).__module__ + '.' + type(model).__name__,
        spatial_localizer=spatial_localizer, spatial_loss_weight=spatial_loss_weight,
        localizer_lr_multiplier=10. if spatial_localizer else None,
        localizer_scale_lr_multiplier=20. if spatial_localizer else None,
        current_readout=current_readout, current_loss_weight=current_loss_weight,
        raw_current_readout=raw_current_readout, raw_current_loss_weight=raw_current_loss_weight,
        raw_current_lr_multiplier=10. if raw_current_readout else None,
        raw_current_scale_lr_multiplier=20. if raw_current_readout else None,
        raw_current_source='shared_first_spatial_current_frame' if raw_current_readout else None,
        memory_readout=memory_readout, memory_window=memory_window, memory_loss_weight=memory_loss_weight,
        memory_lr_multiplier=10. if memory_readout else None,
        memory_scale_lr_multiplier=20. if memory_readout else None,
        memory_source='received_gru_states_with_invalid_left_padding' if memory_readout else None,
        memory_query_source='current2048_spatial_feature' if memory_readout else None,
        memory_future_frames=0, streaming_state_tensors=5 if memory_readout or learned_initial_state else 4,
        learned_initial_state=learned_initial_state,
        initial_hidden_lr_multiplier=10. if learned_initial_state else None,
        initial_hidden_scale_lr_multiplier=20. if learned_initial_state else None,
        initial_hidden_source='first_received_current_readout_latent' if learned_initial_state else 'zeros',
        current_readout_lr_multiplier=10. if current_readout else None,
        current_readout_scale_lr_multiplier=20. if current_readout else None,
        flip_axis_px=float(flip_axis_px),
        hidden_size=hidden_size, unidirectional=True, readout='last',
        block3_kernel=5, temporal_spatial_kernel=1, input_divisor=1.,
        preprocessing='raw_event_counts', window=train_data.shape[1], history_stride=1,
        temporal_padding='zeros_on_left_only', recurrent_history='entire_received_record',
        training_h5=str(train_data.path), training_h5_sha256=_sha256(train_data.path),
        validation_h5=str(val_data.path), validation_h5_sha256=_sha256(val_data.path),
        initialization=initialization, teacher_cache=str(teacher_cache) if teacher_cache else None,
        teacher_cache_sha256=_sha256(teacher_cache) if teacher_cache else None,
        teacher_weight=teacher_weight, teacher_max_error_px=teacher_max_error_px,
        teacher_used_for_validation=False, final_holdout_used=False, validation_selected=True,
        epochs=epochs, patience=patience, batch_size=batch_size, lr=lr, min_lr=min_lr,
        weight_decay=weight_decay, sigma=sigma, pixel_loss_weight=pixel_loss_weight,
        decode_temperature=decode_temperature, persistent_boost=persistent_boost,
        persistent_min_run=persistent_min_run, cold_boost=cold_boost, cold_frames=cold_frames,
        difficulty_weights_split='train', difficulty_error_threshold_px=3.,
        dropout=dropout, augment=augment, preload=preload, gradient_clip=gradient_clip,
        rotation_degrees=rotation_degrees, scale_jitter=scale_jitter,
        count_gain_jitter=count_gain_jitter, record_augmentation_split='train',
        record_augmentation_convention='physical_pixel_centers_rotate_scale_about_40_30',
        record_augmentation_sampling='independent_records_constant_over_time_and_polarity',
        selection_metric='unsmoothed_causal_soft_expectation_p3_all_timesteps',
        training_supervision='all received timesteps in each independent training record',
        seed=seed, threads=threads, device=str(resolved_device),
        warm_start_steps=warm_start_steps, startup_training='repeat_first_received_frame' if warm_start_steps else 'zeros',
        environment=dict(python=platform.python_version(), torch=torch.__version__,
                         numpy=np.__version__),
    )
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'config.json', config)
    history = []
    best = evaluate_streaming(model, val_loader, resolved_device, warm_start_steps, decode_temperature)
    best_epoch, no_improve = 0, 0
    history.append(dict(epoch=0, train=None, validation=best, selected=True))

    def checkpoint(epoch, completed=False):
        return dict(schema=1, model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), config=config,
                    epoch=epoch, best_epoch=best_epoch, best_validation_p3=best['p3'],
                    best_validation_error_px=best['mean_error_px'],
                    training_completed=completed, final_holdout_used=False)

    _save_checkpoint(output / 'best.pt', checkpoint(0))
    _write_json(output / 'history.json', history)
    print(json.dumps(history[-1]), flush=True)
    rng = np.random.RandomState(seed)
    try:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            rate = min_lr + .5 * (lr - min_lr) * (
                1 + math.cos(math.pi * (epoch - 1) / max(1, epochs - 1)))
            for group in optimizer.param_groups:
                group['lr'] = rate * group.get('lr_multiplier', 1.)
            model.train()
            sums = np.zeros(8)
            count = 0
            for step, (frames, labels, teacher) in enumerate(train_loader, 1):
                frames, labels, teacher = (t.to(resolved_device) for t in (frames, labels, teacher))
                original_valid = ((labels >= 0) & (labels <= 1)).all(-1)
                if augment:
                    from .offline_experiment import spatial_view
                    view = (int(rng.randint(-2, 3)), int(rng.randint(-2, 3)), bool(rng.randint(2)))
                    frames = spatial_view(frames, view)
                    labels = transform_coordinates(labels, view, flip_axis_px)
                    teacher = transform_coordinates(teacher, view, flip_axis_px)
                if any(augmentation_values):
                    from .streaming_augmentation import warp_streaming_records
                    batch = len(frames)
                    parameters = (rng.uniform(-rotation_degrees, rotation_degrees, batch),
                                  rng.uniform(1. - scale_jitter, 1. + scale_jitter, batch),
                                  rng.uniform(1. - count_gain_jitter, 1. + count_gain_jitter, batch))
                    frames, labels, teacher = warp_streaming_records(frames, labels, teacher, *parameters)
                valid = original_valid & ((labels >= 0) & (labels <= 1)).all(-1)
                if not valid.any():
                    continue
                optimizer.zero_grad(set_to_none=True)
                frames = _startup_frames(frames, warm_start_steps)
                if memory_readout:
                    x, y, current_x, current_y, memory_x, memory_y = model.forward_with_memory_logits(frames)
                    spatial = x.sum() * 0
                elif raw_current_readout:
                    x, y, current_x, current_y, raw_x, raw_y = model.forward_with_raw_logits(frames)
                    spatial = x.sum() * 0
                elif current_readout:
                    x, y, current_x, current_y = model.forward_with_current_logits(frames)
                    spatial = x.sum() * 0
                elif spatial_localizer:
                    x, y, heatmap = model.forward_with_spatial_logits(frames)
                    heatmap = heatmap[:, warm_start_steps:]
                    spatial = spatial_heatmap_loss(heatmap, labels, valid)
                    current = x.sum() * 0
                else:
                    x, y = model(frames)
                    spatial = x.sum() * 0
                    current = x.sum() * 0
                x, y = x[:, warm_start_steps:], y[:, warm_start_steps:]
                weights = streaming_frame_weights(x, y, labels, valid, temperature=decode_temperature,
                    persistent_boost=persistent_boost, min_run=persistent_min_run,
                    cold_boost=cold_boost, cold_frames=cold_frames)
                if current_readout:
                    current = masked_official_simdr_loss(current_x[:, warm_start_steps:],
                        current_y[:, warm_start_steps:], labels, valid, sigma=sigma, weights=weights)
                raw = (masked_official_simdr_loss(raw_x[:, warm_start_steps:],
                       raw_y[:, warm_start_steps:], labels, valid, sigma=sigma, weights=weights)
                       if raw_current_readout else x.sum() * 0)
                memory = (masked_official_simdr_loss(memory_x[:, warm_start_steps:],
                          memory_y[:, warm_start_steps:], labels, valid, sigma=sigma, weights=weights)
                          if memory_readout else x.sum() * 0)
                supervised = masked_official_simdr_loss(x, y, labels, valid, sigma=sigma, weights=weights)
                pixel = (weighted_coordinate_huber(x, y, labels, valid,
                            weights=weights, temperature=decode_temperature)
                         if pixel_loss_weight else supervised * 0)
                distilled = (teacher_coordinate_loss(x, y, teacher, labels, teacher_max_error_px,
                                weights=weights, temperature=decode_temperature, valid=valid)
                             if teacher_weight else supervised * 0)
                loss = (supervised + pixel_loss_weight * pixel + teacher_weight * distilled
                        + spatial_loss_weight * spatial + current_loss_weight * current
                        + raw_current_loss_weight * raw + memory_loss_weight * memory)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite streaming loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                n = int(valid.sum())
                sums += np.asarray([loss.item(), supervised.item(), pixel.item(), distilled.item(), spatial.item(), current.item(), raw.item(), memory.item()]) * n
                count += n
                if step % 64 == 0:
                    print(json.dumps(dict(epoch=epoch, batch=step, batches=len(train_loader))), flush=True)
            if not count:
                raise ValueError('No valid supervised timesteps')
            validation = evaluate_streaming(model, val_loader, resolved_device, warm_start_steps, decode_temperature)
            selected = (validation['p3'] > best['p3'] or
                        validation['p3'] == best['p3'] and validation['mean_error_px'] < best['mean_error_px'])
            if selected:
                best, best_epoch, no_improve = validation, epoch, 0
                _save_checkpoint(output / 'best.pt', checkpoint(epoch))
            else:
                no_improve += 1
            history.append(dict(epoch=epoch, learning_rate=rate,
                                train=dict(zip(('loss', 'simdr_loss', 'pixel_huber_loss', 'teacher_huber_loss', 'spatial_heatmap_loss', 'current_readout_loss', 'raw_current_readout_loss', 'memory_readout_loss'),
                                               (sums / count).tolist())),
                                training_valid_timesteps=count, validation=validation,
                                selected=selected, seconds=time.perf_counter() - started))
            _write_json(output / 'history.json', history)
            _save_checkpoint(output / 'last.pt', checkpoint(epoch, epoch == epochs or no_improve >= patience))
            print(json.dumps(history[-1]), flush=True)
            if no_improve >= patience:
                break
    finally:
        train_data.close()
        val_data.close()
    result = dict(best_epoch=best_epoch, metrics=best, checkpoint=str(output / 'best.pt'),
                  final_holdout_used=False)
    _write_json(output / 'summary.json', result)
    return result
