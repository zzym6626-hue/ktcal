"""Subject-disjoint AIS sequence chunks, masked losses, and STNet training."""
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from models.STNet import STNet

from .data import WindowDataset, load_manifest
from .train_hard_frames import hard_frame_weights


class FlatH5ChunkDataset(Dataset):
    """Lazy fixed-window view over released flat 3ET+2025 HDF5 chunks."""

    def __init__(self, path, window=None, input_divisor=1.0):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f'Flat HDF5 file not found: {self.path}')
        if input_divisor <= 0:
            raise ValueError('Flat HDF5 input divisor must be positive')
        self.input_divisor = float(input_divisor)
        self._handle = None
        with h5py.File(self.path, 'r') as handle:
            label_key = 'label' if 'label' in handle else 'labels'
            if 'frames' not in handle or label_key not in handle:
                raise ValueError('Flat HDF5 requires frames and label datasets')
            frames_shape = tuple(handle['frames'].shape)
            labels_shape = tuple(handle[label_key].shape)
        if (len(frames_shape) != 5 or frames_shape[2:] != (2, 60, 80)
                or frames_shape[0] < 1 or frames_shape[1] < 1):
            raise ValueError(
                'Flat HDF5 frames must have shape [N,T,2,60,80], '
                f'got {frames_shape}')
        if (len(labels_shape) != 3 or labels_shape[:2] != frames_shape[:2]
                or labels_shape[-1] < 2):
            raise ValueError(
                'Flat HDF5 labels must have shape [N,T,at least 2] '
                'and match frames')
        if window is None:
            window = frames_shape[1]
        if window < 1:
            raise ValueError('Flat HDF5 window must be positive')
        self.frames_shape = frames_shape
        self.labels_shape = labels_shape
        self.label_key = label_key
        self.window = int(window)
        self.chunks = [
            (sequence, start, min(self.window, frames_shape[1] - start))
            for sequence in range(frames_shape[0])
            for start in range(0, frames_shape[1], self.window)
        ]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        if self._handle is None:
            self._handle = h5py.File(
                self.path, 'r', rdcc_nbytes=32 * 1024 * 1024)
        sequence, start, real_length = self.chunks[index]
        stop = start + real_length
        frames = self._handle['frames'][sequence, start:stop].astype(np.float32)
        frames /= self.input_divisor
        labels = self._handle[self.label_key][sequence, start:stop, :2].astype(
            np.float32)
        valid = np.ones(real_length, dtype=bool)
        padding = self.window - real_length
        if padding:
            frames = np.concatenate(
                (frames, np.repeat(frames[-1:], padding, axis=0)))
            labels = np.concatenate(
                (labels, np.repeat(labels[-1:], padding, axis=0)))
            valid = np.concatenate((valid, np.zeros(padding, dtype=bool)))
        return (torch.from_numpy(frames), torch.from_numpy(labels),
                torch.from_numpy(valid))

    def close(self):
        if getattr(self, '_handle', None) is not None:
            self._handle.close()
            self._handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handle'] = None
        return state

    def __del__(self):
        self.close()


class FlatH5CausalWindowDataset(FlatH5ChunkDataset):
    """Historical windows, padded on the left like the online tracker.

    Only the target frame is labelled: preceding frames are context, not
    supervised predictions from a bidirectional model.
    """

    def __init__(self, path, window=16, stride=1, input_divisor=1.0,
                 history_stride=1, preload_frames=False):
        if stride < 1:
            raise ValueError('Causal stride must be positive')
        if isinstance(history_stride, bool) or not isinstance(history_stride, int) or history_stride < 1:
            raise ValueError('Positive integer history stride required')
        super().__init__(path, window=window, input_divisor=input_divisor)
        self.stride = int(stride)
        self.history_stride = history_stride
        self._frames_cache = self._labels_cache = None
        if preload_frames:
            with h5py.File(self.path, 'r') as handle:
                self._frames_cache = handle['frames'][:]
                self._labels_cache = handle[self.label_key][:]
        self.targets = [
            (sequence, target)
            for sequence in range(self.frames_shape[0])
            for target in range(0, self.frames_shape[1], self.stride)
        ]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if self._frames_cache is None and self._handle is None:
            self._handle = h5py.File(
                self.path, 'r', rdcc_nbytes=32 * 1024 * 1024)
        sequence, target = self.targets[index]
        start = max(0, target - (self.window - 1) * self.history_stride)
        frame_data = self._frames_cache if self._frames_cache is not None else self._handle['frames']
        label_data = self._labels_cache if self._labels_cache is not None else self._handle[self.label_key]
        frames = frame_data[sequence, start:target + 1].astype(np.float32)
        if self.history_stride != 1:
            indices = np.maximum(0, target - np.arange(self.window - 1, -1, -1)
                                 * self.history_stride) - start
            frames = frames[indices]
        frames /= self.input_divisor
        if len(frames) < self.window:
            frames = np.concatenate((
                np.repeat(frames[:1], self.window - len(frames), axis=0),
                frames))
        labels = label_data[
            sequence, target:target + 1, :2].astype(np.float32)
        return (torch.from_numpy(frames), torch.from_numpy(labels),
                torch.ones(1, dtype=torch.bool))


class RecordChunkDataset(Dataset):
    """Non-overlapping fixed windows that never cross recording boundaries."""

    def __init__(self, manifest_path, split, window=64, chunk_indices=None):
        if window < 1:
            raise ValueError('Window must be positive')
        self.path = Path(manifest_path).resolve()
        self.manifest = load_manifest(self.path)
        if isinstance(split, (list, tuple)):
            recordings = [str(name) for name in split]
        elif split in self.manifest['splits']:
            recordings = list(self.manifest['splits'][split])
        else:
            raise ValueError(f'Unknown split: {split}')
        self.h5_path = self.path.parent / self.manifest['h5']
        self.window = window
        self._handle = None
        chunks = self.enumerate_chunks(self.h5_path, recordings, window)
        if chunk_indices is not None:
            chunks = [chunks[int(index)] for index in chunk_indices]
            if not chunks:
                raise ValueError('Empty chunk subset')
        self.chunks = chunks

    @staticmethod
    def enumerate_chunks(h5_path, recordings, window):
        """List non-overlapping chunks for recordings in a stable order."""
        chunks = []
        with h5py.File(h5_path, 'r') as handle:
            for name in recordings:
                count = len(handle[name]['frames'])
                if count < 1:
                    raise ValueError(f'Empty recording: {name}')
                for start in range(0, count, window):
                    chunks.append((name, start, min(window, count - start)))
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        if self._handle is None:
            self._handle = h5py.File(
                self.h5_path, 'r', rdcc_nbytes=16 * 1024 * 1024)
        name, start, real_length = self.chunks[index]
        group = self._handle[name]
        stop = start + real_length
        frames = group['frames'][start:stop].astype(np.float32) / 15
        labels = group['labels'][start:stop].astype(np.float32)
        valid = group['valid'][start:stop].astype(np.float32)
        padding = self.window - real_length
        if padding:
            frames = np.concatenate((frames, np.repeat(frames[-1:], padding, axis=0)))
            labels = np.concatenate((labels, np.repeat(labels[-1:], padding, axis=0)))
            valid = np.concatenate((valid, np.zeros(padding, dtype=np.float32)))
        labels = np.column_stack((labels, valid))
        return torch.from_numpy(frames), torch.from_numpy(labels)

    def close(self):
        if getattr(self, '_handle', None) is not None:
            self._handle.close()
            self._handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handle'] = None
        return state

    def __del__(self):
        self.close()


def random_chunk_split(manifest_path, window, val_ratio=0.1, seed=42):
    """Hold out a random fraction of chunks drawn from every recording.

    This mirrors a same-distribution (random) holdout instead of the
    recording/subject-disjoint protocol: validation chunks reuse the same
    recordings as training, so the number is an in-distribution estimate.
    """
    if not 0 < val_ratio < 1:
        raise ValueError('Validation ratio must be inside (0, 1)')
    path = Path(manifest_path).resolve()
    manifest = load_manifest(path)
    h5_path = path.parent / manifest['h5']
    recordings = [name
                  for split in ('train', 'val', 'holdout')
                  for name in manifest['splits'][split]]
    chunks = RecordChunkDataset.enumerate_chunks(h5_path, recordings, window)
    order = list(range(len(chunks)))
    random.Random(seed).shuffle(order)
    val_count = max(1, int(round(len(chunks) * val_ratio)))
    if val_count >= len(chunks):
        raise ValueError('Not enough chunks for a random split')
    val_indices = sorted(order[:val_count])
    train_indices = sorted(order[val_count:])
    return recordings, train_indices, val_indices, chunks


def simdr_targets(normalized_coordinates, bins, sigma=1.0):
    if bins < 1 or sigma <= 0:
        raise ValueError('Require positive bins and sigma')
    grid = torch.arange(
        bins, device=normalized_coordinates.device,
        dtype=normalized_coordinates.dtype)
    center = normalized_coordinates.unsqueeze(-1) * bins
    target = torch.exp(-0.5 * ((grid - center) / sigma).square())
    return target / target.sum(-1, keepdim=True).clamp_min(
        torch.finfo(target.dtype).tiny)


def decode_soft_expectation(logits_x, logits_y):
    if logits_x.shape[:-1] != logits_y.shape[:-1]:
        raise ValueError('X and Y logits must have matching batch/time dimensions')
    probability_x = torch.softmax(logits_x, dim=-1)
    probability_y = torch.softmax(logits_y, dim=-1)
    grid_x = torch.arange(
        logits_x.shape[-1], device=logits_x.device, dtype=logits_x.dtype)
    grid_y = torch.arange(
        logits_y.shape[-1], device=logits_y.device, dtype=logits_y.dtype)
    coordinate_x = (probability_x * grid_x).sum(-1) / logits_x.shape[-1]
    coordinate_y = (probability_y * grid_y).sum(-1) / logits_y.shape[-1]
    return torch.stack((coordinate_x, coordinate_y), dim=-1)


def _validated_coordinates(logits_x, logits_y, labels, valid):
    if logits_x.ndim != 3 or logits_y.ndim != 3:
        raise ValueError('Expected B x T x bins logits')
    if logits_x.shape[:2] != logits_y.shape[:2] or labels.shape[:2] != logits_x.shape[:2]:
        raise ValueError('Logits and labels must have matching batch/time dimensions')
    if labels.shape[-1] < 2 or valid.shape != logits_x.shape[:2]:
        raise ValueError('Expected coordinate labels and a B x T validity mask')
    valid = valid.bool()
    if not torch.any(valid):
        raise ValueError('At least one valid timestep is required')
    return labels[..., :2], valid


def masked_simdr_loss(logits_x, logits_y, labels, valid, sigma=1.0):
    coordinates, valid = _validated_coordinates(
        logits_x, logits_y, labels, valid)
    target_x = simdr_targets(coordinates[..., 0], logits_x.shape[-1], sigma)
    target_y = simdr_targets(coordinates[..., 1], logits_y.shape[-1], sigma)
    loss_x = -(target_x * F.log_softmax(logits_x, dim=-1)).sum(-1)
    loss_y = -(target_y * F.log_softmax(logits_y, dim=-1)).sum(-1)
    return (0.5 * (loss_x + loss_y))[valid].mean()


def _official_simdr_target(normalized_coordinates, bins, sigma):
    if sigma <= 0:
        raise ValueError('Official SimDR sigma must be positive')
    grid = torch.arange(
        bins, device=normalized_coordinates.device,
        dtype=normalized_coordinates.dtype)
    centre = normalized_coordinates.unsqueeze(-1) * bins
    target = torch.exp(-0.5 * ((grid - centre) / sigma).square())
    minimum = target.amin(dim=-1, keepdim=True)
    maximum = target.amax(dim=-1, keepdim=True)
    return (target - minimum) / (maximum - minimum).clamp_min(
        torch.finfo(target.dtype).tiny)


def masked_official_simdr_loss(logits_x, logits_y, labels, valid, sigma=1.0,
                               weights=None):
    """Vectorized Gaussian SimDR KL objective."""
    coordinates, valid = _validated_coordinates(
        logits_x, logits_y, labels, valid)
    # Mask before generating targets: a NaN target has NaN KL derivatives
    # even when its loss is discarded after computation.
    coordinates = coordinates[valid]
    if not bool(torch.isfinite(coordinates).all()):
        raise ValueError('Valid SimDR coordinates must be finite')
    target_x = _official_simdr_target(
        coordinates[..., 0], logits_x.shape[-1], sigma)
    target_y = _official_simdr_target(
        coordinates[..., 1], logits_y.shape[-1], sigma)
    loss_x = F.kl_div(
        F.log_softmax(logits_x[valid], dim=-1), target_x,
        reduction='none').mean(-1)
    loss_y = F.kl_div(
        F.log_softmax(logits_y[valid], dim=-1), target_y,
        reduction='none').mean(-1)
    per_frame = loss_x + loss_y
    if weights is None:
        return per_frame.mean()
    if weights.shape != valid.shape:
        raise ValueError('Training weights must match the valid mask')
    selected = weights.to(device=per_frame.device, dtype=per_frame.dtype)[valid]
    if (not bool(torch.isfinite(selected).all()) or bool((selected <= 0).any())):
        raise ValueError('Training weights must be finite and positive on valid frames')
    return (per_frame * selected).sum() / selected.sum()


def masked_distillation_loss(student_x, student_y, teacher_x, teacher_y,
                             valid, temperature=2.0):
    """Temperature-scaled x/y logit distillation over valid timesteps."""
    if temperature <= 0:
        raise ValueError('Distillation temperature must be positive')
    if (student_x.shape != teacher_x.shape
            or student_y.shape != teacher_y.shape
            or student_x.shape[:2] != student_y.shape[:2]
            or valid.shape != student_x.shape[:2]):
        raise ValueError('Student, teacher and validity shapes must match')
    valid = valid.bool()
    if not torch.any(valid):
        raise ValueError('At least one valid timestep is required')
    scale = temperature * temperature

    def divergence(student, teacher):
        return F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            F.softmax(teacher.detach() / temperature, dim=-1),
            reduction='none').sum(-1) * scale

    return (0.5 * (divergence(student_x, teacher_x)
                   + divergence(student_y, teacher_y)))[valid].mean()


def masked_feature_distillation_loss(student_features, teacher_features,
                                     valid):
    """Cosine feature distillation over valid timesteps."""
    if student_features.shape != teacher_features.shape:
        raise ValueError('Student and teacher feature shapes must match')
    if student_features.ndim != 3:
        raise ValueError('Features must have shape [batch, sequence, channels]')
    if valid.shape != student_features.shape[:2]:
        raise ValueError('Feature validity mask must match batch and sequence')
    valid = valid.bool()
    if not torch.any(valid):
        raise ValueError('At least one valid timestep is required')
    student = F.normalize(student_features, dim=-1)
    teacher = F.normalize(teacher_features.detach(), dim=-1)
    cosine = (student * teacher).sum(dim=-1)
    return (1.0 - cosine)[valid].mean()


def masked_pixel_loss(logits_x, logits_y, labels, valid, beta=1.0):
    if beta <= 0:
        raise ValueError('Pixel Huber beta must be positive')
    coordinates, valid = _validated_coordinates(
        logits_x, logits_y, labels, valid)
    prediction = decode_soft_expectation(logits_x, logits_y)
    scale = prediction.new_tensor([logits_x.shape[-1], logits_y.shape[-1]])
    loss = F.smooth_l1_loss(
        prediction * scale, coordinates * scale,
        beta=beta, reduction='none').mean(-1)
    return loss[valid].mean()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def _save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    torch.save(value, temporary)
    temporary.replace(path)


def _configure(seed, threads, device):
    if threads < 1:
        raise ValueError('Threads must be positive')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if device != 'cpu' and not device.startswith('cuda'):
        raise ValueError('Only CPU and CUDA are supported')
    return torch.device(device)


@torch.inference_mode()
def evaluate_stnet(model, loader, device, mode='last_step'):
    """Score a loader by last-step or all-timestep soft-expectation error."""
    if mode not in ('last_step', 'all_steps'):
        raise ValueError('Evaluation mode must be last_step or all_steps')
    model.eval()
    errors = []
    for frames, labels in loader:
        frames = frames.to(device)
        logits_x, logits_y = model(frames)
        scale = labels.new_tensor([80.0, 60.0]).to(device)
        if mode == 'last_step':
            prediction = decode_soft_expectation(
                logits_x[:, -1:], logits_y[:, -1:])[:, 0]
            target = labels[:, -1, :2].to(device)
            errors.append(torch.linalg.vector_norm(
                (prediction - target) * scale, dim=-1).cpu())
        else:
            prediction = decode_soft_expectation(logits_x, logits_y)
            labels = labels.to(device)
            target = labels[..., :2]
            valid = labels[..., 3].bool()
            if not bool(valid.any()):
                continue
            distance = torch.linalg.vector_norm(
                (prediction - target) * scale, dim=-1)
            errors.append(distance[valid].cpu())
    if not errors:
        raise ValueError('Validation split has no valid windows')
    errors = torch.cat(errors).numpy()
    return {
        'frames': int(len(errors)),
        'mean_error_px': float(errors.mean()),
        'median_error_px': float(np.median(errors)),
        'p3': float(np.mean(errors <= 3)),
        'p5': float(np.mean(errors <= 5)),
        'p10': float(np.mean(errors <= 10)),
    }


def train_stnet(manifest_path, output, *, epochs=30, patience=8, window=64,
                batch_size=8, val_stride=16, lr=0.002,
                weight_decay=0.01, sigma=1.0, pixel_weight=0.1,
                pixel_beta=1.0, min_delta=0.01, gradient_clip=5.0,
                temporal_spatial_kernel=1, split_mode='record',
                val_ratio=0.1, scheduler='none', min_lr=None,
                selection_metric='last_step', initialize_from=None,
                seed=42, threads=4, device='auto'):
    """Train heatmap STNet from scratch and select only on validation subjects."""
    if min(epochs, patience, window, batch_size, val_stride) < 1:
        raise ValueError('Epochs, patience, window, batch size and stride must be positive')
    if lr <= 0 or weight_decay < 0 or sigma <= 0 or pixel_weight < 0:
        raise ValueError('Invalid optimizer or loss configuration')
    if min_delta < 0 or gradient_clip <= 0:
        raise ValueError('Invalid selection or gradient clipping configuration')
    if temporal_spatial_kernel not in (1, 3):
        raise ValueError('Temporal spatial kernel must be 1 or 3')
    if split_mode not in ('record', 'random'):
        raise ValueError('Split mode must be record or random')
    if split_mode == 'random' and not 0 < val_ratio < 1:
        raise ValueError('Validation ratio must be inside (0, 1)')
    if scheduler not in ('none', 'cosine'):
        raise ValueError('Learning-rate scheduler must be none or cosine')
    if selection_metric not in ('last_step', 'all_steps'):
        raise ValueError('Selection metric must be last_step or all_steps')
    if min_lr is None:
        min_lr = lr * 0.1
    if min_lr <= 0 or min_lr > lr:
        raise ValueError('Minimum learning rate must be inside (0, lr]')
    manifest_path = Path(manifest_path).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    manifest = load_manifest(manifest_path)
    resolved_device = _configure(seed, threads, device)
    if split_mode == 'random':
        recordings, train_indices, val_indices, chunks = random_chunk_split(
            manifest_path, window, val_ratio=val_ratio, seed=seed)
        train_data = RecordChunkDataset(
            manifest_path, recordings, window=window,
            chunk_indices=train_indices)
        val_data = RecordChunkDataset(
            manifest_path, recordings, window=window,
            chunk_indices=val_indices)
        val_recordings = sorted({chunks[index][0] for index in val_indices})
    else:
        train_data = RecordChunkDataset(manifest_path, 'train', window=window)
        val_data = WindowDataset(
            manifest_path, 'val', window=window, stride=val_stride)
        val_recordings = list(manifest['splits']['val'])
    if not len(train_data) or not len(val_data):
        raise ValueError('Train and validation splits must be nonempty')
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=generator)
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    args = SimpleNamespace(
        stnet_block3_kernel=5, stnet_unidirectional=False,
        stnet_decode='soft',
        stnet_temporal_spatial_kernel=temporal_spatial_kernel)
    model = STNet(args).to(resolved_device)
    initialization = 'random'
    if initialize_from is not None:
        source = torch.load(Path(initialize_from).resolve(),
                            map_location='cpu', weights_only=False)
        if 'model_state_dict' in source:
            source = source['model_state_dict']
        try:
            model.load_state_dict(source, strict=True)
        except RuntimeError as error:
            raise ValueError(f'Incompatible STNet initialization: {error}') from error
        initialization = {
            'source': str(Path(initialize_from).resolve()),
            'mapped_parameters': len(source),
            'retrained': True}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    config = {
        'epochs': epochs, 'patience': patience, 'window': window,
        'batch_size': batch_size, 'val_stride': val_stride, 'lr': lr,
        'weight_decay': weight_decay, 'sigma': sigma,
        'pixel_weight': pixel_weight, 'pixel_beta': pixel_beta,
        'min_delta': min_delta, 'gradient_clip': gradient_clip,
        'temporal_spatial_kernel': temporal_spatial_kernel,
        'seed': seed, 'threads': threads, 'device': device,
        'resolved_device': str(resolved_device),
        'manifest': str(manifest_path),
        'manifest_sha256': _sha256(manifest_path),
        'dataset': manifest['dataset'],
        'architecture': (
            'models.STNet.STNet kernel5 bidirectional '
            f'temporal3x{temporal_spatial_kernel}x{temporal_spatial_kernel}'),
        'initialization': initialization,
        'training_supervision': 'all valid timesteps in non-overlapping per-record chunks',
        'selection_metric': selection_metric,
        'selection_metric_description': (
            'validation soft-expectation mean pixel error over '
            + ('all valid timesteps' if selection_metric == 'all_steps'
               else 'the last timestep')),
        'train_chunks': len(train_data), 'validation_windows': len(val_data),
        'split_mode': split_mode,
        'random_val_ratio': (val_ratio if split_mode == 'random' else None),
        'random_split_seed': (seed if split_mode == 'random' else None),
        'training_recordings': (
            list(recordings) if split_mode == 'random'
            else manifest['splits']['train']),
        'validation_recordings': val_recordings,
        'split_policy': (
            'random chunk holdout across all recordings'
            if split_mode == 'random'
            else 'recording and subject disjoint'),
        'scheduler': scheduler, 'min_lr': min_lr,
        'holdout_used': False,
        'environment': {
            'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'platform': platform.platform(),
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'config.json', config)
    history = []
    initial = evaluate_stnet(
        model, val_loader, resolved_device, mode=selection_metric)
    best_error = initial['mean_error_px']
    best_epoch = 0
    history.append({
        'epoch': 0, 'train': None,
        'validation_error_px': best_error,
        'validation': initial, 'selected': True, 'seconds': 0.0,
    })

    def checkpoint(epoch):
        return {
            'schema': 1, 'model_state_dict': model.state_dict(),
            'epoch': epoch, 'best_validation_error_px': best_error,
            'config': config, 'manifest_sha256': config['manifest_sha256'],
            'selection_split': 'val', 'holdout_used': False,
            'training_recordings': config['training_recordings'],
            'validation_recordings': config['validation_recordings'],
        }

    _save_checkpoint(output / 'best.pt', checkpoint(0))
    no_improve = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        if scheduler == 'cosine':
            progress = (epoch - 1) / max(1, epochs - 1)
            rate = min_lr + 0.5 * (lr - min_lr) * (
                1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group['lr'] = rate
        learning_rate = optimizer.param_groups[0]['lr']
        model.train()
        sums = np.zeros(3, dtype=np.float64)
        samples = 0
        for frames, labels in train_loader:
            frames = frames.to(resolved_device)
            labels = labels.to(resolved_device)
            valid = labels[..., 3].bool()
            optimizer.zero_grad(set_to_none=True)
            logits_x, logits_y = model(frames)
            simdr = masked_simdr_loss(
                logits_x, logits_y, labels, valid, sigma=sigma)
            pixel = masked_pixel_loss(
                logits_x, logits_y, labels, valid, beta=pixel_beta)
            loss = simdr + pixel_weight * pixel
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite STNet training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            count = int(valid.sum())
            sums += np.array([loss.item(), simdr.item(), pixel.item()]) * count
            samples += count
        if samples < 1:
            raise ValueError('Training split has no valid labels')
        validation = evaluate_stnet(
            model, val_loader, resolved_device, mode=selection_metric)
        error = validation['mean_error_px']
        selected = error < best_error - min_delta
        if selected:
            best_error = error
            best_epoch = epoch
            no_improve = 0
            _save_checkpoint(output / 'best.pt', checkpoint(epoch))
        else:
            no_improve += 1
        row = {
            'epoch': epoch,
            'learning_rate': learning_rate,
            'train': dict(zip(
                ('loss', 'simdr_loss', 'pixel_huber_loss'),
                (sums / samples).tolist())),
            'training_valid_timesteps': samples,
            'validation_error_px': error,
            'validation': validation, 'selected': selected,
            'seconds': time.perf_counter() - started,
        }
        history.append(row)
        _write_json(output / 'history.json', history)
        print(json.dumps(row, allow_nan=False), flush=True)
        if no_improve >= patience:
            break
    _write_json(output / 'history.json', history)
    summary = {
        'status': 'complete', 'checkpoint': 'best.pt',
        'checkpoint_sha256': _sha256(output / 'best.pt'),
        'best_epoch': best_epoch,
        'best_validation_error_px': best_error,
        'epochs_completed': len(history) - 1,
        'train_chunks': len(train_data),
        'validation_windows': len(val_data),
        'holdout_used_for_model_selection': False,
    }
    _write_json(output / 'summary.json', summary)
    train_data.close()
    val_data.close()
    return output / 'best.pt'


def _checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must contain a state dictionary')
    for key in ('model_state_dict', 'model_state', 'state_dict'):
        if key in checkpoint:
            state = checkpoint[key]
            break
    else:
        state = checkpoint
    if not state or not all(isinstance(key, str) and torch.is_tensor(value)
                            for key, value in state.items()):
        raise ValueError('Checkpoint does not contain model weights')
    return state


def _initialize_3et_student(model, checkpoint_path, temporal_spatial_kernel,
                            transplant_gain, input_divisor):
    """Initialize from our own STNet checkpoint, or start from scratch."""
    if checkpoint_path is None:
        return {'source_architecture': 'STNet', 'source': None,
                'mapped_parameters': 0, 'retrained': True}
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    state = _checkpoint_state_dict(checkpoint)
    if not any(key.startswith('block1.') for key in state):
        raise ValueError('Only STNet checkpoints can initialize STNet training')
    source_config = checkpoint.get('config', {})
    source_divisor = float(source_config.get('input_divisor', 1.0))
    if source_divisor != float(input_divisor):
        raise ValueError(
            'STNet initialization input divisor does not match requested '
            f'training: {source_divisor} != {input_divisor}')
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f'Incompatible STNet initialization: {error}') from error
    return {
        'source': str(checkpoint_path), 'source_sha256': _sha256(checkpoint_path),
        'source_architecture': 'STNet', 'mapped_parameters': len(state),
        'retrained': True,
    }


@torch.inference_mode()
def evaluate_flat_stnet(model, loader, device, causal_last_step=False,
                        offline_ma_window=None, decode='hard'):
    """Hard-argmax metrics, or per-record soft+MA offline checkpoint selection."""
    if offline_ma_window is not None and (causal_last_step or offline_ma_window < 1
                                          or offline_ma_window % 2 == 0):
        raise ValueError('Offline MA window must be positive/odd and non-causal')
    model.eval()
    if decode not in ('hard', 'soft'):
        raise ValueError('Validation decode must be hard or soft')
    errors = []
    offline_predictions = []
    offline_labels = []
    for frames, labels, valid in loader:
        logits_x, logits_y = model(frames.to(device))
        if offline_ma_window is not None:
            # Keep checkpoint selection bit-for-bit aligned with the released
            # inference pipeline: NumPy soft decode and independent MA for
            # each recording. Never average across batch/record boundaries.
            from .combined_3et import SCALE, moving_average_sequence_predictions
            from .evaluation_protocol import soft_expectation_decode
            # Follow precompute_global_predictions and the combined evaluator's
            # pixel -> normalized float32 roundtrip before temporal smoothing.
            pixel_predictions = (soft_expectation_decode(logits_x, logits_y)
                                 .astype(np.float32) * SCALE)
            predictions = moving_average_sequence_predictions(
                list(pixel_predictions / SCALE),
                offline_ma_window)
            for predicted, expected, mask in zip(predictions, labels.numpy(),
                                                  valid.numpy()):
                if mask.any():
                    offline_predictions.append(predicted[mask])
                    offline_labels.append(expected[mask])
            continue
        if causal_last_step:
            logits_x = logits_x[:, -1:]
            logits_y = logits_y[:, -1:]
        prediction = (decode_soft_expectation(logits_x, logits_y) if decode == 'soft' else torch.stack((
            logits_x.argmax(-1).to(logits_x.dtype) / logits_x.shape[-1],
            logits_y.argmax(-1).to(logits_y.dtype) / logits_y.shape[-1],
        ), dim=-1))
        labels = labels.to(device)
        valid = valid.to(device).bool()
        if not bool(valid.any()):
            continue
        scale = prediction.new_tensor(
            [float(logits_x.shape[-1]), float(logits_y.shape[-1])])
        distance = torch.linalg.vector_norm(
            (prediction - labels) * scale, dim=-1)
        errors.append(distance[valid].cpu())
    if offline_ma_window is not None:
        if not offline_predictions:
            raise ValueError('Validation HDF5 has no valid frames')
        from .evaluation_protocol import summarize_pixel_errors
        metrics = summarize_pixel_errors(
            np.concatenate(offline_predictions), np.concatenate(offline_labels))
        return {key: metrics[key] for key in (
            'frames', 'mean_error_px', 'median_error_px', 'p3', 'p5', 'p10')}
    if not errors:
        raise ValueError('Validation HDF5 has no valid frames')
    errors = torch.cat(errors).numpy()
    return {
        'frames': int(len(errors)),
        'mean_error_px': float(errors.mean()),
        'median_error_px': float(np.median(errors)),
        'p3': float(np.mean(errors < 3)),
        'p5': float(np.mean(errors < 5)),
        'p10': float(np.mean(errors < 10)),
    }


def select_3et_checkpoint(p3, error, best_p3, best_error, *,
                          causal_last_step=False, min_delta=0.001,
                          selection_metric='hard_argmax'):
    """Choose checkpoints by P3, with mean error as a tie breaker."""
    if selection_metric == 'offline_soft_ma11':
        min_delta = 0.
    return p3 > best_p3 or (p3 == best_p3 and error < best_error - min_delta)


def train_stnet_3et(train_h5, val_h5, initialize_from=None, output=None, *,
                    teacher_checkpoint=None, epochs=60, patience=12,
                    window=100, batch_size=8, lr=2e-4, min_lr=2e-5,
                    weight_decay=1e-4, sigma=1.0, distill_weight=0.,
                    distill_temperature=2.0, feature_distill_weight=0.0,
                    min_delta=0.001,
                    gradient_clip=5.0, temporal_spatial_kernel=1,
                    transplant_gain=1.0, input_divisor=1.0,
                    seed=42, threads=4, device='auto',
                    causal_last_step=False, train_stride=8, readout='last',
                    selection_metric='hard_argmax', persistent_boost=0.,
                    sparse_boost=0., history_stride=1, preload_frames=False,
                    pixel_loss_weight=0.):
    """Train our STNet on labeled 3ET train/validation HDF5."""
    if output is None:
        raise ValueError("Training output is required")
    if teacher_checkpoint is not None or distill_weight or feature_distill_weight:
        raise ValueError("External teacher distillation is not available")
    if min(epochs, patience, window, batch_size, train_stride) < 1:
        raise ValueError('Epochs, patience, window, batch size and stride must be positive')
    if not np.isfinite(pixel_loss_weight) or pixel_loss_weight < 0:
        raise ValueError('Pixel loss weight must be finite and nonnegative')
    if (lr <= 0 or min_lr <= 0 or min_lr > lr or weight_decay < 0
            or sigma <= 0 or distill_weight < 0
            or feature_distill_weight < 0
            or distill_temperature <= 0 or min_delta < 0
            or gradient_clip <= 0 or input_divisor <= 0):
        raise ValueError('Invalid 3ET fine-tuning configuration')
    if temporal_spatial_kernel not in (1, 3):
        raise ValueError('Temporal spatial kernel must be 1 or 3')
    if readout not in ('last', 'final_hidden', 'attention'):
        raise ValueError('STNet readout must be last, final_hidden, or attention')
    if (isinstance(history_stride, bool) or not isinstance(history_stride, int)
            or history_stride < 1 or (history_stride != 1 and not causal_last_step)):
        raise ValueError('History stride is positive and requires causal training')
    if selection_metric not in ('hard_argmax', 'offline_soft_ma11', 'causal_soft'):
        raise ValueError('Unknown 3ET checkpoint selection metric')
    if selection_metric == 'causal_soft' and not causal_last_step:
        raise ValueError('Causal soft selection requires past-only current-frame training')
    if (not np.isfinite(persistent_boost) or persistent_boost < 0
            or not np.isfinite(sparse_boost) or sparse_boost < 0):
        raise ValueError('Training hard-frame boosts must be nonnegative and finite')
    if selection_metric == 'offline_soft_ma11' and (causal_last_step or window != 100):
        raise ValueError('Offline soft+MA11 selection requires non-causal 100-frame chunks')
    if causal_last_step and (persistent_boost or sparse_boost):
        raise ValueError('Training hard-frame boosts require offline sequences')
    train_h5 = Path(train_h5).resolve()
    val_h5 = Path(val_h5).resolve()
    output = Path(output).resolve()
    if train_h5 == val_h5:
        raise ValueError('Training and validation HDF5 files must differ')
    if train_h5.name.lower() == 'test_200_200.h5':
        raise ValueError('Refusing to read the final holdout for training')
    if causal_last_step or selection_metric == 'offline_soft_ma11':
        # An exploratory validation run must never tune against final holdout.
        from .combined_3et import validate_validation_h5
        validate_validation_h5(val_h5)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    # Check metadata before constructing a training Dataset (which may read
    # labels): a renamed official holdout must not become training input.
    from .combined_3et import validate_validation_h5
    validate_validation_h5(train_h5)
    resolved_device = _configure(seed, threads, device)
    if causal_last_step:
        train_data = FlatH5CausalWindowDataset(
            train_h5, window=window, stride=train_stride,
            input_divisor=input_divisor, history_stride=history_stride,
            preload_frames=preload_frames)
        val_data = FlatH5CausalWindowDataset(
            val_h5, window=window, stride=1,
            input_divisor=input_divisor, history_stride=history_stride,
            preload_frames=preload_frames)
    else:
        train_data = FlatH5ChunkDataset(
            train_h5, window=window, input_divisor=input_divisor)
        val_data = FlatH5ChunkDataset(
            val_h5, window=window, input_divisor=input_divisor)
        if (selection_metric == 'offline_soft_ma11'
                and val_data.frames_shape[1] != 100):
            raise ValueError('Offline selection requires full 100-frame validation chunks')
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=generator)
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    args = SimpleNamespace(
        stnet_block3_kernel=5, stnet_unidirectional=False,
        stnet_decode='argmax',
        stnet_readout=readout,
        stnet_temporal_spatial_kernel=temporal_spatial_kernel)
    model = STNet(args).to(resolved_device)
    initialization = _initialize_3et_student(
        model, initialize_from, temporal_spatial_kernel,
        transplant_gain, input_divisor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    config = {
        'schema': 1,
        'dataset': '3ET+2025',
        'training_h5': str(train_h5),
        'training_h5_sha256': _sha256(train_h5),
        'validation_h5': str(val_h5),
        'validation_h5_sha256': _sha256(val_h5),
        'epochs': epochs, 'patience': patience, 'window': window,
        'readout': readout,
        'batch_size': batch_size, 'lr': lr, 'min_lr': min_lr,
        'weight_decay': weight_decay, 'sigma': sigma,
        'distill_weight': distill_weight,
        'pixel_loss_weight': float(pixel_loss_weight),
        'hard_frame_persistent_boost': float(persistent_boost),
        'hard_frame_sparse_boost': float(sparse_boost),
        'hard_frame_min_run': 5,
        'hard_frame_source': 'training_batches_only',
        'distill_temperature': distill_temperature,
        'feature_distill_weight': feature_distill_weight,
        'teacher_checkpoint': None,
        'teacher_checkpoint_sha256': None,
        'min_delta': min_delta, 'gradient_clip': gradient_clip,
        'selection_tiebreak_delta': (
            0. if selection_metric == 'offline_soft_ma11' else min_delta),
        'temporal_spatial_kernel': temporal_spatial_kernel,
        'input_divisor': float(input_divisor),
        'preprocessing': (
            'raw_event_counts' if input_divisor == 1
            else f'event_counts_divided_by_{input_divisor:g}'),
        'architecture': (
            'models.STNet.STNet kernel5 bidirectional '
            f'temporal3x{temporal_spatial_kernel}x{temporal_spatial_kernel}'),
        'initialization': initialization,
        'training_supervision': (
            'Gaussian SimDR KL over every valid timestep '
            'without external teacher distillation'),
        'selection_metric': ('soft_expectation_ma11_80x60_p3_all_valid_steps'
                             if selection_metric == 'offline_soft_ma11'
                             else 'hard_argmax_80x60_p3_all_valid_steps'),
        'selection_split': 'validation_h5',
        'holdout_used': False,
        'train_chunks': len(train_data),
        'validation_chunks': len(val_data),
        'train_windows': len(train_data) if causal_last_step else None,
        'validation_windows': len(val_data) if causal_last_step else None,
        'training_stride': train_stride if causal_last_step else None,
        'validation_stride': 1 if causal_last_step else None,
        'causal_last_step': bool(causal_last_step),
        'history_stride': history_stride,
        'preload_frames': bool(preload_frames),
        'attention_scope': 'historical_window_only' if causal_last_step else 'complete_window',
        'window_padding': 'repeat_first_frame_on_left' if causal_last_step else 'repeat_last_frame_on_right',
        'seed': seed, 'threads': threads, 'device': device,
        'resolved_device': str(resolved_device),
        'environment': {
            'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'platform': platform.platform(),
        },
    }
    if causal_last_step:
        config['training_supervision'] = (
            'Gaussian SimDR KL for the last step of each historical window '
            'without external teacher distillation')
        config['selection_metric'] = ('soft_expectation_last_step_p3'
                                      if selection_metric == 'causal_soft'
                                      else 'hard_argmax_last_step_p3')
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'config.json', config)
    history = []
    initial = evaluate_flat_stnet(
        model, val_loader, resolved_device, causal_last_step,
        offline_ma_window=11 if selection_metric == 'offline_soft_ma11' else None,
        decode='soft' if selection_metric == 'causal_soft' else 'hard')
    best_error = initial['mean_error_px']
    best_p3 = initial['p3']
    best_epoch = 0
    history.append({
        'epoch': 0, 'train': None, 'validation': initial,
        'validation_error_px': best_error, 'validation_p3': best_p3,
        'selected': True, 'seconds': 0.0,
    })

    def checkpoint(epoch, completed=False):
        return {
            'schema': 1, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch, 'best_epoch': best_epoch,
            'best_validation_error_px': best_error,
            'best_validation_p3': best_p3,
            'config': config, 'selection_split': 'validation_h5',
            'holdout_used': False, 'training_completed': completed,
        }

    _save_checkpoint(output / 'best.pt', checkpoint(0))
    _save_checkpoint(output / 'last.pt', checkpoint(0))
    no_improve = 0
    try:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            progress = (epoch - 1) / max(1, epochs - 1)
            rate = min_lr + 0.5 * (lr - min_lr) * (
                1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group['lr'] = rate
            model.train()
            sums = np.zeros(5, dtype=np.float64)
            valid_timesteps = 0
            for frames, labels, valid in train_loader:
                frames = frames.to(resolved_device)
                labels = labels.to(resolved_device)
                valid = valid.to(resolved_device).bool()
                optimizer.zero_grad(set_to_none=True)
                if feature_distill_weight:
                    student_features, logits_x, logits_y = (
                        model.forward_with_features(frames))
                else:
                    student_features = None
                    logits_x, logits_y = model(frames)
                if causal_last_step:
                    logits_x, logits_y = logits_x[:, -1:], logits_y[:, -1:]
                    if student_features is not None:
                        student_features = student_features[:, -1:]
                supervised = masked_official_simdr_loss(
                    logits_x, logits_y, labels, valid, sigma=sigma,
                    weights=(hard_frame_weights(
                        frames, logits_x, logits_y, labels, valid,
                        persistent_boost=persistent_boost,
                        sparse_boost=sparse_boost)
                             if persistent_boost or sparse_boost else None))
                distilled = supervised.new_zeros(())
                feature_distilled = supervised.new_zeros(())
                pixel = (masked_pixel_loss(logits_x, logits_y, labels, valid)
                         if pixel_loss_weight else supervised.new_zeros(()))
                loss = (supervised + distill_weight * distilled
                        + feature_distill_weight * feature_distilled
                        + pixel_loss_weight * pixel)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite 3ET STNet training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                count = int(valid.sum())
                sums += np.array([
                    loss.item(), supervised.item(), distilled.item(),
                    feature_distilled.item(), pixel.item()]) * count
                valid_timesteps += count
            if valid_timesteps < 1:
                raise ValueError('Training HDF5 has no valid frames')
            validation = evaluate_flat_stnet(
                model, val_loader, resolved_device, causal_last_step,
                offline_ma_window=11 if selection_metric == 'offline_soft_ma11' else None,
                decode='soft' if selection_metric == 'causal_soft' else 'hard')
            error = validation['mean_error_px']
            p3 = validation['p3']
            selected = select_3et_checkpoint(
                p3, error, best_p3, best_error,
                causal_last_step=causal_last_step, min_delta=min_delta,
                selection_metric=selection_metric)
            if selected:
                best_error = error
                best_p3 = p3
                best_epoch = epoch
                no_improve = 0
                _save_checkpoint(output / 'best.pt', checkpoint(epoch))
            else:
                no_improve += 1
            row = {
                'epoch': epoch, 'learning_rate': rate,
                'train': dict(zip(
                    ('loss', 'supervised_simdr_loss', 'distillation_loss',
                     'feature_distillation_loss', 'pixel_huber_loss'),
                    (sums / valid_timesteps).tolist())),
                'training_valid_timesteps': valid_timesteps,
                'validation': validation,
                'validation_error_px': error,
                'validation_p3': p3,
                'selected': selected,
                'seconds': time.perf_counter() - started,
            }
            history.append(row)
            _write_json(output / 'history.json', history)
            _save_checkpoint(output / 'last.pt', checkpoint(epoch))
            print(json.dumps(row, allow_nan=False), flush=True)
            if no_improve >= patience:
                break
        _save_checkpoint(
            output / 'last.pt', checkpoint(len(history) - 1, completed=True))
        _write_json(output / 'history.json', history)
        summary = {
            'status': 'complete', 'checkpoint': 'best.pt',
            'selection_metric': config['selection_metric'],
            'checkpoint_sha256': _sha256(output / 'best.pt'),
            'best_epoch': best_epoch,
            'best_validation_error_px': best_error,
            'best_validation_p3': best_p3,
            'epochs_completed': len(history) - 1,
            'train_chunks': len(train_data),
            'validation_chunks': len(val_data),
            'holdout_used_for_model_selection': False,
        }
        _write_json(output / 'summary.json', summary)
    finally:
        train_data.close()
        val_data.close()
    return output / 'best.pt'
