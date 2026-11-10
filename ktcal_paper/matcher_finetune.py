"""Fine-tune only the KTCAL matcher on frozen STNet pixel predictions."""

import copy
import hashlib
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from .matching import SEARCH, TEMPLATE, PaperMatcher, crop_rectangle, response_to_pixel
from .evaluation_protocol import hard_argmax_decode


SCALE = np.array([80.0, 60.0], dtype=np.float32)


def weighted_validation_error(rows):
    """Return the frame-weighted mean error across validation gaps."""
    total_error = 0.0
    total_frames = 0
    for row in rows:
        frames = int(row["frames"])
        error = float(row["mean_error_px"])
        if frames < 0 or not np.isfinite(error):
            raise ValueError("Validation rows require non-negative frames and finite errors")
        total_error += frames * error
        total_frames += frames
    if total_frames < 1:
        raise ValueError("Validation rows contain no scored frames")
    return total_error / total_frames


def checkpoint_with_matcher(source_checkpoint, matcher, metadata):
    """Copy a full KTCal checkpoint while replacing only its matcher weights."""
    if not isinstance(source_checkpoint, dict):
        raise ValueError("Source checkpoint must be a dictionary")
    for key in ("model_state", "matcher_state"):
        if key not in source_checkpoint:
            raise ValueError(f"Source checkpoint must contain {key}")
    result = copy.deepcopy(source_checkpoint)
    result["matcher_state"] = {
        name: value.detach().cpu().clone()
        for name, value in matcher.state_dict().items()
    }
    result["matcher_finetune"] = copy.deepcopy(metadata)
    return result


def validate_prediction_shape(predictions, sequence_count, sequence_length, name):
    """Reject stale or mismatched frozen-prediction caches."""
    expected = (int(sequence_count), int(sequence_length), 2)
    if tuple(predictions.shape) != expected:
        raise ValueError(
            f"{name} prediction cache shape {tuple(predictions.shape)} "
            f"does not match HDF5 shape {expected}"
        )
    if not np.isfinite(predictions).all():
        raise ValueError(f"{name} prediction cache must contain only finite coordinates")
    return predictions


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def _save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _configure(seed, threads, device):
    if threads < 1:
        raise ValueError("threads must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(device)


def _h5_description(path, name):
    path = Path(path).resolve()
    if path.name == "test_200_200.h5":
        raise ValueError("Final test_200_200.h5 must not be used for matcher selection")
    with h5py.File(path, "r") as handle:
        if "frames" not in handle:
            raise ValueError(f"{name} HDF5 must contain frames")
        label_key = "label" if "label" in handle else "labels" if "labels" in handle else None
        if label_key is None:
            raise ValueError(f"{name} HDF5 must contain label or labels")
        frames = handle["frames"]
        labels = handle[label_key]
        if tuple(frames.shape) == (50, 200, 2, 60, 80):
            raise ValueError("Refusing official final holdout shape (50,200,2,60,80)")
        if frames.ndim != 5 or tuple(frames.shape[2:]) != (2, 60, 80):
            raise ValueError(f"{name} frames must have shape [N,T,2,60,80]")
        if labels.shape[:2] != frames.shape[:2] or labels.shape[-1] < 2:
            raise ValueError(f"{name} labels must align with frames")
        return path, label_key, int(frames.shape[0]), int(frames.shape[1])


def _pixel_labels(labels):
    coordinates = np.asarray(labels[..., :2], dtype=np.float32)
    finite = coordinates[np.isfinite(coordinates)]
    if finite.size and float(np.max(np.abs(finite))) <= 1.5:
        coordinates = coordinates * SCALE
    return coordinates


@torch.inference_mode()
def precompute_stnet_predictions(h5_path, model, output, *, batch_size, device,
                                 input_divisor=1.0):
    """Stream STNet over HDF5 into a pixel-coordinate cache."""
    if batch_size < 1:
        raise ValueError("STNet prediction batch size must be positive")
    if input_divisor <= 0:
        raise ValueError("STNet input divisor must be positive")
    path, _, sequence_count, sequence_length = _h5_description(h5_path, "prediction")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    temporary = output.with_suffix(output.suffix + ".partial")
    predictions = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32,
        shape=(sequence_count, sequence_length, 2)
    )
    model = model.to(device).eval()
    try:
        with h5py.File(path, "r", rdcc_nbytes=64 * 1024 * 1024) as handle:
            frames = handle["frames"]
            for start in range(0, sequence_count, batch_size):
                stop = min(start + batch_size, sequence_count)
                inputs = torch.from_numpy(
                    frames[start:stop].astype(np.float32)
                ).to(device) / input_divisor
                logits_x, logits_y = model(inputs)
                predictions[start:stop] = (
                    hard_argmax_decode(logits_x, logits_y) * SCALE
                ).astype(np.float32)
        predictions.flush()
    except BaseException:
        del predictions
        if temporary.exists():
            temporary.unlink()
        raise
    del predictions
    temporary.replace(output)
    return output


def _sample_sequence_pairs(frames, labels, predictions, rng, samples_per_sequence):
    previous_frames = []
    current_frames = []
    template_centers = []
    search_centers = []
    targets = []
    length = frames.shape[1]
    for row in range(len(frames)):
        for _ in range(samples_per_sequence):
            current = int(rng.integers(1, length))
            gap = int(rng.integers(1, min(8, current) + 1))
            previous = current - gap
            previous_frames.append(frames[row, previous])
            current_frames.append(frames[row, current])
            template_centers.append(predictions[row, previous])
            search_centers.append(predictions[row, current])
            targets.append(labels[row, current])
    return (
        np.asarray(previous_frames),
        np.asarray(current_frames),
        np.asarray(template_centers, dtype=np.float32),
        np.asarray(search_centers, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def _train_epoch(matcher, optimizer, h5_path, label_key, predictions, rng,
                 batch_size, sequence_batch_size, samples_per_sequence, device):
    matcher.train()
    sequence_order = rng.permutation(len(predictions))
    loss_sum = 0.0
    valid_sum = 0
    sampled = 0
    with h5py.File(h5_path, "r", rdcc_nbytes=64 * 1024 * 1024) as handle:
        frame_data = handle["frames"]
        label_data = handle[label_key]
        for start in range(0, len(sequence_order), sequence_batch_size):
            indices = np.sort(sequence_order[start:start + sequence_batch_size])
            frames = frame_data[indices]
            labels = _pixel_labels(label_data[indices])
            cached = np.asarray(predictions[indices], dtype=np.float32)
            pairs = _sample_sequence_pairs(
                frames, labels, cached, rng, samples_per_sequence
            )
            sampled += len(pairs[0])
            order = rng.permutation(len(pairs[0]))
            for offset in range(0, len(order), batch_size):
                chosen = order[offset:offset + batch_size]
                previous = torch.from_numpy(pairs[0][chosen].astype(np.float32)).to(device)
                current = torch.from_numpy(pairs[1][chosen].astype(np.float32)).to(device)
                template_centers = torch.from_numpy(pairs[2][chosen]).to(device)
                search_centers = torch.from_numpy(pairs[3][chosen]).to(device)
                targets = torch.from_numpy(pairs[4][chosen]).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, valid = matcher.predicted_center_loss(
                    previous, current, template_centers, search_centers, targets
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite matcher training loss")
                if valid == 0:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    matcher.parameters(), 5.0, error_if_nonfinite=True
                )
                optimizer.step()
                loss_sum += float(loss.detach()) * valid
                valid_sum += valid
    if sampled < 1:
        raise ValueError("Training HDF5 produced no matcher samples")
    return {
        "sampled": sampled,
        "valid": valid_sum,
        "loss": loss_sum / max(1, valid_sum),
    }


@torch.inference_mode()
def _validate_gap(matcher, h5_path, label_key, predictions, gap,
                  batch_size, sequence_batch_size, device):
    matcher.eval()
    errors = []
    with h5py.File(h5_path, "r", rdcc_nbytes=64 * 1024 * 1024) as handle:
        frame_data = handle["frames"]
        label_data = handle[label_key]
        for start in range(0, len(predictions), sequence_batch_size):
            stop = min(start + sequence_batch_size, len(predictions))
            frames = frame_data[start:stop]
            labels = _pixel_labels(label_data[start:stop])
            cached = np.asarray(predictions[start:stop], dtype=np.float32)
            previous = frames[:, :-gap].reshape(-1, 2, 60, 80)
            current = frames[:, gap:].reshape(-1, 2, 60, 80)
            template_centers = cached[:, :-gap].reshape(-1, 2)
            search_centers = cached[:, gap:].reshape(-1, 2)
            targets = labels[:, gap:].reshape(-1, 2)
            for offset in range(0, len(targets), batch_size):
                limit = min(offset + batch_size, len(targets))
                before = torch.from_numpy(
                    previous[offset:limit].astype(np.float32)
                ).to(device)
                after = torch.from_numpy(
                    current[offset:limit].astype(np.float32)
                ).to(device)
                template_center = torch.from_numpy(
                    template_centers[offset:limit].copy()
                ).to(device)
                search_center = torch.from_numpy(
                    search_centers[offset:limit].copy()
                ).to(device)
                expected = torch.from_numpy(targets[offset:limit]).to(device)
                template = crop_rectangle(before, template_center, TEMPLATE)
                search = crop_rectangle(after, search_center, SEARCH)
                local, _, _ = matcher(template, search, subpixel=True)
                predicted = response_to_pixel(local, search_center)
                errors.append(
                    torch.linalg.vector_norm(predicted - expected, dim=-1).cpu().numpy()
                )
    if not errors:
        raise ValueError(f"Validation gap {gap} produced no scored frames")
    errors = np.concatenate(errors)
    return {
        "gap": int(gap),
        "frames": int(len(errors)),
        "mean_error_px": float(errors.mean()),
        "median_error_px": float(np.median(errors)),
        "p1": float(np.mean(errors <= 1.0)),
        "p3": float(np.mean(errors <= 3.0)),
        "p5": float(np.mean(errors <= 5.0)),
    }


def train_matcher(train_h5, validation_h5, source_checkpoint, output,
                  train_predictions, validation_predictions, *, epochs=30,
                  patience=6, batch_size=128, sequence_batch_size=16,
                  samples_per_sequence=8, lr=1e-4, weight_decay=1e-4,
                  seed=42, threads=4, device="auto",
                  validation_gaps=(1, 4, 8)):
    """Train ``PaperMatcher`` while preserving every source STNet parameter."""
    if min(epochs, patience, batch_size, sequence_batch_size, samples_per_sequence) < 1:
        raise ValueError("Training counts must be positive")
    if lr <= 0 or weight_decay < 0:
        raise ValueError("Require a positive learning rate and non-negative weight decay")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    train_path, train_label_key, train_count, train_length = _h5_description(
        train_h5, "train"
    )
    val_path, val_label_key, val_count, val_length = _h5_description(
        validation_h5, "validation"
    )
    gaps = tuple(int(gap) for gap in validation_gaps if 0 < int(gap) < val_length)
    if not gaps:
        raise ValueError("No validation gap fits the validation sequence length")
    source_path = Path(source_checkpoint).resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(source, dict) or "matcher_state" not in source:
        raise ValueError("Source checkpoint must contain matcher_state")
    if "model_state" not in source:
        raise ValueError("Source checkpoint must contain model_state")
    config = source.get("config", {})
    matcher = PaperMatcher(
        int(config.get("ncc_channels", 16)),
        float(config.get("temperature", 0.1)),
    )
    matcher.load_state_dict(source["matcher_state"])
    train_cache = validate_prediction_shape(
        np.load(train_predictions, mmap_mode="r"), train_count, train_length, "train"
    )
    val_cache = validate_prediction_shape(
        np.load(validation_predictions, mmap_mode="r"), val_count, val_length,
        "validation"
    )
    resolved_device = _configure(seed, threads, device)
    matcher.to(resolved_device)
    optimizer = torch.optim.AdamW(
        matcher.parameters(), lr=lr, weight_decay=weight_decay
    )
    output.mkdir(parents=True, exist_ok=False)
    run_config = {
        "schema": 1,
        "pipeline": "matcher_only_stnet_centers",
        "train_h5": str(train_path),
        "validation_h5": str(val_path),
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": _sha256(source_path),
        "train_prediction_cache": str(Path(train_predictions).resolve()),
        "validation_prediction_cache": str(Path(validation_predictions).resolve()),
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "sequence_batch_size": sequence_batch_size,
        "samples_per_sequence": samples_per_sequence,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "threads": threads,
        "device": str(resolved_device),
        "validation_gaps": list(gaps),
        "template_size": list(TEMPLATE),
        "search_size": list(SEARCH),
        "validation_locator": "parabolic_subpixel",
        "frame_preprocessing": "raw float32 values from released 3ET HDF5",
        "final_holdout_used": False,
    }
    _write_json(output / "config.json", run_config)
    history = []
    best_error = float("inf")
    no_improve = 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        training = _train_epoch(
            matcher, optimizer, train_path, train_label_key, train_cache, rng,
            batch_size, sequence_batch_size, samples_per_sequence, resolved_device
        )
        validation = [
            _validate_gap(
                matcher, val_path, val_label_key, val_cache, gap, batch_size,
                sequence_batch_size, resolved_device
            )
            for gap in gaps
        ]
        validation_error = weighted_validation_error(validation)
        improved = validation_error < best_error - 1e-9
        if improved:
            best_error = validation_error
            no_improve = 0
        else:
            no_improve += 1
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "training_loss": training["loss"],
            "training_samples": training["sampled"],
            "training_valid_samples": training["valid"],
            "validation": validation,
            "validation_error_px": validation_error,
            "selected": improved,
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        metadata = {
            "schema": 1,
            "pipeline": "matcher_only_stnet_centers",
            "selected_epoch": epoch,
            "validation_error_px": validation_error,
            "validation": validation,
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": run_config["source_checkpoint_sha256"],
            "stnet_frozen": True,
            "final_holdout_used": False,
            "config": run_config,
        }
        current = checkpoint_with_matcher(source, matcher, metadata)
        _save_checkpoint(output / "last.pt", current)
        if improved:
            _save_checkpoint(output / "best.pt", current)
        _write_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
        if no_improve >= patience:
            break
    best = output / "best.pt"
    if not best.is_file():
        raise RuntimeError("Matcher training finished without a selected checkpoint")
    summary = {
        "status": "complete",
        "epochs_completed": len(history),
        "selected_epoch": next(
            row["epoch"] for row in history
            if abs(row["validation_error_px"] - best_error) <= 1e-12
        ),
        "best_validation_error_px": best_error,
        "checkpoint": best.name,
        "checkpoint_sha256": _sha256(best),
        "source_model_state_preserved": True,
        "holdout_used_for_model_selection": False,
    }
    _write_json(output / "training_summary.json", summary)
    return best
