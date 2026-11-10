#!/usr/bin/env python3
"""Evaluate STNet with the released 3ET+2025 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from ktcal_paper.evaluation_protocol import (
    hard_argmax_decode,
    official_postprocess,
    summarize_pixel_errors,
    trim_record_chunks,
)


OFFICIAL_RECORDING_LENGTHS = (699, 565, 672, 412, 669, 905, 1039, 736, 1160, 645, 1238)
OFFICIAL_FRAMES_SHAPE = (50, 200, 2, 60, 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Official 3ET+2025 protocol evaluation")
    parser.add_argument("--h5", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--architecture", choices=("stnet",), default="stnet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", help="Official num_events.txt for post-processing")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser


def validate_official_h5(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"official HDF5 file not found: {path}")
    with h5py.File(path, "r") as handle:
        label_key = "label" if "label" in handle else "labels"
        if "frames" not in handle or label_key not in handle:
            raise ValueError("official HDF5 must contain frames and label datasets")
        frames_shape = tuple(handle["frames"].shape)
        labels_shape = tuple(handle[label_key].shape)
    if frames_shape != OFFICIAL_FRAMES_SHAPE:
        raise ValueError(
            f"official frames shape must be {OFFICIAL_FRAMES_SHAPE}, got {frames_shape}"
        )
    if labels_shape[:2] != frames_shape[:2] or len(labels_shape) != 3 or labels_shape[-1] < 2:
        raise ValueError(
            "official labels shape must be [50, 200, at least 2] and match frames"
        )
    return {
        "frames_shape": frames_shape,
        "labels_shape": labels_shape,
        "label_key": label_key,
    }


def _checkpoint_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    for key in ("model_state_dict", "model_state", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("checkpoint does not contain model weights")


def load_checkpoint_state(model, checkpoint, *, architecture: str) -> dict:
    if architecture != "stnet":
        raise ValueError(f"unsupported architecture: {architecture}")
    state = _checkpoint_state(checkpoint)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"checkpoint/model mismatch: {error}") from error
    return {"loaded_parameters": len(state), "allowed_missing_parameters": []}


def load_event_statistics(path, *, expected_frames: int) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"event statistics file not found: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        cleaned = line.replace("[", "").replace("]", "").strip()
        if not cleaned:
            continue
        try:
            values = [float(value) for value in cleaned.split()]
        except ValueError as error:
            raise ValueError(f"invalid event statistics at line {line_number}") from error
        if len(values) != 2:
            raise ValueError(f"event statistics line {line_number} must contain two values")
        rows.append(values)
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (expected_frames, 2):
        raise ValueError(
            f"event statistics must have shape ({expected_frames}, 2), got {result.shape}"
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _confidence(logits_x: torch.Tensor, logits_y: torch.Tensor) -> torch.Tensor:
    pooled_x = F.avg_pool1d(logits_x, kernel_size=10, stride=1, padding=1)
    pooled_y = F.avg_pool1d(logits_y, kernel_size=10, stride=1, padding=1)
    return (
        torch.softmax(pooled_x, dim=-1).amax(dim=-1)
        + torch.softmax(pooled_y, dim=-1).amax(dim=-1)
    )


def describe_temporal_protocol(architecture: str, config: dict) -> dict:
    if architecture != "stnet":
        raise ValueError(f"unsupported architecture: {architecture}")
    return {
        "inference_chunk": 200,
        "bidirectional": not bool(config.get("unidirectional", False)),
        "training_window": config.get("window"),
        "decoder": "hard_argmax_80x60",
    }


def resolve_preprocessing(architecture: str, config: dict):
    if architecture != "stnet":
        raise ValueError(f"unsupported architecture: {architecture}")
    divisor = float(config.get("input_divisor", 15.0))
    if divisor <= 0:
        raise ValueError("STNet input divisor must be positive")
    return divisor, (
        "raw_event_counts" if divisor == 1.0
        else f"event_counts_divided_by_{divisor:g}"
    )


def _build_model(architecture: str, checkpoint):
    if architecture != "stnet":
        raise ValueError(f"unsupported architecture: {architecture}")
    from models.STNet import STNet
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = STNet(SimpleNamespace(
        stnet_block3_kernel=int(config.get("block3_kernel", 5)),
        stnet_unidirectional=bool(config.get("unidirectional", False)),
        stnet_decode="argmax", stnet_readout=config.get("readout", "last"),
        stnet_temporal_spatial_kernel=int(config.get("temporal_spatial_kernel", 1)),
    ))
    input_divisor, preprocessing = resolve_preprocessing(architecture, config)
    load_report = load_checkpoint_state(model, checkpoint, architecture=architecture)
    return model, config, input_divisor, preprocessing, load_report


@torch.inference_mode()
def evaluate(args) -> dict:
    h5_path = Path(args.h5).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    h5_shapes = validate_official_h5(h5_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("batch size and threads must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, config, input_divisor, preprocessing, load_report = _build_model(
        args.architecture, checkpoint
    )
    model.to(device).eval()

    prediction_chunks = []
    label_chunks = []
    confidence_chunks = []
    with h5py.File(h5_path, "r") as handle:
        frames = handle["frames"]
        labels = handle[h5_shapes["label_key"]]
        for start in range(0, len(frames), args.batch_size):
            stop = min(start + args.batch_size, len(frames))
            frame_batch = torch.from_numpy(frames[start:stop].astype(np.float32))
            if input_divisor != 1.0:
                frame_batch = frame_batch / input_divisor
            logits_x, logits_y = model(frame_batch.to(device))
            prediction_chunks.append(hard_argmax_decode(logits_x, logits_y))
            confidence_chunks.append(_confidence(logits_x, logits_y).cpu().numpy())
            label_chunks.append(labels[start:stop, :, :2])

    prediction_chunks = np.concatenate(prediction_chunks, axis=0)
    label_chunks = np.concatenate(label_chunks, axis=0)
    confidence_chunks = np.concatenate(confidence_chunks, axis=0)
    predictions = trim_record_chunks(
        prediction_chunks,
        OFFICIAL_RECORDING_LENGTHS,
        chunk_size=200,
    )
    labels = trim_record_chunks(
        label_chunks,
        OFFICIAL_RECORDING_LENGTHS,
        chunk_size=200,
    )
    confidence = trim_record_chunks(
        confidence_chunks,
        OFFICIAL_RECORDING_LENGTHS,
        chunk_size=200,
    )
    raw_metrics = summarize_pixel_errors(predictions, labels)

    post_metrics = None
    events_sha256 = None
    if args.events:
        events_path = Path(args.events).resolve()
        events = load_event_statistics(events_path, expected_frames=len(predictions))
        processed = official_postprocess(
            predictions,
            confidence=confidence,
            num_events_y=events[:, 1],
            recording_lengths=OFFICIAL_RECORDING_LENGTHS,
        )
        post_metrics = summarize_pixel_errors(processed, labels)
        events_sha256 = _sha256(events_path)

    report = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": args.architecture,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "h5": str(h5_path),
        "h5_sha256": _sha256(h5_path),
        "events_sha256": events_sha256,
        "valid_frames": len(predictions),
        "recording_lengths": list(OFFICIAL_RECORDING_LENGTHS),
        **h5_shapes,
        "preprocessing": preprocessing,
        "input_divisor": input_divisor,
        "checkpoint_load": load_report,
        "temporal_protocol": describe_temporal_protocol(args.architecture, config),
        "training_dataset": config.get("dataset"),
        "training_split_mode": config.get("split_mode"),
        "cross_dataset_diagnostic": (
            config.get("dataset") != "3ET+2025"
        ),
        "raw_metrics": raw_metrics,
        "post_metrics": post_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate(args)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
