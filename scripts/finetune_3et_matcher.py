#!/usr/bin/env python3
"""Fine-tune KTCAL's matcher using frozen STNet predictions."""

import argparse
from pathlib import Path

import torch

from ktcal_paper.matcher_finetune import (
    precompute_stnet_predictions,
    train_matcher,
)
from ktcal_paper.combined_3et import load_stnet_checkpoint


def build_parser():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--train-h5", required=True)
    parser.add_argument("--validation-h5", required=True)
    parser.add_argument("--stnet-checkpoint", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-batch-size", type=int, default=16)
    parser.add_argument("--samples-per-sequence", type=int, default=8)
    parser.add_argument("--model-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-gaps", default="1,4,8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser


def _prediction_device(value):
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(value)


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    output = Path(arguments.output).resolve()
    cache = output.parent / f"{output.name}_prediction_cache"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if cache.exists():
        raise FileExistsError(f"Refusing to overwrite {cache}")
    cache.mkdir(parents=True, exist_ok=False)
    device = _prediction_device(arguments.device)
    model, _, divisor = load_stnet_checkpoint(arguments.stnet_checkpoint, device)
    train_cache = precompute_stnet_predictions(
        arguments.train_h5,
        model,
        cache / "train_predictions.npy",
        batch_size=arguments.model_batch_size,
        device=device,
        input_divisor=divisor,
    )
    validation_cache = precompute_stnet_predictions(
        arguments.validation_h5,
        model,
        cache / "validation_predictions.npy",
        batch_size=arguments.model_batch_size,
        device=device,
        input_divisor=divisor,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gaps = tuple(
        int(value.strip())
        for value in arguments.validation_gaps.split(",")
        if value.strip()
    )
    best = train_matcher(
        arguments.train_h5,
        arguments.validation_h5,
        arguments.source_checkpoint,
        output,
        train_cache,
        validation_cache,
        epochs=arguments.epochs,
        patience=arguments.patience,
        batch_size=arguments.batch_size,
        sequence_batch_size=arguments.sequence_batch_size,
        samples_per_sequence=arguments.samples_per_sequence,
        lr=arguments.lr,
        weight_decay=arguments.weight_decay,
        validation_gaps=gaps,
        seed=arguments.seed,
        threads=arguments.threads,
        device=arguments.device,
    )
    print(f"BEST={best}", flush=True)
    return best


if __name__ == "__main__":
    main()
