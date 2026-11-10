#!/usr/bin/env python3
"""Diagnose KTCal template matching on the released 3ET validation HDF5.

This intentionally does not train or read the final ``test_200_200.h5``.  It
separates feature matching, finite search-range failures, template-source
errors, and template ageing on ``test_aug.h5``.
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from ktcal_paper.matcher_diagnostics import response_map, summarize_matches
from ktcal_paper.matching import (
    COARSE_FACTOR,
    COARSE_SEARCH,
    COARSE_TEMPLATE,
    SEARCH,
    TEMPLATE,
    PaperMatcher,
    crop_rectangle,
    downsample,
    response_to_pixel,
)
from ktcal_paper.combined_3et import (
    load_stnet_checkpoint, precompute_global_predictions, validate_validation_h5,
)

SCALE = np.array([80.0, 60.0], dtype=np.float32)


def parser():
    result = argparse.ArgumentParser(__doc__)
    result.add_argument("--h5", required=True, help="3ET+2025 test_aug.h5")
    result.add_argument("--stnet-checkpoint", required=True)
    result.add_argument("--matcher-checkpoint", required=True,
                        help="KTCal checkpoint containing matcher_state")
    result.add_argument("--output", required=True)
    result.add_argument("--gaps", default="1,4,8")
    result.add_argument("--metrics", default="cosine,zero_mean_ncc")
    result.add_argument("--search-sizes", default="35x30",
                        help="comma-separated width x height fine search patches")
    result.add_argument("--locators", default="hard",
                        help="comma-separated hard and/or subpixel peak decoding")
    result.add_argument("--include-coarse", action="store_true",
                        help="also evaluate the existing factor-2 coarse proposal pass")
    result.add_argument("--skip-offset-cases", action="store_true",
                        help="omit the synthetic 1/2/3/5-pixel center offsets")
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--model-batch-size", type=int, default=4)
    result.add_argument("--max-sequences", type=int)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--device", default="cuda")
    return result


def parse_search_sizes(value):
    sizes = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        try:
            width, height = (int(part) for part in item.split("x", 1))
        except (TypeError, ValueError):
            raise ValueError(f"invalid search size: {item!r}") from None
        if width < TEMPLATE[0] or height < TEMPLATE[1]:
            raise ValueError(
                f"search size {width}x{height} is smaller than template "
                f"{TEMPLATE[0]}x{TEMPLATE[1]}")
        sizes.append((width, height))
    if not sizes:
        raise ValueError("search sizes cannot be empty")
    return sizes


def load_matcher(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "matcher_state" not in checkpoint:
        raise ValueError("matcher checkpoint must contain matcher_state")
    config = checkpoint.get("config", {})
    matcher = PaperMatcher(
        int(config.get("ncc_channels", 16)),
        float(config.get("temperature", 0.1)))
    matcher.load_state_dict(checkpoint["matcher_state"])
    return matcher.to(device).eval(), config


@torch.inference_mode()
def stnet_predictions(model, frames, batch_size, device, input_divisor):
    predictions = []
    for start in range(0, len(frames), batch_size):
        inputs = torch.from_numpy(
            frames[start:start + batch_size].astype(np.float32)).to(device)
        predictions.append(precompute_global_predictions(
            model, inputs, device, input_divisor=input_divisor))
    return np.concatenate(predictions, axis=0).astype(np.float32)


def fixed_offset_centers(truth, radius):
    directions = np.array(
        ((1, 0), (0, 1), (-1, 0), (0, -1),
         (2 ** -0.5, 2 ** -0.5), (-2 ** -0.5, 2 ** -0.5),
         (-2 ** -0.5, -2 ** -0.5), (2 ** -0.5, -2 ** -0.5)),
        dtype=np.float32)
    return truth + directions[np.arange(len(truth)) % len(directions)] * radius


def psr(response, peak_index):
    flat = response.flatten(1)
    mask = torch.nn.functional.one_hot(peak_index, flat.shape[-1]).bool()
    others = flat.masked_fill(mask, 0.0)
    count = max(1, flat.shape[-1] - 1)
    mean = others.sum(-1) / count
    variance = ((others - mean[:, None]).square().masked_fill(mask, 0.0)).sum(-1) / count
    peak = flat.gather(1, peak_index[:, None]).squeeze(1)
    return (peak - mean) / variance.clamp_min(1e-12).sqrt()


@torch.inference_mode()
def evaluate_case(matcher, frames, target, template_centers, search_centers,
                  metric, batch_size, device, search_size=SEARCH,
                  locator="hard", capture="fine"):
    errors, offsets, borders, confidences, psrs = [], [], [], [], []
    if locator not in ("hard", "subpixel"):
        raise ValueError("locator must be hard or subpixel")
    if capture not in ("fine", "coarse"):
        raise ValueError("capture must be fine or coarse")
    fine_span = np.array(search_size, dtype=np.float32) - np.array(TEMPLATE, dtype=np.float32)
    fine_span /= 2
    coarse_span = ((np.array(COARSE_SEARCH, dtype=np.float32)
                    - np.array(COARSE_TEMPLATE, dtype=np.float32))
                   * COARSE_FACTOR / 2)
    capture_span = fine_span if capture == "fine" else coarse_span
    for start in range(0, len(target), batch_size):
        stop = min(start + batch_size, len(target))
        previous = torch.from_numpy(frames[0][start:stop].astype(np.float32)).to(device)
        current = torch.from_numpy(frames[1][start:stop].astype(np.float32)).to(device)
        template_center = torch.from_numpy(template_centers[start:stop]).to(device)
        search_center = torch.from_numpy(search_centers[start:stop]).to(device)
        template = crop_rectangle(previous, template_center, TEMPLATE)
        proposal = search_center
        coarse_border = None
        coarse_psr = None
        if capture == "coarse":
            previous_small = downsample(previous)
            current_small = downsample(current)
            coarse_template = crop_rectangle(
                previous_small, template_center / COARSE_FACTOR, COARSE_TEMPLATE)
            coarse_search = crop_rectangle(
                current_small, search_center / COARSE_FACTOR, COARSE_SEARCH)
            coarse_template_features = matcher.extractor(coarse_template)
            coarse_search_features = matcher.extractor(coarse_search)
            coarse_response = response_map(
                coarse_template_features, coarse_search_features, metric=metric)
            coarse_local, _ = matcher.locate(
                coarse_response, subpixel=locator == "subpixel")
            proposal = response_to_pixel(
                coarse_local, search_center / COARSE_FACTOR,
                COARSE_TEMPLATE, COARSE_SEARCH) * COARSE_FACTOR
            coarse_flat = coarse_response.flatten(1)
            coarse_index = coarse_flat.argmax(-1)
            coarse_height, coarse_width = coarse_response.shape[-2:]
            coarse_row = coarse_index // coarse_width
            coarse_column = coarse_index % coarse_width
            coarse_border = ((coarse_row == 0) | (coarse_row == coarse_height - 1) |
                             (coarse_column == 0) | (coarse_column == coarse_width - 1))
            coarse_psr = psr(coarse_response, coarse_index)
        search = crop_rectangle(current, proposal, search_size)
        template_features = matcher.extractor(template)
        search_features = matcher.extractor(search)
        response = response_map(template_features, search_features, metric=metric)
        flat = response.flatten(1)
        confidence, index = flat.max(-1)
        height, width = response.shape[-2:]
        row, column = index // width, index % width
        local, _ = matcher.locate(response, subpixel=locator == "subpixel")
        matched = response_to_pixel(local, proposal, TEMPLATE, search_size)
        expected = torch.from_numpy(target[start:stop]).to(device)
        errors.append(torch.linalg.vector_norm(matched - expected, dim=-1).cpu().numpy())
        offsets.append((expected - search_center).cpu().numpy())
        fine_border = ((row == 0) | (row == height - 1) |
                       (column == 0) | (column == width - 1))
        borders.append((fine_border if coarse_border is None else coarse_border).cpu().numpy())
        confidences.append(confidence.cpu().numpy())
        fine_psr = psr(response, index)
        psrs.append((fine_psr if coarse_psr is None
                     else torch.minimum(coarse_psr, fine_psr)).cpu().numpy())
    return summarize_matches(
        np.concatenate(errors), np.concatenate(offsets), np.concatenate(borders),
        np.concatenate(confidences), np.concatenate(psrs), capture_span)


def main():
    args = parser().parse_args()
    if args.batch_size < 1 or args.model_batch_size < 1 or args.threads < 1:
        raise ValueError("batch sizes and threads must be positive")
    gaps = [int(value) for value in args.gaps.split(",") if value]
    metrics = [value for value in args.metrics.split(",") if value]
    search_sizes = parse_search_sizes(args.search_sizes)
    locators = [value.strip() for value in args.locators.split(",") if value.strip()]
    if not gaps or min(gaps) < 1:
        raise ValueError("gaps must contain positive integers")
    if any(value not in ("cosine", "zero_mean_ncc") for value in metrics):
        raise ValueError("metrics must be cosine and/or zero_mean_ncc")
    if not locators or any(value not in ("hard", "subpixel") for value in locators):
        raise ValueError("locators must be hard and/or subpixel")

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    stnet, _, input_divisor = load_stnet_checkpoint(args.stnet_checkpoint, device)
    matcher, matcher_config = load_matcher(args.matcher_checkpoint, device)

    h5_path, _, _, _ = validate_validation_h5(args.h5)
    with h5py.File(h5_path, "r") as handle:
        label_key = "label" if "label" in handle else "labels"
        sequence_count = len(handle["frames"])
        if args.max_sequences is not None:
            sequence_count = min(sequence_count, args.max_sequences)
        frames = handle["frames"][:sequence_count]
        labels = handle[label_key][:sequence_count, :, :2].astype(np.float32) * SCALE
    predictions = stnet_predictions(
        stnet, frames, args.model_batch_size, device, input_divisor)

    cases = []
    for gap in gaps:
        if gap >= frames.shape[1]:
            continue
        previous_frames = frames[:, :-gap].reshape(-1, 2, 60, 80)
        current_frames = frames[:, gap:].reshape(-1, 2, 60, 80)
        previous_truth = labels[:, :-gap].reshape(-1, 2)
        current_truth = labels[:, gap:].reshape(-1, 2)
        previous_prediction = predictions[:, :-gap].reshape(-1, 2)
        current_prediction = predictions[:, gap:].reshape(-1, 2)
        definitions = [
            ("oracle_template_oracle_center", previous_truth, current_truth),
            ("oracle_template_stnet_center", previous_truth, current_prediction),
            ("stnet_template_stnet_center", previous_prediction, current_prediction),
        ]
        if gap == min(gaps) and not args.skip_offset_cases:
            definitions.extend(
                (f"oracle_template_offset_{radius:g}px", previous_truth,
                 fixed_offset_centers(current_truth, radius))
                for radius in (1.0, 2.0, 3.0, 5.0))
        configurations = [("fine", size) for size in search_sizes]
        if args.include_coarse:
            configurations.append(("coarse", SEARCH))
        for metric in metrics:
            for capture, search_size in configurations:
                for locator in locators:
                    for name, template_centers, search_centers in definitions:
                        summary = evaluate_case(
                            matcher, (previous_frames, current_frames), current_truth,
                            template_centers, search_centers, metric, args.batch_size,
                            device, search_size, locator, capture)
                        row = {
                            "gap": gap,
                            "metric": metric,
                            "capture": capture,
                            "search_size": f"{search_size[0]}x{search_size[1]}",
                            "locator": locator,
                            "case": name,
                            **summary,
                        }
                        cases.append(row)
                        print(json.dumps(row), flush=True)

    report = {
        "schema": 2,
        "purpose": "3ET validation-only KTCal template-matching diagnosis",
        "h5": str(h5_path),
        "sequences": int(len(frames)),
        "sequence_length": int(frames.shape[1]),
        "stnet_checkpoint": str(Path(args.stnet_checkpoint).resolve()),
        "matcher_checkpoint": str(Path(args.matcher_checkpoint).resolve()),
        "matcher_training_dataset": matcher_config.get("dataset"),
        "template_size": list(TEMPLATE),
        "search_size": list(SEARCH),
        "search_sizes": [list(size) for size in search_sizes],
        "locators": locators,
        "coarse_included": bool(args.include_coarse),
        "final_test_used": False,
        "cases": cases,
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    if cases:
        with (output / "cases.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(cases)
    print(output / "report.json", flush=True)


if __name__ == "__main__":
    main()
