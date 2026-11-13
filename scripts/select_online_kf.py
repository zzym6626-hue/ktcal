#!/usr/bin/env python3
"""Validation-only KF selection using cached past-only current-frame predictions.

Batching recordings here accelerates parameter selection. It is not an online
latency measurement: finalists must be rerun with evaluate_3et_online.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from ktcal_paper.combined_3et import (
    SCALE, _sha256, load_matcher_checkpoint, load_stnet_checkpoint, validate_validation_h5,
)
from ktcal_paper.evaluation_protocol import summarize_pixel_errors
from ktcal_paper.offline_batch import OfflineBatchTracker
from ktcal_paper.online_inference import CausalViewDecoder, CausalOutputFilter
from ktcal_paper.stnet_sequence import FlatH5CausalWindowDataset
from ktcal_paper.streaming_inference import predict_streaming_sequences
from models.streaming_stnet import StreamingSTNet


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--h5', required=True)
    parser.add_argument('--stnet', required=True)
    parser.add_argument('--matcher', default='weights/3et/matcher_best.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--views', default='0:0:0,0:0:1')
    parser.add_argument('--temperature', type=float, default=.65)
    parser.add_argument('--ensemble', choices=('mean', 'median'), default='median')
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--history-stride', type=int, default=2)
    parser.add_argument('--warm-start-steps', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    output = Path(args.output)
    cache_path = output.with_suffix('.npz')
    if output.exists() or cache_path.exists():
        raise FileExistsError('Choose new selection outputs')
    h5, label_key, shape, _ = validate_validation_h5(args.h5)
    if h5.name != 'test_aug.h5' or shape != (93, 100, 2, 60, 80):
        raise ValueError('Full validation only')
    torch.set_num_threads(4)
    model, config, divisor = load_stnet_checkpoint(args.stnet, args.device)
    matcher, _ = load_matcher_checkpoint(args.matcher, args.device)
    views = []
    for item in args.views.split(','):
        dx, dy, flip = map(int, item.split(':'))
        views.append((dx, dy, flip))
    with h5py.File(h5, 'r') as handle:
        frames = torch.from_numpy(handle['frames'][:].astype(np.float32))
        labels = handle[label_key][:, :, :2].astype(np.float32)
    streaming = isinstance(model, StreamingSTNet)
    if streaming:
        if args.history_stride != 1:
            raise ValueError('Streaming selection requires history_stride=1')
        prediction = predict_streaming_sequences(
            model, frames, views, divisor=divisor, temperature=args.temperature,
            ensemble=args.ensemble, batch_size=args.batch_size,
            warm_start_steps=args.warm_start_steps)
    else:
        if args.warm_start_steps:
            raise ValueError('Warm-start steps require a streaming model')
        decoder = CausalViewDecoder(model, divisor, views, args.temperature,
                                    args.ensemble).to(args.device)
        data = FlatH5CausalWindowDataset(h5, window=args.window, stride=1,
                                         history_stride=args.history_stride, preload_frames=True)
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        current = []
        with torch.inference_mode():
            for window, _, _ in loader:
                current.append(decoder(window.to(args.device)).cpu().numpy())
        prediction = np.concatenate(current).reshape(93, 100, 2)
        data.close()
    frames = frames.to(args.device)
    print('Causal STNet-only: ' + str(summarize_pixel_errors(prediction / SCALE, labels)), flush=True)
    rows = []
    confidence_values = (.28, .45, .6, .75, .82, .9, 1.) if streaming else (.28, .45, .6, .75, .82, .9)
    for confidence in confidence_values:
        for r_px in (2., 8., 32.):
            for velocity in (0., .5, 1.):
                options = dict(search_size=(35, 30) if streaming else (37, 32),
                               confidence_threshold=confidence,
                               min_psr=None if streaming else 2.55,
                               q_pos=.1 if streaming else .05,
                               q_vel=.5 if streaming else .2, r_px=r_px,
                               fusion_alpha=.75, medium_policy='state', tolerance_px=1.,
                               reset_velocity_scale=velocity, template_refresh=0)
                tracker = OfflineBatchTracker(matcher, device=args.device,
                                               cuda_graph=args.device.startswith('cuda'), **options)
                raw, stats = tracker.track(frames, prediction)
                for alpha in ((None, .25, .5, .75) if streaming else (None, .5)):
                    smoother = CausalOutputFilter(alpha, 2.5)
                    filtered = np.stack([smoother(raw[:, step]) for step in range(100)], 1)
                    metrics = summarize_pixel_errors(filtered / SCALE, labels)
                    rows.append(dict(config=dict(options, ema_alpha=alpha, ema_max_step=2.5),
                                     metrics=metrics, stnet_requests=stats['stnet_requests']))
                # Free per-candidate graph buffers before capturing the next graph.
                del tracker
        print(f'Finished confidence={confidence}: best P3={max(r["metrics"]["p3"] for r in rows):.4%}', flush=True)
    rows.sort(key=lambda row: (row['metrics']['p3'], -row['metrics']['mean_error_px']), reverse=True)
    report = dict(protocol='validation-only cached causal current-frame KF selection',
                  config=vars(args), stnet_config=config, future_frames_used=0,
                  recurrent_history='entire_received_record' if streaming else 'limited_window',
                  labels_used_for_inference=False, final_holdout_used=False,
                  latency_measured=False, final_on_demand_verification_required=True,
                  h5_sha256=_sha256(h5), stnet_sha256=_sha256(args.stnet),
                  matcher_sha256=_sha256(args.matcher), candidates=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    with cache_path.open('xb') as handle:
        np.savez_compressed(handle, causal_global_px=prediction, labels=labels)
    print(json.dumps(rows[:5], indent=2), flush=True)


if __name__ == '__main__':
    main()
