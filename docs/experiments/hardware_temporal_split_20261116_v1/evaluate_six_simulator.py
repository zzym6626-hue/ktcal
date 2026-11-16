"""Isolated six INT8 graph + ORT CPU-tail SDK simulation, not board test."""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from rknn.api import RKNN

from temporal_probe import load, write_json, INPUTS, OUTPUTS, NAMES, checked_inputs
import onnxruntime as ort
from ktcal_paper.combined_3et import _sha256, validate_validation_h5, SCALE
from ktcal_paper.offline_experiment import spatial_view
from hardware.rv1106.export_matcher import export_matcher
from hardware.rv1106.records import score_records

VIEWS = [(0, 0, False), (0, 0, True), (-2, 0, False), (-2, 0, True),
         (2, 0, False), (2, 0, True), (0, -2, False), (0, -2, True),
         (0, 2, False), (0, 2, True)]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--h5', required=True)
    parser.add_argument('--export-json', required=True)
    parser.add_argument('--calibration-json', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint', default='weights/3et/stnet_best.pt')
    parser.add_argument('--matcher', default='weights/3et/matcher_best.pt')
    parser.add_argument('--limit', type=int, default=93)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    h5, key, shape, _ = validate_validation_h5(args.h5)
    manifest = json.loads(Path('weights/3et/manifest.json').read_text())
    if (h5.name != 'test_aug.h5' or shape != (93, 100, 2, 60, 80) or
            not 1 <= args.limit <= 93 or _sha256(h5) != manifest['validation_h5_sha256'] or
            _sha256(args.checkpoint) != manifest['stnet']['sha256'] or
            _sha256(args.matcher) != manifest['matcher']['sha256']):
        raise ValueError('exact released weights/test_aug required')
    metadata, calibration = checked_inputs(args.export_json, args.calibration_json, args.checkpoint)
    if (not metadata['passed'] or metadata['checkpoint_sha256'] != _sha256(args.checkpoint) or
            metadata['checkpoint_sha256'] != calibration['checkpoint_sha256'] or
            calibration['labels_read'] or calibration['validation_used'] or
            Path(calibration['source']).name != 'train_aug.h5'):
        raise ValueError('export/calibration provenance mismatch')
    torch.set_num_threads(4)
    _, split, _ = load(args.checkpoint)
    binary = out / 'full_bridge'
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-Ihardware/rv1106', 'hardware/rv1106/fast_path.cc',
                    'hardware/rv1106/sequence_path.cc', '/root/ktcal_hardware_split_20261116_v1/probe/full_bridge.cc',
                    '-o', str(binary)], check=True)
    export_matcher(args.matcher, out / 'matcher.bin')
    runtimes = []
    shapes_seen = [[] for _ in range(6)]
    calls = [0] * 6
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    tail = ort.InferenceSession(metadata['cpu_tail']['onnx'], sess_options=options,
                                providers=['CPUExecutionProvider'])
    tail_calls = 0
    tail_max_abs = 0.
    def run_tail(features):
        nonlocal tail_calls, tail_max_abs
        output = tail.run(None, {'post_block3': features.numpy()})
        reference = split.tail(features)
        for actual, expected in zip(output, reference):
            np.testing.assert_allclose(actual, expected.numpy(), atol=3e-4, rtol=3e-4)
            tail_max_abs = max(tail_max_abs, float(np.max(np.abs(actual - expected.numpy()))))
        tail_calls += 1
        return tuple(torch.from_numpy(x) for x in output)
    def check(status, stage):
        if status != 0 and not (status is None and stage == 'config'):
            raise RuntimeError(f'{stage} returned {status}')
    try:
        # load_rknn cannot initialize a simulator. Build identical verified
        # ONNX/train-calibrated graphs and retain each object for inference.
        for index, (stage, data) in enumerate(zip(metadata['stages'], calibration['datasets'])):
            if _sha256(stage['onnx']) != stage['onnx_sha256'] or _sha256(data['dataset']) != data['dataset_sha256']:
                raise ValueError('ONNX/dataset hash mismatch')
            if Path(data['dataset']).read_text().splitlines() != [x['path'] for x in data['samples']]:
                raise ValueError('calibration list mismatch')
            for item in data['samples']:
                if _sha256(item['path']) != item['sha256']:
                    raise ValueError('calibration sample hash mismatch')
            rknn = RKNN(verbose=False)
            runtimes.append(rknn)
            check(rknn.config(target_platform='rv1106'), 'config')
            check(rknn.load_onnx(model=stage['onnx']), 'load_onnx')
            check(rknn.build(do_quantization=True, dataset=data['dataset']), 'build')
            model_file = out / (NAMES[index] + '.rknn')
            check(rknn.export_rknn(str(model_file)), 'export_rknn')
            check(rknn.init_runtime(), 'init_runtime_simulator')
            write_json(out / (NAMES[index] + '_simulation_manifest.json'), dict(
                stage=index + 1, onnx_sha256=stage['onnx_sha256'], rknn_sha256=_sha256(model_file),
                calibration_sha256=_sha256(args.calibration_json), simulation_only=True, board_verified=False))
        def run_spatial(index, frames):
            values = []
            for sample in frames.numpy():
                result = runtimes[index].inference(inputs=[sample[None]], data_format=['nchw'])
                calls[index] += 1
                if result is None or len(result) != 1:
                    raise ValueError('missing simulator output')
                value = np.asarray(result[0], dtype=np.float32)
                if tuple(value.shape) != (1, *OUTPUTS[index]) or not np.isfinite(value).all():
                    raise ValueError('simulator shape or finite contract failed')
                if list(value.shape) not in shapes_seen[index]:
                    shapes_seen[index].append(list(value.shape))
                values.append(value)
            return torch.from_numpy(np.concatenate(values))
        counts = {}
        timings, rows = [], []
        with (out / 'predictions.csv').open('x') as file, h5py.File(h5, 'r') as handle, torch.inference_mode():
            writer = csv.writer(file)
            writer.writerow(['record', 'frame', 'x_px', 'y_px', 'raw_x_px', 'raw_y_px', 'tier', 'stnet_x_px', 'stnet_y_px'])
            for record in range(args.limit):
                start = time.perf_counter()
                raw = np.asarray(handle['frames'][record:record + 1], dtype=np.float32)
                frames = torch.from_numpy(raw)
                transforms, logits = [], []
                for view in VIEWS:
                    transformed = spatial_view(frames, view).contiguous()
                    transforms.append(transformed.numpy()[0])
                    result = split(transformed, runner=run_spatial, tail_runner=run_tail)
                    logits.append(np.concatenate([v.numpy() for v in result], axis=-1)[0])
                for name, value in [('raw.bin', raw), ('views.bin', transforms), ('logits.bin', logits)]:
                    np.asarray(value, dtype='<f4').tofile(out / name)
                bridge = json.loads(subprocess.check_output([str(binary), str(out / 'matcher.bin'),
                    str(out / 'raw.bin'), str(out / 'views.bin'), str(out / 'logits.bin'), str(out / 'cpp.bin')], text=True))
                if bridge['small_fusion_calls'] or not bridge['low_global_invariant'] or not bridge['all_raw_views_equal']:
                    raise ValueError('forbidden low fusion or transform mismatch')
                data = np.fromfile(out / 'cpp.bin', dtype='<f8').reshape(100, 9)
                for frame in range(100):
                    writer.writerow([record, frame, *data[frame, 4:6], *data[frame, 2:4],
                                     int(data[frame, 6]), *data[frame, 7:9]])
                file.flush()
                for k in ['tier1', 'tier2', 'tier3', 'medium_refinements', 'medium_rollbacks',
                          'fusion_calls', 'small_fusion_calls', 'stnet_requests', 'border_peaks']:
                    counts[k] = counts.get(k, 0) + bridge[k]
                seconds = time.perf_counter() - start
                timings.append(seconds)
                row = dict(record=record, seconds=seconds, stats=bridge)
                rows.append(row)
                print(json.dumps(row), flush=True)
        scored = score_records(h5, out / 'predictions.csv', out / 'score.json') if args.limit == 93 else None
        with (out / 'predictions.csv').open() as file:
            values = list(csv.DictReader(file))
        with h5py.File(h5, 'r') as handle:
            labels = handle[key][:args.limit, :, :2].astype(np.float32).reshape(-1, 2).astype(np.float64) * SCALE
        metrics = {}
        for name, x, y in [('stnet', 'stnet_x_px', 'stnet_y_px'), ('ktcal', 'x_px', 'y_px')]:
            pixels = np.array([[float(v[x]), float(v[y])] for v in values])
            errors = np.linalg.norm(pixels - labels, axis=-1)
            metrics[name] = dict(frames=len(errors), mean_error_px=float(errors.mean()),
                                 **{f'p{k}':float(np.mean(errors < k)) for k in [3, 5, 10]})
        if scored is not None:
            for k in ['p3', 'p5', 'p10']:
                if scored['metrics'][k] != metrics['ktcal'][k]:
                    raise ValueError('independent pixel score mismatch')
        report = dict(schema=1, completed=True, complete=args.limit == 93, records=args.limit,
                      checkpoint_sha256=_sha256(args.checkpoint), matcher_sha256=_sha256(args.matcher),
                      h5_sha256=_sha256(h5), calibration_sha256=_sha256(args.calibration_json),
                      export_sha256=_sha256(args.export_json), metrics=metrics, tracker_stats=counts,
                      low_global_invariant_all_records=True, raw_transforms_byte_equal_all_records=True,
                      simulator_calls=calls, shapes_seen=shapes_seen, simulation_only=True, board_verified=False,
                      cpu_tail='ORT floating full100 pool/LayerNorm/BiGRU/attention/FC',
                      cpu_tail_calls=tail_calls, cpu_tail_max_abs_error_vs_torch=tail_max_abs,
                      host_wall_seconds_per_record=timings,
                      timing_scope='six SDK simulator graphs plus ORT tail/reference/Python/replay/IO, no warmup, not NPU latency',
                      csv_sha256=_sha256(out / 'predictions.csv'), record_checks=rows)
        write_json(out / 'report.json', report)
        print(json.dumps({k:v for k,v in report.items() if k != 'record_checks'}), flush=True)
    finally:
        for rknn in runtimes:
            rknn.release()


if __name__ == '__main__':
    main()
