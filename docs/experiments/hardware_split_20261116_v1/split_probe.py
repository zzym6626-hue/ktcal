"""Isolated equivalent-split feasibility spike; not a deployable RKNN backend.

Spatial kernels run per frame. Original dense temporal blocks and the complete
100-frame readout are retained. No checkpoint tensor is trained or overwritten.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hardware.rv1106.export_stnet import validate_sequence_config
from ktcal_paper.combined_3et import _sha256, load_stnet_checkpoint

INPUTS = ((2, 60, 80), (32, 20, 26), (64, 6, 8))
OUTPUTS = ((32, 60, 80), (64, 20, 26), (128, 6, 8))


class Spatial2D(nn.Sequential):
    def __init__(self, block):
        conv, bn, relu = block.spatial
        if (conv.kernel_size[0] != 1 or conv.stride[0] != 1 or
                conv.padding[0] != 0 or conv.dilation[0] != 1):
            raise ValueError('only frame-independent spatial kernels can be split')
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels,
                             conv.kernel_size[1:], conv.stride[1:],
                             conv.padding[1:], conv.dilation[1:], conv.groups,
                             bias=conv.bias is not None).to(conv.weight)
        new_bn = nn.BatchNorm2d(bn.num_features, eps=bn.eps,
                                momentum=bn.momentum, affine=bn.affine,
                                track_running_stats=bn.track_running_stats).to(conv.weight)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight.squeeze(2))
            if conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
        new_bn.load_state_dict(bn.state_dict(), strict=True)
        super().__init__(new_conv, new_bn, copy.deepcopy(relu))
        self.eval()


class SplitSTNet(nn.Module):
    def __init__(self, model, divisor):
        super().__init__()
        if model.training or not math.isfinite(divisor) or divisor <= 0:
            raise ValueError('evaluation mode and finite positive divisor required')
        blocks = (model.block1, model.block2, model.block3)
        for block in blocks:
            convs = [layer for layer in block.temporal if isinstance(layer, nn.Conv3d)]
            if len(convs) != 1 or convs[0].kernel_size != (3, 1, 1) or convs[0].groups != 1:
                raise ValueError('this spike only supports the released dense 311 model')
        self.spatial = nn.ModuleList([Spatial2D(block) for block in blocks])
        self.temporal = nn.ModuleList([copy.deepcopy(block.temporal) for block in blocks])
        # Reuse the original complete extract_temporal/attention/head path. It
        # accepts post-block3 features after replacing just the three blocks.
        self.tail = copy.deepcopy(model)
        self.tail.block1 = nn.Identity()
        self.tail.block2 = nn.Identity()
        self.tail.block3 = nn.Identity()
        self.divisor = float(divisor)
        self.eval()

    def forward(self, raw_frames, spatial_runner=None):
        if raw_frames.ndim != 5 or tuple(raw_frames.shape[1:]) != (100, 2, 60, 80):
            raise ValueError('expected full [B,100,2,60,80] records')
        x = raw_frames.float() / self.divisor
        batch, length = x.shape[:2]
        for index, (spatial, temporal) in enumerate(zip(self.spatial, self.temporal)):
            frames = x.reshape(batch * length, *INPUTS[index]).contiguous()
            if spatial_runner is None:
                features = spatial(frames)
            else:
                features = spatial_runner(index, frames)
            if tuple(features.shape) != (batch * length, *OUTPUTS[index]):
                raise ValueError('spatial backend output shape mismatch')
            x = features.reshape(batch, length, *OUTPUTS[index]).permute(0, 2, 1, 3, 4)
            x = temporal(x).permute(0, 2, 1, 3, 4).contiguous()
        return self.tail(x)


def load(checkpoint):
    model, config, divisor = load_stnet_checkpoint(checkpoint, 'cpu')
    validate_sequence_config(config)
    return model, SplitSTNet(model, divisor), config


def write_json(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def export(checkpoint, directory):
    import onnx
    import onnxruntime as ort
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model, split, config = load(checkpoint)
    checkpoint_sha = _sha256(checkpoint)
    stages = []
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    torch.manual_seed(719)
    with torch.inference_mode():
        for index, spatial in enumerate(split.spatial):
            path = directory / f'spatial{index + 1}.onnx'
            sample = torch.randn(1, *INPUTS[index])
            torch.onnx.export(spatial, sample, str(path), opset_version=17,
                              input_names=['features'], output_names=['spatial_features'],
                              do_constant_folding=True)
            graph = onnx.load(str(path))
            onnx.checker.check_model(graph, full_check=True)
            ops = [node.op_type for node in graph.graph.node]
            if ops != ['Conv', 'Relu']:
                raise ValueError(f'expected fused single Conv2D/Relu, got {ops}')
            session = ort.InferenceSession(str(path), sess_options=options,
                                          providers=['CPUExecutionProvider'])
            error = 0.
            for value in (sample, torch.zeros_like(sample), sample * 100.):
                actual = session.run(None, {'features': value.numpy()})[0]
                expected = spatial(value).numpy()
                np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
                error = max(error, float(np.max(np.abs(actual - expected))))
            stages.append(dict(stage=index + 1, input_shape=[1, *INPUTS[index]],
                               output_shape=[1, *OUTPUTS[index]], onnx=str(path),
                               onnx_sha256=_sha256(path), bytes=path.stat().st_size,
                               operators=ops, onnx_verified=True,
                               random_scaled_zero_max_abs_error=error))
    report = dict(schema=1, experimental=True, passed=True, checkpoint_sha256=checkpoint_sha,
                  config=config, runtime_divisor=split.divisor, stages=stages,
                  cpu_path='original temporal Conv3D/BN/ReLU/pool + full100 BiGRU/attention/FC',
                  checkpoint_values_changed=False, board_verified=False)
    write_json(directory / 'export.json', report)
    print(json.dumps(report), flush=True)


def calibrate(checkpoint, train_h5, directory):
    import h5py
    from ktcal_paper.offline_experiment import spatial_view
    train_h5, directory = Path(train_h5).resolve(), Path(directory)
    if train_h5.name != 'train_aug.h5':
        raise ValueError('only explicit train_aug frames can be used for calibration')
    directory.mkdir(parents=True, exist_ok=False)
    model, split, _ = load(checkpoint)
    split = split.to('cuda')
    samples = [[] for _ in range(3)]
    for i in range(3):
        (directory / f'spatial{i + 1}').mkdir()
    indices = (0, 25, 50, 75, 99)
    with h5py.File(train_h5, 'r') as handle, torch.inference_mode():
        frames = handle['frames']
        if frames.shape[1:] != (100, 2, 60, 80) or frames.shape[0] < 8:
            raise ValueError('100-frame training records required')
        shape = list(frames.shape)
        for record in range(8):
            raw = torch.from_numpy(np.asarray(frames[record:record + 1], dtype=np.float32)).cuda()
            for flip in (False, True):
                x = spatial_view(raw, (0, 0, flip)).contiguous() / split.divisor
                for index, (spatial, temporal) in enumerate(zip(split.spatial, split.temporal)):
                    for frame in indices:
                        value = x[:, frame].contiguous().cpu().numpy()
                        if not np.isfinite(value).all():
                            raise ValueError('nonfinite training intermediate')
                        path = directory / f'spatial{index + 1}' / f'train_r{record}_f{frame}_m{int(flip)}.npy'
                        with path.open('xb') as out:
                            np.save(out, value, allow_pickle=False)
                        samples[index].append(dict(record=record, frame=frame, flip=flip,
                                                   path=str(path), sha256=_sha256(path), shape=list(value.shape)))
                    x = spatial(x.reshape(100, *INPUTS[index])).reshape(1, 100, *OUTPUTS[index])
                    x = temporal(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).contiguous()
    datasets = []
    for index, values in enumerate(samples):
        path = directory / f'spatial{index + 1}' / 'train_inputs.txt'
        with path.open('x') as out:
            out.write(''.join(value['path'] + '\n' for value in values))
        datasets.append(dict(stage=index + 1, dataset=str(path), dataset_sha256=_sha256(path), samples=values))
    report = dict(schema=1, purpose='conversion probe, not accuracy-qualified INT8 calibration',
                  checkpoint_sha256=_sha256(checkpoint), source=str(train_h5), source_h5_sha256=_sha256(train_h5),
                  source_shape=shape, keys_read=['frames'], labels_read=False, validation_used=False,
                  runtime_divisor=split.divisor, intermediate_source='unmodified floating checkpoint',
                  datasets=datasets)
    write_json(directory / 'provenance.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'datasets'}), flush=True)


def convert(export_json, calibration_json, stage, directory):
    from rknn.api import RKNN
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(Path(export_json).read_text())
    calibration = json.loads(Path(calibration_json).read_text())
    if (not metadata.get('passed') or not metadata.get('experimental') or
            metadata['checkpoint_sha256'] != calibration['checkpoint_sha256'] or
            calibration.get('labels_read') is not False or calibration.get('validation_used') is not False or
            Path(calibration['source']).name != 'train_aug.h5'):
        raise ValueError('verified export and training-only calibration required')
    entry, data = metadata['stages'][stage - 1], calibration['datasets'][stage - 1]
    if entry['stage'] != stage or data['stage'] != stage or not entry['onnx_verified']:
        raise ValueError('stage mismatch')
    if _sha256(entry['onnx']) != entry['onnx_sha256'] or _sha256(data['dataset']) != data['dataset_sha256']:
        raise ValueError('export/calibration hash mismatch')
    lines = Path(data['dataset']).read_text().splitlines()
    if lines != [item['path'] for item in data['samples']]:
        raise ValueError('dataset samples mismatch')
    for item in data['samples']:
        if _sha256(item['path']) != item['sha256']:
            raise ValueError('calibration input hash mismatch')
    def check(status, name):
        if status != 0 and not (status is None and name == 'config'):
            raise RuntimeError(f'RKNN {name} returned {status}')
    rknn = RKNN(verbose=True)
    try:
        check(rknn.config(target_platform='rv1106'), 'config')
        check(rknn.load_onnx(model=entry['onnx']), 'load_onnx')
        check(rknn.build(do_quantization=True, dataset=data['dataset']), 'build')
        output = directory / f'spatial{stage}.rknn'
        check(rknn.export_rknn(str(output)), 'export_rknn')
        report = dict(schema=1, stage=stage, conversion_completed=True,
                      checkpoint_sha256=metadata['checkpoint_sha256'], onnx_sha256=entry['onnx_sha256'],
                      rknn_sha256=_sha256(output), rknn_bytes=output.stat().st_size,
                      calibration_sha256=_sha256(calibration_json), quantized=True,
                      target='rv1106', simulator_verified=False, board_verified=False)
        write_json(directory / 'conversion.json', report)
        print(json.dumps(report), flush=True)
        # Retaining this same built object is required for SDK host simulation;
        # load_rknn alone does not provide a host simulator graph.
        check(rknn.init_runtime(), 'init_runtime_simulator')
        errors = []
        import onnxruntime as ort
        session = ort.InferenceSession(entry['onnx'], providers=['CPUExecutionProvider'])
        for item in data['samples'][:5]:
            sample = np.load(item['path'], allow_pickle=False)
            actual = rknn.inference(inputs=[sample], data_format=['nchw'])
            if actual is None or len(actual) != 1 or not np.isfinite(actual[0]).all():
                raise RuntimeError('simulator output missing/nonfinite')
            expected = session.run(None, {'features': sample})[0]
            if actual[0].shape != expected.shape:
                raise RuntimeError(f'simulator shape {actual[0].shape}, expected {expected.shape}')
            delta = np.abs(actual[0] - expected)
            errors.append(dict(sample_sha256=item['sha256'], max_abs=float(delta.max()),
                               mean_abs=float(delta.mean()), reference_max=float(np.abs(expected).max())))
        write_json(directory / 'simulator_probe.json', dict(schema=1, executed=True, stage=stage,
                   simulation_only=True, board_verified=False, calibration_sample_probes=errors))
    finally:
        rknn.release()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('phase', choices=('export', 'calibrate', 'convert'))
    parser.add_argument('--checkpoint', default='weights/3et/stnet_best.pt')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--train-h5')
    parser.add_argument('--export-json')
    parser.add_argument('--calibration-json')
    parser.add_argument('--stage', type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase == 'export':
        export(args.checkpoint, args.output_dir)
    elif args.phase == 'calibrate':
        calibrate(args.checkpoint, args.train_h5, args.output_dir)
    else:
        convert(args.export_json, args.calibration_json, args.stage, args.output_dir)


if __name__ == '__main__':
    main()
