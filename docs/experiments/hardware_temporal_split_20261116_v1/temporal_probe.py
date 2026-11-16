"""Hardware-first isolated six-graph probe. Not a production board backend."""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from split_probe import Spatial2D, write_json
from hardware.rv1106.export_stnet import StaticSpatialPool, validate_sequence_config
from ktcal_paper.combined_3et import _sha256, load_stnet_checkpoint

SPATIAL_INPUTS = ((2, 60, 80), (32, 20, 26), (64, 6, 8))
SPATIAL_OUTPUTS = ((32, 60, 80), (64, 20, 26), (128, 6, 8))
POOLED_OUTPUTS = ((32, 20, 26), (64, 6, 8), (128, 6, 8))
INPUTS = tuple(shape for pair in zip(SPATIAL_INPUTS,
    ((96, 60, 80), (192, 20, 26), (384, 6, 8))) for shape in pair)
OUTPUTS = tuple(shape for pair in zip(SPATIAL_OUTPUTS, POOLED_OUTPUTS) for shape in pair)
NAMES = ('spatial1', 'temporal1', 'spatial2', 'temporal2', 'spatial3', 'temporal3')


def stack_neighbors(features):
    """[B,T,C,H,W] -> [B*T,3C,H,W], zero outside each individual record."""
    if features.ndim != 5 or features.shape[1] < 1:
        raise ValueError('expected nonempty [B,T,C,H,W] features')
    zero = torch.zeros_like(features[:, :1])
    previous = torch.cat((zero, features[:, :-1]), dim=1)
    following = torch.cat((features[:, 1:], zero), dim=1)
    return torch.cat((previous, features, following), dim=2).flatten(0, 1).contiguous()


class Temporal2D(nn.Sequential):
    def __init__(self, block):
        conv, bn, relu, *rest = block.temporal
        if (not isinstance(conv, nn.Conv3d) or conv.kernel_size != (3, 1, 1)
                or conv.groups != 1 or conv.stride != (1, 1, 1)
                or conv.padding != (1, 0, 0) or conv.dilation != (1, 1, 1)
                or not isinstance(bn, nn.BatchNorm3d)):
            raise ValueError('only released dense temporal 311 is supported')
        mapped = nn.Conv2d(conv.in_channels * 3, conv.out_channels, 1,
                           bias=conv.bias is not None).to(conv.weight)
        with torch.no_grad():
            # Conv3D sums [input channel, temporal offset]. Concatenation is
            # [offset, channel], not a direct reshape of Conv3D's weight.
            mapped.weight.copy_(conv.weight.permute(0, 2, 1, 3, 4).reshape_as(mapped.weight))
            if conv.bias is not None:
                mapped.bias.copy_(conv.bias)
        mapped_bn = nn.BatchNorm2d(bn.num_features, eps=bn.eps,
            momentum=bn.momentum, affine=bn.affine,
            track_running_stats=bn.track_running_stats).to(conv.weight)
        mapped_bn.load_state_dict(bn.state_dict(), strict=True)
        layers = [mapped, mapped_bn, copy.deepcopy(relu)]
        for layer in rest:
            if isinstance(layer, nn.AvgPool3d):
                if layer.kernel_size != (1, 3, 3) or layer.padding != 0:
                    raise ValueError('unsupported temporal pooling')
                layers.append(nn.AvgPool2d(3, stride=3))
            elif not isinstance(layer, nn.Dropout):
                raise ValueError('unsupported temporal extra operator')
        super().__init__(*layers)
        self.eval()


class SixGraphSTNet(nn.Module):
    def __init__(self, model, divisor):
        super().__init__()
        if model.training or not math.isfinite(divisor) or divisor <= 0:
            raise ValueError('evaluation model and finite positive divisor required')
        blocks = (model.block1, model.block2, model.block3)
        self.graphs = nn.ModuleList([graph for block in blocks
            for graph in (Spatial2D(block), Temporal2D(block))])
        self.tail = copy.deepcopy(model)
        self.tail.block1 = self.tail.block2 = self.tail.block3 = nn.Identity()
        self.tail.pool = StaticSpatialPool()
        self.divisor = float(divisor)
        self.eval()

    def forward_features(self, raw_frames, runner=None):
        if raw_frames.ndim != 5 or tuple(raw_frames.shape[1:]) != (100, 2, 60, 80):
            raise ValueError('expected full [B,100,2,60,80] records')
        x = raw_frames.float() / self.divisor
        batch, length = x.shape[:2]
        for index, graph in enumerate(self.graphs):
            inputs = stack_neighbors(x) if index % 2 else x.flatten(0, 1).contiguous()
            outputs = graph(inputs) if runner is None else runner(index, inputs)
            if tuple(outputs.shape) != (batch * length, *OUTPUTS[index]):
                raise ValueError('six-graph backend output shape mismatch')
            x = outputs.reshape(batch, length, *OUTPUTS[index])
        return x

    def forward(self, raw_frames, runner=None, tail_runner=None):
        features = self.forward_features(raw_frames, runner)
        return self.tail(features) if tail_runner is None else tail_runner(features)


def load(checkpoint):
    model, config, divisor = load_stnet_checkpoint(checkpoint, 'cpu')
    validate_sequence_config(config)
    return model, SixGraphSTNet(model, divisor), config


def export(checkpoint, directory):
    import onnx
    import onnxruntime as ort
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    _, split, config = load(checkpoint)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    stages = []
    torch.manual_seed(427)
    with torch.inference_mode():
        for index, graph in enumerate(split.graphs):
            path = directory / (NAMES[index] + '.onnx')
            sample = torch.randn(1, *INPUTS[index])
            torch.onnx.export(graph, sample, str(path), opset_version=17,
                input_names=['features'], output_names=['out_features'], do_constant_folding=True)
            model = onnx.load(str(path))
            onnx.checker.check_model(model, full_check=True)
            operators = [node.op_type for node in model.graph.node]
            if any(op not in ('Conv', 'Relu', 'AveragePool') for op in operators):
                raise ValueError(f'unsupported graph operators: {operators}')
            session = ort.InferenceSession(str(path), sess_options=options,
                                          providers=['CPUExecutionProvider'])
            error = 0.
            for value in (sample, sample * 100., torch.zeros_like(sample)):
                actual = session.run(None, {'features': value.numpy()})[0]
                expected = graph(value).numpy()
                np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
                error = max(error, float(np.max(np.abs(actual - expected))))
            stages.append(dict(stage=index + 1, name=NAMES[index], onnx=str(path),
                input_shape=[1, *INPUTS[index]], output_shape=[1, *OUTPUTS[index]],
                onnx_sha256=_sha256(path), bytes=path.stat().st_size, operators=operators,
                onnx_verified=True, max_abs_error=error))
        tail_file = directory / 'cpu_tail.onnx'
        sample = torch.randn(1, 100, 128, 6, 8)
        torch.onnx.export(split.tail, sample, str(tail_file), opset_version=17,
            input_names=['post_block3'], output_names=['x_logits', 'y_logits'])
        onnx.checker.check_model(onnx.load(str(tail_file)), full_check=True)
        session = ort.InferenceSession(str(tail_file), sess_options=options,
                                      providers=['CPUExecutionProvider'])
        errors = []
        for value in (sample, torch.zeros_like(sample)):
            for actual, expected in zip(session.run(None, {'post_block3': value.numpy()}), split.tail(value)):
                np.testing.assert_allclose(actual, expected.numpy(), atol=3e-4, rtol=3e-4)
                errors.append(float(np.max(np.abs(actual - expected.numpy()))))
    report = dict(schema=1, experimental=True, passed=True,
        checkpoint_sha256=_sha256(checkpoint), config=config, runtime_divisor=split.divisor,
        stages=stages, cpu_tail=dict(onnx=str(tail_file), onnx_sha256=_sha256(tail_file),
            bytes=tail_file.stat().st_size, input_shape=[1, 100, 128, 6, 8],
            output_shapes=[[1, 100, 80], [1, 100, 60]], max_abs_error=max(errors),
            host_onnx_verified=True, board_verified=False),
        checkpoint_values_changed=False, board_verified=False)
    write_json(directory / 'export.json', report)
    print(json.dumps(report), flush=True)


def calibrate(checkpoint, train_h5, directory):
    import h5py
    from ktcal_paper.offline_experiment import spatial_view
    train_h5, directory = Path(train_h5).resolve(), Path(directory).resolve()
    if train_h5.name != 'train_aug.h5':
        raise ValueError('only explicit train_aug.h5 calibration is allowed')
    directory.mkdir(parents=True, exist_ok=False)
    model, split, _ = load(checkpoint)
    model, split = model.cuda(), split.cuda()
    samples = [[] for _ in range(6)]
    for name in NAMES:
        (directory / name).mkdir()
    with h5py.File(train_h5, 'r') as handle, torch.inference_mode():
        source = handle['frames']
        if source.shape[1:] != (100, 2, 60, 80) or source.shape[0] < 8:
            raise ValueError('full100 training records required')
        shape = list(source.shape)
        for record in range(8):
            raw = torch.from_numpy(np.asarray(source[record:record + 1], dtype=np.float32)).cuda()
            for flip in (False, True):
                x = spatial_view(raw, (0, 0, flip)).contiguous() / split.divisor
                for block_index, block in enumerate((model.block1, model.block2, model.block3)):
                    for temporal in (False, True):
                        index = block_index * 2 + int(temporal)
                        for frame in (0, 25, 50, 75, 99):
                            if temporal:
                                zero = torch.zeros_like(x[:, frame])
                                value = torch.cat((x[:, frame - 1] if frame else zero,
                                    x[:, frame], x[:, frame + 1] if frame < 99 else zero), dim=1)
                            else:
                                value = x[:, frame]
                            array = value.contiguous().cpu().numpy()
                            if not np.isfinite(array).all():
                                raise ValueError('nonfinite training intermediate')
                            path = directory / NAMES[index] / f'r{record}_t{frame}_m{int(flip)}.npy'
                            with path.open('xb') as out:
                                np.save(out, array, allow_pickle=False)
                            samples[index].append(dict(record=record, frame=frame, flip=flip,
                                path=str(path), shape=list(array.shape), sha256=_sha256(path)))
                        operation = block.temporal if temporal else block.spatial
                        x = operation(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).contiguous()
    datasets = []
    for index, entries in enumerate(samples):
        dataset = directory / NAMES[index] / 'train_inputs.txt'
        with dataset.open('x') as out:
            out.write(''.join(item['path'] + '\n' for item in entries))
        datasets.append(dict(stage=index + 1, name=NAMES[index], dataset=str(dataset),
                             dataset_sha256=_sha256(dataset), samples=entries))
    report = dict(schema=1, purpose='hardware-first probe, no accuracy floor',
        checkpoint_sha256=_sha256(checkpoint), source=str(train_h5), source_shape=shape,
        source_h5_sha256=_sha256(train_h5), keys_read=['frames'], labels_read=False,
        validation_used=False, runtime_divisor=split.divisor,
        intermediate_source='original floating model, not mapped/quantized approximation', datasets=datasets)
    write_json(directory / 'provenance.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'datasets'}), flush=True)


def checked_inputs(export_json, calibration_json, checkpoint):
    metadata = json.loads(Path(export_json).read_text())
    calibration = json.loads(Path(calibration_json).read_text())
    if (not metadata.get('passed') or not metadata.get('experimental')
            or metadata['checkpoint_sha256'] != _sha256(checkpoint)
            or metadata['checkpoint_sha256'] != calibration['checkpoint_sha256']
            or calibration.get('labels_read') is not False
            or calibration.get('validation_used') is not False
            or Path(calibration['source']).name != 'train_aug.h5'
            or len(metadata['stages']) != 6 or len(calibration['datasets']) != 6):
        raise ValueError('verified six graphs and train-only calibration required')
    for index, (stage, data) in enumerate(zip(metadata['stages'], calibration['datasets'])):
        if (stage['stage'] != index + 1 or data['stage'] != index + 1
                or stage['name'] != NAMES[index] or data['name'] != NAMES[index]
                or not stage['onnx_verified'] or _sha256(stage['onnx']) != stage['onnx_sha256']
                or stage['input_shape'] != [1, *INPUTS[index]]
                or stage['output_shape'] != [1, *OUTPUTS[index]]
                or _sha256(data['dataset']) != data['dataset_sha256']
                or Path(data['dataset']).read_text().splitlines() != [x['path'] for x in data['samples']]):
            raise ValueError('graph/calibration contract or hash mismatch')
        for sample in data['samples']:
            if _sha256(sample['path']) != sample['sha256']:
                raise ValueError('calibration sample hash mismatch')
    tail = metadata['cpu_tail']
    if _sha256(tail['onnx']) != tail['onnx_sha256'] or not tail['host_onnx_verified']:
        raise ValueError('CPU tail hash mismatch')
    return metadata, calibration


def convert(checkpoint, export_json, calibration_json, stage_number, directory):
    from rknn.api import RKNN
    import onnxruntime as ort
    metadata, calibration = checked_inputs(export_json, calibration_json, checkpoint)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    stage, data = metadata['stages'][stage_number - 1], calibration['datasets'][stage_number - 1]
    def check(status, operation):
        if status != 0 and not (status is None and operation == 'config'):
            raise RuntimeError(f'RKNN {operation} returned {status}')
    rknn = RKNN(verbose=True)
    try:
        check(rknn.config(target_platform='rv1106'), 'config')
        check(rknn.load_onnx(model=stage['onnx']), 'load_onnx')
        check(rknn.build(do_quantization=True, dataset=data['dataset']), 'build')
        path = directory / (stage['name'] + '.rknn')
        check(rknn.export_rknn(str(path)), 'export_rknn')
        write_json(directory / 'conversion.json', dict(schema=1, stage=stage_number,
            name=stage['name'], conversion_completed=True, target='rv1106', quantized=True,
            checkpoint_sha256=metadata['checkpoint_sha256'], onnx_sha256=stage['onnx_sha256'],
            rknn_sha256=_sha256(path), rknn_bytes=path.stat().st_size,
            calibration_sha256=_sha256(calibration_json), board_verified=False))
        check(rknn.init_runtime(), 'init_runtime_simulator')
        session = ort.InferenceSession(stage['onnx'], providers=['CPUExecutionProvider'])
        probes = []
        for item in data['samples'][:5]:
            sample = np.load(item['path'], allow_pickle=False)
            actual = rknn.inference(inputs=[sample], data_format=['nchw'])
            expected = session.run(None, {'features': sample})[0]
            if actual is None or len(actual) != 1 or actual[0].shape != expected.shape or not np.isfinite(actual[0]).all():
                raise RuntimeError('simulator output contract failed')
            delta = np.abs(actual[0] - expected)
            probes.append(dict(sample_sha256=item['sha256'], max_abs=float(delta.max()), mean_abs=float(delta.mean())))
        write_json(directory / 'simulator_probe.json', dict(executed=True, simulation_only=True,
            board_verified=False, stage=stage_number, name=stage['name'], probes=probes))
        print(json.dumps(dict(stage=stage_number, name=stage['name'], converted=True,
                              simulated=True, board_verified=False)), flush=True)
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
    parser.add_argument('--stage', type=int, choices=range(1, 7))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase == 'export':
        export(args.checkpoint, args.output_dir)
    elif args.phase == 'calibrate':
        calibrate(args.checkpoint, args.train_h5, args.output_dir)
    else:
        convert(args.checkpoint, args.export_json, args.calibration_json, args.stage, args.output_dir)


if __name__ == '__main__':
    main()
