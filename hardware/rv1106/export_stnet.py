"""Export released 100-frame STNet to static ONNX, without changing weights.

ONNX is an intermediate only: use a separately installed RKNN-Toolkit2 and
validate converted model outputs on the RV1106 before deploying it.
"""

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path

import torch

from ktcal_paper.combined_3et import _sha256, load_stnet_checkpoint


def validate_board_config(config):
    """Legacy causal16 contract, available only with --mode causal16."""
    if config.get('window') != 16 or config.get('causal_last_step') is not True:
        raise ValueError('board export requires a 16-frame causal checkpoint')


def validate_sequence_config(config):
    expected = dict(window=100, readout='attention', temporal_variant='dense',
                    temporal_spatial_kernel=1, block3_kernel=5,
                    unidirectional=False)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('sequence100 export requires the released dense attention 100-frame config')
    divisor = float(config.get('input_divisor', 1.))
    if not math.isfinite(divisor) or divisor <= 0:
        raise ValueError('input divisor must be finite and positive')
    if config.get('causal_last_step', False):
        raise ValueError('sequence100 export cannot use causal last-step checkpoints')


class StaticSpatialPool(torch.nn.Module):
    """Exact 6x8 -> 4x4 adaptive spatial average using static slices.

    STNet pools [B,T,128,6,8] to [B,T,128,4,4]; channels never mix.
    Unequal adaptive H bins are not representable by one ONNX AveragePool.
    """

    def forward(self, features):
        rows = []
        for row in range(4):
            begin, end = row * 6 // 4, ((row + 1) * 6 + 3) // 4
            cells = [features[..., begin:end, col * 2:col * 2 + 2].mean((-2, -1))
                     for col in range(4)]
            rows.append(torch.stack(cells, -1))
        return torch.stack(rows, -2)


class FullFrameLogits(torch.nn.Module):
    """Raw [1,100,2,60,80] -> [1,100,80], [1,100,60], all timesteps."""

    def __init__(self, model, input_divisor):
        super().__init__()
        if not math.isfinite(input_divisor) or input_divisor <= 0:
            raise ValueError('input divisor must be finite and positive')
        self.model = copy.deepcopy(model).eval()
        if hasattr(self.model, 'pool'):
            self.model.pool = StaticSpatialPool()
        self.input_divisor = float(input_divisor)

    def forward(self, frames):
        return self.model(frames.float() / self.input_divisor)


class LastFrameLogits(torch.nn.Module):
    """Adapt [B,16,2,60,80] raw counts to [B,80] and [B,60] logits."""

    def __init__(self, model, input_divisor):
        super().__init__()
        if not math.isfinite(input_divisor) or input_divisor <= 0:
            raise ValueError('input divisor must be finite and positive')
        self.model = model.eval()
        self.input_divisor = float(input_divisor)

    def forward(self, frames):
        logits_x, logits_y = self.model(frames.float() / self.input_divisor)
        return logits_x[:, -1], logits_y[:, -1]


def export_stnet(checkpoint, output, mode='sequence100'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    sidecar = output.with_suffix(output.suffix + '.json')
    if sidecar.exists():
        raise FileExistsError(f'refusing to overwrite {sidecar}')
    metadata = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(metadata, dict):
        raise ValueError('checkpoint must be a dictionary')
    if mode not in ('sequence100', 'causal16'):
        raise ValueError('mode must be sequence100 or causal16')
    config = metadata.get('config', {})
    (validate_sequence_config if mode == 'sequence100' else validate_board_config)(config)
    if importlib.util.find_spec('onnx') is None:
        raise RuntimeError('ONNX export requires the onnx Python package')
    model, _, divisor = load_stnet_checkpoint(checkpoint, 'cpu')
    bridge = (FullFrameLogits if mode == 'sequence100' else LastFrameLogits)(model, divisor).eval()
    length = 100 if mode == 'sequence100' else 16
    sample = torch.zeros((1, length, 2, 60, 80), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(bridge, sample, str(output), opset_version=17,
                          input_names=['raw_frames'], output_names=['x_logits', 'y_logits'],
                          do_constant_folding=True)
    with sidecar.open('x') as handle:
        json.dump(dict(schema=1, mode=mode, checkpoint_sha256=_sha256(checkpoint),
                       onnx_sha256=_sha256(output), config=config,
                       input_shape=list(sample.shape),
                       output_shapes=([[1, 100, 80], [1, 100, 60]] if length == 100
                                      else [[1, 80], [1, 60]]),
                       input_divisor_embedded=divisor, runtime_input_divisor=1.,
                       rknn_conversion_verified=False, board_verified=False), handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('sequence100', 'causal16'), default='sequence100')
    options = parser.parse_args()
    export_stnet(options.checkpoint, options.output, options.mode)


if __name__ == '__main__':
    main()
