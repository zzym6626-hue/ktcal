"""Export a 16-frame causal STNet checkpoint to ONNX last-frame logits.

ONNX is an intermediate only: use a separately installed RKNN-Toolkit2 and
validate converted model outputs on the RV1106 before deploying it.
"""

import argparse
import importlib.util
import math
from pathlib import Path

import torch

from ktcal_paper.combined_3et import load_stnet_checkpoint


def validate_board_config(config):
    if config.get('window') != 16 or config.get('causal_last_step') is not True:
        raise ValueError('board export requires a 16-frame causal checkpoint')


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


def export_stnet(checkpoint, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    # Check metadata before loading weights: offline best.pt is not a board model.
    metadata = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(metadata, dict):
        raise ValueError('checkpoint must be a dictionary')
    validate_board_config(metadata.get('config', {}))
    if importlib.util.find_spec('onnx') is None:
        raise RuntimeError('ONNX export requires the onnx Python package')
    model, _, divisor = load_stnet_checkpoint(checkpoint, 'cpu')
    bridge = LastFrameLogits(model, divisor).eval()
    sample = torch.zeros((1, 16, 2, 60, 80), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(bridge, sample, str(output), opset_version=17,
                          input_names=['raw_frames'], output_names=['x_logits', 'y_logits'],
                          do_constant_folding=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    options = parser.parse_args()
    export_stnet(options.checkpoint, options.output)


if __name__ == '__main__':
    main()
