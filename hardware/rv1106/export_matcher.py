"""Export OUR trained matcher extractor to the RV1106 C++ float fast path.

This exports only the 3x3 depthwise / 1x1 pointwise weights; it does NOT
convert or quantize STNet. No source checkpoint is modified.
"""

import argparse
import struct
from pathlib import Path

import torch

from ktcal_paper.matching import PaperMatcher


def export_matcher(checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or 'matcher_state' not in checkpoint:
        raise ValueError('checkpoint must contain matcher_state')
    channels = int(checkpoint.get('config', {}).get('ncc_channels', 16))
    if not 1 <= channels <= 256:
        raise ValueError('unsupported matcher channel count')
    model = PaperMatcher(channels=channels).eval()
    model.load_state_dict(checkpoint['matcher_state'], strict=True)
    depth = model.extractor[0].weight.detach().contiguous().flatten().numpy()
    point = model.extractor[1].weight.detach().contiguous().flatten().numpy()
    if not torch.isfinite(model.extractor[0].weight).all() or not torch.isfinite(model.extractor[1].weight).all():
        raise ValueError('matcher weights must be finite')
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    output.write_bytes(b'KTMATCH1' + struct.pack('<I', channels) +
                       depth.astype('<f4').tobytes() + point.astype('<f4').tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export_matcher(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
