"""Check exported full-frame ONNX against its unmodified PyTorch checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ktcal_paper.combined_3et import _sha256, load_stnet_checkpoint
from .export_stnet import validate_sequence_config


def validate_export_manifest(checkpoint, onnx_path):
    onnx_path = Path(onnx_path)
    metadata = json.loads(onnx_path.with_suffix(onnx_path.suffix + '.json').read_text())
    if (metadata.get('checkpoint_sha256') != _sha256(checkpoint)
            or metadata.get('onnx_sha256') != _sha256(onnx_path)):
        raise ValueError('checkpoint/ONNX hash mismatch')
    if metadata.get('mode') != 'sequence100' or metadata.get('runtime_input_divisor') != 1.:
        raise ValueError('expected sequence100 export with embedded normalization')
    validate_sequence_config(metadata['config'])
    return metadata


def verify(checkpoint, onnx_path, output, frames=None):
    """Inputs are raw [N,100,2,60,80]; targets never enter this check."""
    import onnx
    import onnxruntime as ort
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    metadata = validate_export_manifest(checkpoint, onnx_path)
    onnx.checker.check_model(str(onnx_path))
    torch.set_num_threads(4)
    model, config, divisor = load_stnet_checkpoint(checkpoint, 'cpu')
    validate_sequence_config(config)
    if frames is None:
        rng = np.random.RandomState(20261116)
        cases = np.concatenate((np.zeros((1, 100, 2, 60, 80), np.float32),
                                rng.poisson(1.5, (1, 100, 2, 60, 80)).astype(np.float32)))
        input_source = 'synthetic_zero_and_poisson; not accuracy evaluation'
    else:
        cases = np.load(frames, allow_pickle=False)
        input_source = str(frames)
    if (cases.ndim != 5 or cases.shape[1:] != (100, 2, 60, 80)
            or not len(cases) or not np.isfinite(cases).all()):
        raise ValueError('finite raw [N,100,2,60,80] frames required')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].shape != [1, 100, 2, 60, 80]
            or [value.shape for value in outputs] != [[1, 100, 80], [1, 100, 60]]):
        raise ValueError('ONNX static input/output contract mismatch')
    errors = []
    with torch.inference_mode():
        for case in cases:
            raw = np.ascontiguousarray(case[None], dtype=np.float32)
            expected = [value.numpy() for value in model(torch.from_numpy(raw) / divisor)]
            actual = session.run(None, {inputs[0].name: raw})
            for reference, result in zip(expected, actual):
                np.testing.assert_allclose(result, reference, atol=2e-4, rtol=2e-4)
                errors.append(float(np.max(np.abs(result - reference))))
    report = dict(schema=1, passed=True, checkpoint_sha256=_sha256(checkpoint),
                  onnx_sha256=metadata['onnx_sha256'], records=len(cases),
                  input_source=input_source, max_logit_error=max(errors),
                  atol=2e-4, rtol=2e-4, runtime='ONNX Runtime CPU',
                  rknn_conversion_verified=False, board_verified=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', default='weights/3et/stnet_best.pt')
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', help='Optional raw float32 records in .npy')
    args = parser.parse_args()
    verify(args.checkpoint, args.onnx, args.output, args.frames)


if __name__ == '__main__':
    main()
