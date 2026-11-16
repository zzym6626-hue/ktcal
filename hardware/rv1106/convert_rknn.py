"""Attempt conversion of verified STNet100 ONNX; fail closed on SDK errors."""
import argparse
import json
from pathlib import Path

from ktcal_paper.combined_3et import _sha256


def convert(onnx_path, verified_report, output, target='rv1106', dataset=None):
    output = Path(output)
    sidecar = output.with_suffix(output.suffix + '.json')
    if output.exists() or sidecar.exists():
        raise FileExistsError(f'refusing to overwrite {output} or its metadata')
    onnx_path = Path(onnx_path)
    verified = json.loads(Path(verified_report).read_text())
    if not verified.get('passed') or verified.get('onnx_sha256') != _sha256(onnx_path):
        raise ValueError('successful hash-matched ONNX verification required')
    metadata = json.loads(onnx_path.with_suffix(onnx_path.suffix + '.json').read_text())
    if (metadata.get('mode') != 'sequence100'
            or metadata.get('onnx_sha256') != verified['onnx_sha256']
            or metadata.get('checkpoint_sha256') != verified.get('checkpoint_sha256')
            or metadata.get('runtime_input_divisor') != 1.):
        raise ValueError('sequence100 export metadata mismatch')
    if dataset is not None:
        dataset = Path(dataset).resolve()
        if not dataset.is_file():
            raise FileNotFoundError(dataset)
        # Calibration must be explicitly supplied from training raw inputs;
        # do not use labels, validation records or final holdout for fitting.
        if 'test_' in dataset.name.lower():
            raise ValueError('calibration must come from training records')
    from rknn.api import RKNN
    rknn = RKNN(verbose=True)
    try:
        def check(status, stage):
            if status != 0 and not (stage == 'config' and status is None):
                raise RuntimeError(f'RKNN {stage} failed ({status}); do not use the output')
        # No mean/std normalization: ONNX already embeds the checkpoint divisor.
        check(rknn.config(target_platform=target), 'config')
        check(rknn.load_onnx(model=str(onnx_path)), 'load_onnx')
        check(rknn.build(do_quantization=dataset is not None,
                         dataset=str(dataset) if dataset is not None else None), 'build')
        check(rknn.export_rknn(str(output)), 'export_rknn')
        report = dict(schema=1, target=target, mode='sequence100',
                      checkpoint_sha256=metadata['checkpoint_sha256'],
                      onnx_sha256=metadata['onnx_sha256'], rknn_sha256=_sha256(output),
                      quantized=dataset is not None, conversion_completed=True,
                      rknn_outputs_verified=False, board_verified=False)
        with sidecar.open('x') as handle:
            json.dump(report, handle, indent=2)
        return report
    finally:
        rknn.release()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--verified-report', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target', default='rv1106')
    parser.add_argument('--quantization-dataset', help='Optional SDK list of training raw .npy inputs')
    args = parser.parse_args()
    print(json.dumps(convert(args.onnx, args.verified_report, args.output,
                             args.target, args.quantization_dataset), indent=2))


if __name__ == '__main__':
    main()
