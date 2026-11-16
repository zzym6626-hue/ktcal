"""Isolated CPU-readout RKNN feasibility; training-only calibration, no board claim."""
import json
from pathlib import Path
import h5py
import numpy as np
import onnxruntime as ort
import torch
from rknn.api import RKNN
from temporal_probe import load, write_json
from ktcal_paper.combined_3et import _sha256
from ktcal_paper.offline_experiment import spatial_view

ROOT = Path('/root/ktcal_hardware_temporal_split_20261116_v1')
SOURCE = Path('/root/ktcal_hardware_preflight_20261116_v1/source')
TRAIN = Path('/root/autodl-tmp/TDTracker-main/TDTracker-main/3ET+2025/train_aug.h5')


def main():
    torch.set_num_threads(4)
    out = ROOT / 'tail_probe_v1'
    out.mkdir(exist_ok=False)
    ck = SOURCE / 'weights/3et/stnet_best.pt'
    model, split, _ = load(ck)
    model = model.cuda()
    samples = []
    with h5py.File(TRAIN, 'r') as handle, torch.inference_mode():
        for record in range(8):
            frames = torch.from_numpy(np.asarray(handle['frames'][record:record+1], dtype=np.float32)).cuda()
            for flip in (False, True):
                value = spatial_view(frames, (0, 0, flip)) / split.divisor
                value = value.permute(0, 2, 1, 3, 4)
                for block in (model.block1, model.block2, model.block3):
                    value = block(value)
                value = value.permute(0, 2, 1, 3, 4).contiguous().cpu().numpy()
                path = out / f'train_r{record}_m{int(flip)}.npy'
                with path.open('xb') as file:
                    np.save(file, value, allow_pickle=False)
                samples.append(dict(record=record, flip=flip, path=str(path), sha256=_sha256(path)))
    dataset = out / 'train_inputs.txt'
    dataset.write_text(''.join(item['path']+'\n' for item in samples))
    meta = json.loads((ROOT/'export_v1/export.json').read_text())
    tail = meta['cpu_tail']
    if _sha256(tail['onnx']) != tail['onnx_sha256'] or _sha256(ck) != meta['checkpoint_sha256']:
        raise ValueError('source hash mismatch')
    write_json(out/'calibration.json', dict(source=str(TRAIN), source_h5_sha256=_sha256(TRAIN),
        checkpoint_sha256=_sha256(ck), onnx_sha256=tail['onnx_sha256'], dataset_sha256=_sha256(dataset),
        keys_read=['frames'], labels_read=False, validation_used=False, samples=samples))
    def check(code, operation):
        if code != 0 and not (code is None and operation == 'config'):
            raise RuntimeError(f'{operation} returned {code}')
    rknn = RKNN(verbose=True)
    try:
        check(rknn.config(target_platform='rv1106'), 'config')
        check(rknn.load_onnx(model=tail['onnx']), 'load_onnx')
        check(rknn.build(do_quantization=True, dataset=str(dataset)), 'build')
        output = out / 'readout100.rknn'
        check(rknn.export_rknn(str(output)), 'export_rknn')
        write_json(out/'conversion.json', dict(converted=True, target='rv1106',
            rknn_sha256=_sha256(output), bytes=output.stat().st_size, board_verified=False))
        check(rknn.init_runtime(), 'init_runtime_simulator')
        session = ort.InferenceSession(tail['onnx'], providers=['CPUExecutionProvider'])
        probes = []
        for sample in samples[:2]:
            x = np.load(sample['path'], allow_pickle=False)
            actual = rknn.inference(inputs=[x], data_format=['nchw'])
            reference = session.run(None, {'post_block3':x})
            if actual is None or len(actual) != 2:
                raise RuntimeError('missing tail outputs')
            errors = []
            for a,b in zip(actual,reference):
                if a.shape != b.shape or not np.isfinite(a).all():
                    raise RuntimeError('tail output contract failed')
                errors.append(dict(max_abs=float(np.max(np.abs(a-b))), mean_abs=float(np.mean(np.abs(a-b)))))
            probes.append(dict(sample_sha256=sample['sha256'], errors=errors))
        write_json(out/'simulation.json', dict(executed=True, simulation_only=True,
            board_verified=False, probes=probes))
    finally:
        rknn.release()


if __name__ == '__main__':
    main()
