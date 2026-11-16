"""Partition record41 error: backend logits vs same-logit C++ decoding."""
import json
from pathlib import Path
import h5py
import numpy as np
import onnxruntime as ort
import torch
from ktcal_paper.combined_3et import load_stnet_checkpoint, SCALE
from ktcal_paper.evaluation_protocol import soft_expectation_decode
from ktcal_paper.offline_experiment import spatial_view, undo_view, average_views, smooth_pixels
from evaluate_full import VIEWS

base = Path('/root/ktcal_hardware_preflight_20261116_v1')
old = base / 'artifacts/full9300_v1'
torch.set_num_threads(4)
model, config, divisor = load_stnet_checkpoint('weights/3et/stnet_best.pt', 'cpu')
options = ort.SessionOptions(); options.intra_op_num_threads = 4
session = ort.InferenceSession('ktcal_stnet100_20261116_v1.onnx', options, providers=['CPUExecutionProvider'])
with h5py.File('/root/autodl-tmp/TDTracker-main/TDTracker-main/3ET+2025/test_aug.h5', 'r') as h:
    raw = np.asarray(h['frames'][41:42], dtype=np.float32)
np.testing.assert_array_equal(raw.reshape(-1), np.fromfile(old/'raw.bin', dtype='<f4'))
saved = np.fromfile(old/'logits.bin', dtype='<f4').reshape(10,100,140)
cpp = np.fromfile(old/'cpp.bin', dtype='<f8').reshape(100,9)
py_views, ort_views = [], []
logit_error = 0.
with torch.inference_mode():
    for i, view in enumerate(VIEWS):
        inputs = spatial_view(torch.from_numpy(raw), view).contiguous()
        lx, ly = model(inputs/divisor)
        ox, oy = session.run(None, {'raw_frames': inputs.numpy()})
        np.testing.assert_array_equal(np.concatenate([ox,oy],axis=-1)[0], saved[i])
        logit_error = max(logit_error, float(np.max(np.abs(lx.numpy()-ox))), float(np.max(np.abs(ly.numpy()-oy))))
        py_views.append(undo_view(soft_expectation_decode(lx/.65,ly/.65).astype(np.float32)*SCALE,view))
        ort_views.append(undo_view(soft_expectation_decode(ox/.65,oy/.65).astype(np.float32)*SCALE,view))
py = average_views(np.stack(py_views),list(range(10)),method='median')[0]
op = average_views(np.stack(ort_views),list(range(10)),method='median')[0]
report = dict(record=41, logits_max_abs=logit_error,
              onnx_vs_torch_decoded_max_px=float(np.max(np.abs(op-py))),
              cpp_vs_same_onnx_logits_max_px=float(np.max(np.abs(cpp[:,:2]-op))),
              cpp_vs_torch_decoded_max_px=float(np.max(np.abs(cpp[:,:2]-py))),
              cpp_vs_same_onnx_ma7_max_px=float(np.max(np.abs(cpp[:,7:9]-smooth_pixels(op[None],'ma7')[0]))),
              cpp_vs_torch_ma7_max_px=float(np.max(np.abs(cpp[:,7:9]-smooth_pixels(py[None],'ma7')[0]))))
print(json.dumps(report,indent=2))
with (base/'artifacts/record41_numerical_diagnosis_v1.json').open('x') as f:
    json.dump(report,f,indent=2)
