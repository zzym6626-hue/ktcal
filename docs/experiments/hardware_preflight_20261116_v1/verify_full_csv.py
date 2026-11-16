"""Independent CSV coverage/scoring pass, with published-reference comparison."""
import csv
import json
from pathlib import Path
import h5py
import numpy as np
from hardware.rv1106.records import score_records
from ktcal_paper.combined_3et import _sha256, validate_validation_h5

base = Path('/root/ktcal_hardware_preflight_20261116_v1/artifacts')
full = base/'full9300_v2'
h5 = Path('/root/autodl-tmp/TDTracker-main/TDTracker-main/3ET+2025/test_aug.h5')
if h5.name != 'test_aug.h5':
    raise ValueError('test_aug only')
report = json.loads((full/'report.json').read_text())
if not report['passed'] or not report['complete'] or report['frames'] != 9300:
    raise ValueError('successful complete report required')
if _sha256(h5) != report['h5_sha256']:
    raise ValueError('input provenance differs')
scored = score_records(h5,full/'predictions.csv',base/'cpp_csv_scored_v3.json')
for key in ['p3','p5','p10','mean_error_px','frames']:
    if scored['metrics'][key] != report['metrics']['onnx_cpp_ktcal_ma7'][key]:
        raise ValueError('separate score disagreement '+key)
with (full/'predictions.csv').open() as f:
    rows = list(csv.DictReader(f))
if len(rows) != 9300 or [(int(r['record']),int(r['frame'])) for r in rows] != [(r,t) for r in range(93) for t in range(100)]:
    raise ValueError('exact complete ordered coverage required')
with h5py.File(h5,'r') as h:
    _, label_key, _, _ = validate_validation_h5(h5)
    labels = h[label_key][:,:,:2].astype(np.float32).reshape(-1,2).astype(np.float64)*[80.,60.]
metrics, errors = {}, {}
for name,x,y in [('onnx_cpp_ktcal_ma7','x_px','y_px'),('pytorch_ktcal_ma7','pytorch_x_px','pytorch_y_px'),
                 ('onnx_cpp_stnet_ma7','stnet_x_px','stnet_y_px'),('pytorch_stnet_ma7','pytorch_stnet_x_px','pytorch_stnet_y_px')]:
    coords = np.array([[float(r[x]),float(r[y])] for r in rows],dtype=np.float64)
    if not np.isfinite(coords).all():
        raise ValueError('finite coordinates required')
    error = np.linalg.norm(coords-labels,axis=1)
    errors[name] = error
    values = dict(frames=9300,mean_error_px=float(error.mean()),**{f'p{k}':float(np.mean(error<k)) for k in [3,5,10]})
    for key in ['p3','p5','p10']:
        if values[key] != report['metrics'][name][key]:
            raise ValueError('independent pixel scoring disagreement '+name+' '+key)
    # CSV holds pixels. The PyTorch metrics divide float32 pixels by SCALE
    # before float64 scoring, producing a measured <6e-9px aggregate drift.
    if abs(values['mean_error_px']-report['metrics'][name]['mean_error_px']) > 1e-7:
        raise ValueError('independent mean disagreement '+name)
    metrics[name] = values
classification_mismatches = {}
for suffix in ['stnet_ma7','ktcal_ma7']:
    classification_mismatches[suffix] = {f'p{k}':int(np.count_nonzero((errors['pytorch_'+suffix]<k)!=(errors['onnx_cpp_'+suffix]<k))) for k in [3,5,10]}
    if any(classification_mismatches[suffix].values()):
        raise ValueError('individual threshold classifications differ '+suffix)
reference_path = base/'published_reference.json'
reference = json.loads(reference_path.read_text())
release_deltas = {}
for model in ['stnet','ktcal']:
    actual = metrics['onnx_cpp_'+model+'_ma7']; old = reference['metrics'][model]
    release_deltas[model] = {key:actual[key]-old[key] for key in ['p3','p5','p10','mean_error_px']}
    # Historical reference uses CUDA, while this parity pass uses CPU.
    # Record (do not hide) platform differences separately from stage parity.
gate_deltas = {}
for key in ['tier1','tier2','tier3','medium_refinements','medium_rollbacks','fusion_calls','small_fusion_calls','stnet_requests','border_peaks']:
    gate_deltas[key] = report['tracker_stats'][key]-reference['tracker_stats'][key]
verification = dict(schema=1,passed=True,frames=9300,report_sha256=_sha256(full/'report.json'),
                    published_reference_sha256=_sha256(reference_path),
                    predictions_sha256=_sha256(full/'predictions.csv'),h5_sha256=_sha256(h5),
                    independently_scored_metrics=metrics,pixel_csv_mean_tolerance_px=1e-7,
                    threshold_classification_mismatches=classification_mismatches,
                    published_release_deltas=release_deltas,published_gate_count_deltas=gate_deltas,
                    published_gate_counts_equal=not any(gate_deltas.values()),
                    scope='CPU PyTorch/ONNX+C++ parity; historical CUDA differences recorded, not certified equal',board_execution=False)
with (base/'full9300_independent_verification_v1.json').open('x') as f:
    json.dump(verification,f,indent=2,allow_nan=False)
print(json.dumps(verification,indent=2))
