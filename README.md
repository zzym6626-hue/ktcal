# KTCAL

Event-based eye tracking with dense STNet, template matching and Kalman correction.

### Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-ktcal.txt
```

### Dataset and weights

3ET+ 2025: [official dataset](https://www.kaggle.com/competitions/event-based-eye-tracking-cvpr-2025/data) · [processed 100-frame data (Baidu; code: hsft)](https://pan.baidu.com/s/11-zaux8WKvGqIOBQuUlK3g?pwd=hsft).

Place `test_aug.h5` in `data/3et/`. Both inference checkpoints are included in `weights/3et/`; hashes are recorded in `weights/3et/manifest.json`. Dataset files are not included.

### Evaluation

From the repository root:

```bash
python -m scripts.evaluate_3et
```

Prints STNet and STNet + KTCAL mean pixel error, P3/P5/P10, ms/frame and route percentages. The full report is saved to `local_run/3et_val.json`. For another run, choose a new filename with `--output local_run/3et_val_2.json`; use `--h5 /path/to/test_aug.h5` for a different dataset location.

Both rows use 10-view soft decoding (temperature 0.65), coordinate median and MA7.

| Method | P3 | P5 | P10 | Mean error | ms/frame |
| --- | --- | --- | --- | --- | --- |
| STNet | 90.31% | 96.18% | 99.25% | 1.629 px | 0.362 |
| **STNet + KTCAL** | **91.11%** | **96.45%** | **99.25%** | **1.614 px** | **1.05** |

Reference: RTX 3090, FP32, one 100-frame sequence per input, 9,300 validation frames. Timing includes all STNet views, transfers, matching, KF and smoothing after warm-up; disk reads, model loading and first CUDA Graph capture are excluded. ms/frame is sequence-processing time divided by frame count, not a causal per-frame response measurement. Settings were selected on `test_aug.h5`; these are validation results, not an independent test. `test_200_200.h5` remains excluded.

### Pipeline

100 frames → STNet 10-view soft decoding + coordinate median → KF prediction + template matching → gated correction → per-record MA7.

- **Small:** reliable match with prediction–match distance <3 px → update KF with the match and output KF only; **no current-frame STNet fusion**.
- **Medium:** reliable distance 3–5 px → compare updated KF with STNet: ≤1 px refines with STNet (fusion weight 0.75); >1 px resets to STNet.
- **Large/unreliable:** distance ≥5 px, low confidence/PSR or border peak → STNet reset and template refresh.

[Reference results and checkpoint provenance](docs/releases/kf60_20261112/README.md).

Raw routes: **60.73% KF**, 0.29% medium, 38.98% reset/initialization. Template/search sizes are 30×25/37×32. STNet coordinates are precomputed for all 100 frames; the KF percentage is not a STNet-computation skip percentage. Gate distance is not error against ground truth.

### Verification

```bash
python -m scripts.verify_3et --expected-report local_run/3et_val.json --output local_run/3et_val_verified.json
python -m unittest discover -s tests -t . -p 'test_*.py'
```

The verifier reloads checkpoints, checks hashes and metrics/route counts, and verifies raw KF-route outputs remain unchanged when current-frame STNet coordinates are perturbed.

### RV1106

Experimental CPU fast path and optional RKNN backend: [hardware guide](hardware/rv1106/README.md). SDK, converted model and board measurements are not included; the GPU figures above are not board performance.
