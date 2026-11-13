# KTCAL

Event-based eye tracking.

### Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-ktcal.txt
```

### Dataset and weights

3ET+ 2025: [official dataset](https://www.kaggle.com/competitions/event-based-eye-tracking-cvpr-2025/data) · [processed 100-frame data (Baidu; code: hsft)](https://pan.baidu.com/s/11-zaux8WKvGqIOBQuUlK3g?pwd=hsft).

Place `test_aug.h5` in `data/3et/`. Pretrained weights are included in `weights/3et/`. Dataset files are not included.

### Evaluation

From the repository root:

```bash
python -m scripts.evaluate_3et
```

Prints P3/P5/P10, mean pixel error and ms/frame. Results are saved to `local_run/3et_val.json`.

To use another dataset location or rerun with a new output file:

```bash
python -m scripts.evaluate_3et --h5 /path/to/test_aug.h5 --output local_run/3et_val_2.json
```

### Online evaluation

The online entry point uses only received frames and measures sequential, batch-one processing with GPU completion at every frame. The command above evaluates complete sequences and its ms/frame is an offline average.

The recommended online profile is **KF-majority**, using the included causal weight `weights/3et/stnet_online_current_readout.pt`:

```bash
python -m scripts.evaluate_3et_online \
  --config configs/3et_online_kf_majority.json \
  --h5 /path/to/test_aug.h5 \
  --repeats 3 --save-predictions \
  --output local_run/3et_online_kf_majority.json
```

This profile uses three causal convolution blocks, a unidirectional GRU and a direct current-feature coordinate readout. It updates CNN/GRU state every frame and requests coordinate heads when matching confidence or displacement requires correction. Six spatial views are combined by median; event-count reset gating and output EMA are disabled. Matching supplies KF observations on 5,192 of 9,300 frames (**55.83%**); the remaining 4,108 frames request neural coordinates.

Three full RTX 3090 validation runs with the packaged weights measured **84.5269% P3**, **2.087043 px** mean error and **1.224675 ms/frame** (median of complete-run averages). P3 is the proportion of frames within 3 px. Timing includes every-frame feature updates, requested coordinate heads, matching, KF, transfers and synchronization, including four repeated-first-frame startup updates per recording. It excludes loading, scoring and CUDA Graph capture. Inputs/weights are float32; cuDNN TF32 is enabled and matmul TF32 is disabled. This validation set was used for selection; the final holdout was not used.

See [KF-majority results and reproduction](docs/online/2026-11-13-kf-majority-results.md) and the [weight manifest](weights/3et/online_kf_majority_manifest.json). Explicit CLI options override JSON defaults; existing result files are never overwritten. Earlier two-view and ten-view profiles are documented in the [previous online report](docs/online/2026-11-13-online-results.md).

### Verification

```bash
python -m scripts.verify_3et --expected-report local_run/3et_val.json --output local_run/3et_val_verified.json
python -m unittest discover -s tests -t . -p 'test_*.py'
```
