# KTCAL

Event-based eye tracking with STNet prediction and KTCAL template matching + Kalman correction.

### Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-ktcal.txt
```

### Dataset

Prepare the 3ET+2025 HDF5 files and place `train_aug.h5` and `test_aug.h5` in `data/3et/`. **Data and pretrained 3ET checkpoints are not published in this repository**: train the two models below or supply your own compatible `best.pt` files. Keep `test_200_200.h5` as the final holdout; do not use it for training or model selection.

### Usage

1. Train STNet (100-frame model):

   ```bash
   python -m scripts.train_stnet_3et --train-h5 data/3et/train_aug.h5 \
     --val-h5 data/3et/test_aug.h5 --output local_run/stnet \
     --window 100 --readout attention --selection-metric soft_ma11 \
     --epochs 60 --device cuda
   ```

2. Train the KTCAL matcher (skip if you already have its `best.pt`):

   ```bash
   python -m scripts.finetune_3et_matcher --train-h5 data/3et/train_aug.h5 \
     --validation-h5 data/3et/test_aug.h5 \
     --stnet-checkpoint local_run/stnet/best.pt \
     --source-checkpoint local_run/ais_ktcal/bidirectional_best.pt \
     --output local_run/matcher --device cuda
   ```

   The source must be **our KTCAL** checkpoint containing `model_state` and `matcher_state`; see `python -m ktcal_paper --help` to prepare/train one.

3. Evaluate STNet + KTCAL (P3/P5/P10 and measured ms/frame):

   ```bash
   python -m scripts.evaluate_3et_combined_ktcal --h5 data/3et/test_aug.h5 --stnet-checkpoint local_run/stnet/best.pt --matcher-checkpoint local_run/matcher/best.pt --output local_run/combined_val.json --stnet-decode soft --smoothing ma11 --device cuda
   ```

The terminal prints frame count, mean pixel error, P3/P5/P10 (%) and ms/frame for STNet and STNet + KTCAL. The JSON contains full metrics, component timings and file hashes. Timing uses 100-frame sequences after warm-up; data reads and model loading are excluded. `test_aug.h5` is a validation set, not an independent test result.

### RV1106

Experimental C++ CPU fast path and optional RKNN backend: `hardware/rv1106/README.md`. Board SDK, converted model, and on-device performance measurements are not included.
