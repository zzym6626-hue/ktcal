# KTCAL

Event-based eye tracking with STNet prediction and KTCAL template matching + Kalman correction.

### Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-ktcal.txt
```

### Dataset

Place the 3ET+2025 `test_aug.h5` in `data/3et/`; add `train_aug.h5` if training. The two inference checkpoints are in `weights/3et/`. Dataset files are not included. Keep `test_200_200.h5` as the final holdout; do not use it for training or model selection.

### Usage

Run the evaluation to print mean pixel error, P3/P5/P10 and ms/frame for STNet and STNet + KTCAL:

```bash
python -m scripts.evaluate_3et_combined_ktcal --h5 data/3et/test_aug.h5 --stnet-checkpoint weights/3et/stnet_best.pt --matcher-checkpoint weights/3et/matcher_best.pt --output local_run/combined_val.json --stnet-decode soft --smoothing ma11 --fusion-alpha 0.75 --device cuda
```

Full metrics, timing details and file hashes are saved to `local_run/combined_val.json`. Timing uses 100-frame sequences after warm-up; data reads and model loading are excluded. `test_aug.h5` is a validation set, not an independent test result.

To retrain (optional):

1. Train STNet:

   ```bash
   python -m scripts.train_stnet_3et --train-h5 data/3et/train_aug.h5 \
     --val-h5 data/3et/test_aug.h5 --output local_run/stnet \
     --window 100 --readout attention --selection-metric soft_ma11 \
     --epochs 60 --device cuda
   ```

2. Train the KTCAL matcher:

   ```bash
   python -m scripts.finetune_3et_matcher --train-h5 data/3et/train_aug.h5 \
     --validation-h5 data/3et/test_aug.h5 \
     --stnet-checkpoint local_run/stnet/best.pt \
     --source-checkpoint local_run/ais_ktcal/bidirectional_best.pt \
     --output local_run/matcher --device cuda
   ```

   The source must be **our KTCAL** checkpoint containing `model_state` and `matcher_state`; see `python -m ktcal_paper --help` to prepare/train one.

### RV1106

Experimental C++ CPU fast path and optional RKNN backend: `hardware/rv1106/README.md`. Board SDK, converted model, and on-device performance measurements are not included.
