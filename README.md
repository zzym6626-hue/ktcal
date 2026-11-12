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

### Verification

```bash
python -m scripts.verify_3et --expected-report local_run/3et_val.json --output local_run/3et_val_verified.json
python -m unittest discover -s tests -t . -p 'test_*.py'
```
