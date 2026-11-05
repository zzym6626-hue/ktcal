# Code Map: KTCal paper software experiments

Generated: 2026-11-05. Scope: new `ktcal_paper` package. Goal: trace true-label preprocessing, training/export and online evaluation. Destination: user Git repository `IdeaProjects/tdtracker`; no remote repository Wiki was used.

## Entry points

- `ktcal_paper/__main__.py:6` — CLI `prepare`, `train`, `evaluate`, `diagnose`.
- `ktcal_paper/data.py:49` — raw AIS2024 labeled train records to HDF5 and split manifest.
- `ktcal_paper/experiment.py:115` — new or resumed training, best selection and two-epoch forward fine-tune.
- `ktcal_paper/experiment.py:302` — per-record online evaluation and measured result files.
- `scripts/ktcal_software_suite.sh:1` — explicit batch execution of software architecture and threshold ablations.

## Core logic

1. `prepare_ais` → `read_labels` (`data.py:33`) → `subject_split` (`data.py:22`) → aligned 10ms event bins → Tonic ToFrame/ToBinaRep → `frames.h5`, `manifest.json`.
2. `train` → `WindowDataset` (`data.py:140`) → `train_epoch` (`experiment.py:86`) → `PaperSTNet.forward` (`model.py:57`) + `PaperMatcher.training_loss` (`matching.py:65`) → joint backward/AdamW → validation selection and atomic checkpoint.
3. `PaperSTNet.export_forward` (`model.py:77`) copies ITD, normalization, forward GRU and its supervised coordinate head. `train` then performs two train-only fine-tuning epochs. Holdout is not read.
4. `evaluate` → `load_deployed` (`experiment.py:285`) checks completed forward fine-tune or completed bidirectional training → STNet-only per-window inference or `PaperTracker.track` (`tracker.py:101`).
5. `track` predicts pixel Kalman state (`tracker.py:13`), crops template/search (`matching.py:10`), applies local cosine (`matching.py:37`), converts local coordinates (`matching.py:24`), then `gate` (`tracker.py:44`) selects local, asynchronous, or synchronous processing.
6. Async work stores source frame, submission and arrival times. Completion is checked on the main thread; stale work predating the last sync anchor is discarded. `_replay` (`tracker.py:76`) starts from the preceding snapshot and updates the affected suffix. Already emitted `online` coordinates stay unchanged.
7. `metrics` (`experiment.py:59`) scores un-clipped 80×60 pixel coordinates. Evaluation writes per-frame CSV, JSON diagnostics, measured CSV and Markdown tables.

## Dependencies and storage

- PyTorch: Conv3d/GRU, autograd, AdamW, inference, CUDA streams when available.
- NumPy: timestamp indexing, subject split, pixel Kalman linear algebra and metrics.
- h5py: source compound events; prepared recording groups `frames`, `labels`, `valid`, `bin_start_us`.
- Tonic: four temporal bins to one four-bit two-channel BinaRep frame. Loaded only during preparation.
- Python ThreadPoolExecutor: one asynchronous STNet worker.
- No database, RPC, MQ or remote service is used by the training/inference runtime.

## Risks and explicit uncertainties

- Current adapter consumes AIS2024, whereas the supplied paper tables use other named datasets. Data paths and split manifests must travel with checkpoints.
- Paper patch sizes imply at most 3.536px local displacement, so tau2=5 cannot be reached by internal geometric error alone. Confidence still causes sync fallback.
- Unspecified layer details do not reproduce the paper's stated parameter counts; measured train/deploy counts are recorded. `full_conv` uses explicitly documented (3,Ks,Ks) kernels.
- During a sustained slow worker, queued corrections can become late. Report deadlines, queue-inclusive completion and arrival-to-emission latency; do not substitute instantaneous service time for end-to-end delay.
- Historical labels retain their original status. Both states are included for finite in-sensor nonzero coordinates and separate state metrics are reported; no inferred blink labels are used.
- Suffix replay gives a retrospective state trajectory. It is not an online prediction made at the original frame time.

## 3ET optimization additions

- `--inference-mode bidirectional` keeps the learned BiGRU/256-dimensional coordinate head. Validation selection uses the primary head; the same 16-frame historical window feeds online inference.
- `--initialize-from` validates model/data identity and starts an independently fingerprinted run with a fresh optimizer. `--resume` preserves configuration, optimizer, RNG, learning-rate progress and split identity.
- `ktcal_paper/diagnostics.py:18` evaluates train/val in batches and outputs per-record/state diagnostics and optional overlays. It does not read holdout or claim online timing.
- Completion markers are published after the summary; completed best weights cannot be resumed as intermediate weights.
