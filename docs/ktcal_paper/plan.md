# KTCal Software Implementation Plan

> **For agentic workers:** Use executing-plans; execute tasks locally and obtain a final independent review. The user requested implementation directly.

**Goal:** Implement the provided paper's software training and inference protocol, with honest reproducible experiment reports.
**Architecture:** Add ktcal_paper package to the existing project; preserve legacy models and results. Reuse torch, h5py and tonic; implement pixel Kalman and source-indexed threaded corrections within the new package.
**Tech Stack:** Python3.9, PyTorch2.1, NumPy1.26, Tonic1.6.
**Spec:** design.md

## Global Constraints

- Existing weights must remain untouched; new outputs must not overwrite previous runs.
- AIS2024 results must not be labeled as SEET/3ET+2025/Ini-30 results.
- Holdout is never used for model selection; placeholders never provide supervision.
- CPU latency is not RTX3090 latency; report real timing and parameter counts.

## Review Focus

- Export must copy the forward GRU and trained forward head, not initialize a new recurrent network.
- Even rectangular patches must have consistent geometric centers in both training and inference.
- Temporal windows must remain within one recording and subject splits must be disjoint.
- Async corrections must refer to their source frame and not overwrite newer anchors.
- Checkpoint resume must restore optimizer, phase, RNG and split metadata.

## Tasks

- [x] 1. Test and implement model.py: direct-coordinate BiGRU training, forward supervision/export, ablation switches.
- [x] 2. Test and implement matching.py/tracker.py: rectangular depthwise cosine matching, pixel Kalman covariances, actual async/sync correction.
- [x] 3. Test and implement data.py: labeled AIS preprocessing, fixed subject split, lazy historical windows.
- [x] 4. Implement __main__.py/experiment.py: train, resume, two-epoch forward fine-tune, evaluate, structured metrics/table reports.
- [ ] 5. Run full tests and real-data pilot, independently review changes, install new package/docs/tests into the project and verify entry points.

The current target is the user-provided Git repository `IdeaProjects/tdtracker` (main, initial commit c8b8ae5). Stage and verify changes, migrate additively, then commit and push the authorized source changes. Data, virtual environment and model artifacts are excluded from Git.

## Updated 3ET optimization sequence

- [x] Finish 8+2 pilot; inspect train/val, per-record/state errors and overlays.
- [x] Add tested bidirectional inference, traceable initialization and cosine schedule.
- [ ] Complete convergence experiment on the unchanged train/val split.
- [ ] If necessary, compare an explicit architecture change on this validation split.
- [ ] Migrate tested source, local data and measured reports to IdeaProjects/tdtracker.
- [ ] Review, run checks in the target repository and push source/docs only.
- [ ] Evaluate holdout only after model/configuration selection is frozen.
