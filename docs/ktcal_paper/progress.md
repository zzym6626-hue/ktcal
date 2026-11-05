# Execution ledger

2026-11-05: Implemented the paper-aligned training, forward export, pixel NCC/Kalman and asynchronous tracking path. Original sources and weights are preserved. Model and data assumptions are in design.md.

15 baseline tests passed; 16 tests pass after adding bidirectional inference, a separately recorded initialization experiment and an exactly resumable cosine schedule. Earlier independent review closed the tracking and experiment pipeline findings.

Prepared the complete labeled AIS2024 training archive: 40 recordings, 33,557 frames. Split by subject before generating windows. No holdout evaluation has run. Test placeholder labels are excluded.

Pilot finished 8 bidirectional epochs plus 2 forward fine-tune epochs. The selected forward model is fine-tune epoch 1: train mean distance 5.344px and val 7.142px (stride16). The main-training checkpoint gives train 7.190px and val 7.451px with the bidirectional head. A train-mean constant predictor gives val 12.663px. Validation overlays and per-record/original-state reports were generated from actual predictions.

The user changed the destination to IdeaProjects/tdtracker and authorized bidirectional inference. A 16-epoch continuation experiment is running with new optimizer, lr 0.0002→0.00002 cosine, identical architecture and fixed data split. No dataset change, no holdout tuning, no claim of reproducing the paper tables.
