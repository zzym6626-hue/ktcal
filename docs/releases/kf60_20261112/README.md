# KF60 release evidence — 2026-11-12

The user accepted the 60.73% raw KF candidate as the release configuration. It trades four P3-qualified frames versus the earlier accuracy-first baseline (91.1505% → 91.1075%); it is not an all-metrics-nondecreasing result.

- `reference.json`: complete three-repeat RTX 3090, FP32, batch1 100-frame run. P3/P5/P10 = 91.1075%/96.4516%/99.2473%, mean error 1.614474 px; median 1.052424 ms/frame (105.24 ms/100 frames).
- `reference_verified.json`: independent checkpoint reload, torch expectation decoder and record-filter/pixel scoring; identical threshold scores and tier counts. Raw low-tier STNet perturbation changes no coordinate (maximum 0 px). Its report hash matches `reference.json`.
- `weight_packaging.json`: the source checkpoint was repackaged into the existing inference-only release format, without optimizer/training state. All 57 model parameters/buffers and 100-frame ten-view CPU predictions match exactly. The new file hash differs because the container/metadata changed, not the learned parameters. Released hashes are in `weights/3et/manifest.json`.

A fresh clean-snapshot release-code GPU run with the source container also reproduced the same 9,300-frame metrics and 5648/27/3625 routes; its three warm timings were 1.046/1.041/1.033 ms/frame, with independent verification passing. A subsequent packaged-container CUDA recheck disconnected; SSH then refused connections. That interrupted second run is not reported as completed. Clean-snapshot full tests passed locally (176 total, one CUDA skip) and remotely (176 total, no skips) before the disconnection. The reference timing excludes checkpoint loading, so inference-only repackaging is outside the measured interval.

The published command is `python -m scripts.evaluate_3et`; independently verify its own newly generated report with `python -m scripts.verify_3et --expected-report local_run/3et_val.json --output local_run/3et_val_verified.json`.

All selection/scoring uses `test_aug.h5` (93 independent 100-frame records). Ground-truth labels are used only for scoring, not gates. No final holdout was read. Small tiers match/update/output KF without current-frame STNet fusion; STNet still precomputes all frames, and final MA7 may use neighboring reset information. The warm sequence-processing average is not causal response latency or RV1106 board performance.
