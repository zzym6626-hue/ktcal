# Authorized equivalent-split spike (2026-11-16)

Purpose: investigate whether small, per-frame spatial Conv2D graphs avoid the
observed RV1106 whole-Conv3D compiler expansion/crash. Experimental only; do not
replace the production backend or claim board deployment readiness.

Chosen probe: three original spatial Conv3D layers with temporal kernel1 become
per-frame Conv2D+copied BN+ReLU. Keep original temporal convolution/pooling and
100-frame bidirectional GRU/attention/FC on the CPU. Preserve original weights,
raw divisor, ten views, median, matcher/KF gate and independent MA7. Alternatives
not selected: blindly retry whole-graph SDK compilation; retrain/replace layers.

- [x] RED/GREEN: stage and composed full100 logits, nontrivial BN, edge padding,
      divisor-once, original state unchanged, incompatible input rejection.
- [x] Export three fixed batch1 spatial ONNX; verify ONNX against PyTorch.
- [x] Actual RV1106 INT8 build, using train frames/intermediate features only.
      Preserve logs, calibration/source hashes and return codes. Probe simulation
      only when SDK supports it; simulation is not real NPU verification.
- [x] Compare composed floating split on every test_aug record and all ten views;
      original PyCPU->production C++ gate/precision parity, independent scoring.
- [x] If INT8 outputs can be obtained, measure their precision independently;
      never attribute floating precision to INT8. If blocked, report the blocker.
- [x] Fresh existing Python/C++ suite, author review, archive evidence with hashes.

No training, test_200_200 access, weight replacement, commit, push, dependency
changes to the original environment, or modification of the published backend.
Working dirs are unique; reuse only installed isolated tooling and source copies.

Result: floating9300 passes all original CPU tiers and individual precision
classes. Three INT8 spatial graphs successfully build/export/simulate. Full INT8
spatial + original floating temporal simulation KTCAL P3=.9078494623655914,
P5=.963763440860215, P10=.9926881720430107, mean1.637600245655px. P3 drops
0.3226pp vs float; not promoted. Both whole validation processes exited0,
independent CSV score/classification/hash verification exited0. No board result.
