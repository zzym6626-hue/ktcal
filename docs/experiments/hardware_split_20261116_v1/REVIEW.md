# Author review (experimental split only)

## Scope and evidence
`split_probe.py`, `test_split_probe.py`, `evaluate_split_full.py`,
`evaluate_int8_simulator.py`, `verify_results.py`, copied `full_bridge.cc`.
No production source or weight modified in this turn. All 13 local/remote
dependency/source/weight hashes and six executed script hashes match. Review
used Python rules and AST/manual calls; auxiliary graph found zero functions,
which is an incomplete analyzer result, NOT a clean-code certificate.

## Paths checked
- Spatial2D: squeeze only temporal1 kernel, preserve padding/stride/dilation,
  groups/bias, copy every BN state and eps; eval only. Nontrivial BN test catches
  identity-default false positives. Original model state remains unchanged.
- SplitSTNet: divisor once; each of 100 frames traverses the original three
  spatial stages; dense temporal padding/BN/ReLU/AvgPool retained, then original
  full100 pool/LayerNorm/BiGRU/local attention/FC. No truncated-window inference.
  Shape contract and incompatible temporal variants fail closed.
- Calibration: only explicit train_aug frames, original floating intermediate
  features, eight records, both mirror states, five selected timesteps. Per-stage
  file/list hashes checked before build. No training labels or test calibration.
  Small calibration is a conversion experiment, not release qualification.
- Conversion: independent output directories; every SDK status checked; always
  release object. Successful conversion manifest is separate from simulator and
  board validation. Simulator uses retained built object, explicit NCHW and
  exact output shape/finite checks. Build success never means board success.
- Floating evaluation: original PyCPU and 3 spatial ONNX/temporal PyCPU logits;
  extra unfused split check; production C++ transform replay byte checks, medium
  policy/gate trace and low-global perturbation checks. Input/weight hashes fixed.
  Compare every record and aggregate precision, without tuning tolerances here.
- INT8 evaluation: original ten views, quantized spatial outputs feed subsequent
  original CPU temporal stages (no ground-truth feedback); all transforms/median,
  decode/KF/matcher/MA7 use original production C++. Labels read for scoring only.
- Independent verifier: exact ordered9300 CSV coverage, finite pixels, P3/P5/P10
  with strict `<`, float individual classification parity, separate INT8 deltas,
  tier count recomputation, RKNN hashes and exact93000 calls/stage.

## Material limitations / next work
- This is a spike, not production board code. CPU temporal path uses PyTorch on
  x86; no ARM cross-build/full deployment integration is claimed.
- Leaving dense temporal convolution on ARM CPU is a major potential cost:
  23.494656 billion temporal MACs per100-frame ten-view record, computed from
  weights/shapes. A dedicated NPU temporal1x1 graph may be needed. This is an
  arithmetic count, not a measured runtime.
- Stage1 output alone is61.44MB per100-frame view in float32. Current spike
  retains full intermediates, requiring future tiling/ring-buffer design on
  memory-limited board; no measured board peak-memory figure.
- Simulator outputs cannot certify actual NPU rounding, layout or transfer
  costs. Train-only calibration80 stage samples is deliberately small; a failed
  precision target requires calibration/QAT investigation, not gate changes.
- All observed timings are host checks with reference/replay/IO and concurrent
  CPU jobs. They are NOT NPU or release-latency benchmarks.
- SDK can leave check/cache/partial files on a failed build; isolated directories
  preserve them. Do not reuse a success manifest for a partial/changed artifact.
- No dynamic code execution, mutable default arguments, bare exceptions or
  shell user-input interpolation in experiment Python. torch.load consumes only
  the trusted hash-verified released checkpoint.

## Validation so far
- RED4 missing-feature failures, then GREEN4 tests locally and remote.
- Existing suite255:248 passed/7 skipped (CUDA/dependency skips); no failures.
  Existing PyTorch thread-count warnings retained in log.
- C++ make test passed. ASAN/UBSAN backend tests passed; SDK fixture only.
- Three real RKNN-Toolkit2 2.3.2 RV1106 builds and simulator probes exited0.
  Full floating/INT8 validation processes exited0, each93x100; independent
  verifier passed coverage, per-frame floating threshold parity and tier counts.
  FP KTCAL P3=.9110752688; INT8 simulation P3=.9078494624 (drop0.3226pp),
  candidate NOT adopted. All low fusion0. Max floating final drift4.3869e-5px,
  logits6.6757e-5. Transferred source/RKNN/CSV/report hashes verified.
