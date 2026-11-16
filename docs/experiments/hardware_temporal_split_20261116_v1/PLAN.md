# Authorized hardware-first feasibility spike, 2026-11-16

User waived the 91% accuracy floor; prioritize RV1106-runnable components.
Continue the already approved isolated probe, not a production backend release.

Chosen: preserve weights/full100/ten views/gating; map each dense Conv3D(3,1,1)
to fixed Conv2D(1,1) on [previous,current,next] channels with zero outer frames.
Use the original BN/ReLU/AvgPool. Retain spatial RKNN calibration and add
train-only temporal intermediate calibration. No new accuracy tuning/training.
Alternatives rejected: retry crashed whole graph; alter windows/weights/gates.

- [x] RED/GREEN on channel order, first/last zero padding, BN/pool, full100 logits,
      divisor once, unchanged source state and incompatible layout rejection.
- [x] Export/check three small temporal ONNX graphs and original CPU tail.
- [x] Train-frames-only intermediate calibration; actual rv1106 INT8 compilation
      and SDK host simulation, retain codes/logs/hashes.
- [x] Compose all six quantized graphs and CPU tail through production C++ gate;
      verify output completeness, low-tier no-fusion and score, without P3 floor.
- [x] Fresh tests/review/hash archive; distinguish SDK simulation from real board.

No test_200_200.h5, production/weight replacement, commit or push. The currently
connected host is x86 AutoDL, no board runtime/toolchain/ADB detected. Actual
board execution needs the user's RV1106 device access and matching runtime SDK.

Outcome: six-graph ORT-tail and seven-RKNN-tail 100-frame smoke chains passed.
Actual board execution remains blocked on RV1106 access, matching runtime, RAM
and cross-toolchain. Do not mark hardware deployment complete. No jobs left running.
