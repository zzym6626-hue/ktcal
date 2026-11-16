# Author review: isolated hardware-first probe

Scope: temporal_probe.py, test_temporal_probe.py, evaluate_six_simulator.py,
evaluate_seven_simulator.py, probe_tail_rknn.py, verify_smoke.py; production
STNet/existing export/sequence_path/fast_path and original split call chain read.
Python AST and Python safety/style rules inspected; no delegated review.

Data flow: original train frames -> original floating intermediate features ->
quantization datasets; RKNN spatial/temporal outputs -> next real RKNN inputs ->
ORT or real RKNN readout -> production C++ gate/MA7 -> independent pixel scorer.
The seven-graph experiment never substitutes floating readout predictions.
No calibration labels or test feedback, no training, no checkpoint/gate change.

Checks: [offset,channel] weight permutation matches 3D cross-correlation; zero
boundaries do not leak between batches. Pool maps only validated released config.
Divisor once. Stage count/names/shapes/SHA/sample list validated. SDK statuses,
finite shapes, output coverage checked. SDK contexts released in finally. CPU
fallback/SDK warnings/failed attempts retained rather than hidden.

Limitations: probe intentionally materializes full features and channel stacks,
not ring-buffer ARM deployment code. Full stage1 output is 61.44MB float; concat
is larger. Tail 142MB RKNN and CPU fallback pose real-board memory/runtime risk.
Only one record evaluated; no full precision or timing claim. Host ABI fixtures
are not a vendor runtime validation. No board connection means final objective
'hardware runs' remains unverified, not marked achieved.

Failures and fixes: PYTHONPATH relative to switched SDK cwd caused first import
failure; unique v2 logs/dirs with absolute path fixed it. unittest -s tests without
-t . shadowed ktcal_paper; corrected discovery passed all 255. Local archive first
extraction ran before SCP completion and encountered truncation; completed v2
transfer SHA/gzip checked before successful extraction. Zsh read-only status
variable affected test shell wrapper; rerun uses code, C++ tests exit0.

No critical correctness issue found in the spike; do not promote into production
without actual runtime SDK/board validation and bounded-memory scheduling.
