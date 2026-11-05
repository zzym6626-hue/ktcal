# Independent review

2026-11-05, read-only reviewer `paper_code_review`.

The reviewer checked the new package against the supplied paper and design decisions. Model/export, NCC geometry, subject isolation, original-label supervision and checkpoint recovery had no blocking findings.

Findings fixed and independently rechecked:

1. Paced-input latency now includes arrival-to-emission backlog for both KTCal and STNet-only, separately from service time.
2. Async corrections replay only the suffix affected by the earliest new observation, starting from a stored Kalman snapshot.
3. Async completion and online emission deadlines are separate. Application timestamps are taken after replay; discarded observations have no application timestamp.
4. Online synchronous resets and retrospective reconstructed reset counts have distinct fields.
5. Evaluation rejects forward checkpoints until the two fine-tune epochs are complete. Selection epoch and completed epoch count are distinct metadata.

The reviewer ran the two source-frame tracker tests and an independent delayed-inference timing probe. Final verdict: all raised issues closed. This is an implementation review, not certification of paper accuracy or GPU timing.

## 3ET continuation review

Independent review identified and closed: stale completion flags when resuming finalized best weights; terminal resume state published before summary; missing inference mode in Markdown tables. Tests reproduce both resume/finalization failures, including simulated summary-write failure and recovery.

The diagnostic checkpoint is loaded and hashed from a single byte snapshot, so concurrent training replacement cannot misattribute predictions. A regression test replaces the source immediately after load and verifies the original hash.

Final-hidden readout receives gradients through reverse recurrent weights and preserves the exported forward prediction. Selection, initialization provenance, cosine resume and architecture configuration have integration coverage.
