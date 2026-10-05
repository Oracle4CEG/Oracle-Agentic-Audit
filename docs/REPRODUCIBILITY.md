# Reproduction contract

## Three different checks

**Template smoke:** `python -m oracle_audit smoke` checks a SHA-256-pinned four-case synthetic prediction file and regenerates example Brier/coverage/selective-error/action-loss outputs. It uses no network or model. It is not a research result.

**Saved-prediction replay:** Huaiyu must implement one offline command using the published real case IDs, evidence audit records and per-run predictions. Pin the input revision, check every checksum, recompute every manuscript metric and regenerate each table/figure. Compare outputs against governed references with explicit absolute/relative tolerances; reject missing cases or mismatched checksums.

**Full inference:** Huaiyu must implement a separate command that builds admissible evidence, runs the exact checkpoint/settings and emits raw outputs plus predictions. Document CPU/GPU/RAM, runtime, model access, cost and any nondeterminism. Do not imply that a deterministic replay is a fresh inference replication.

## Required chain for each result

Original source -> acquisition code/receipt -> queried partition -> transformation code -> experiment input -> model/run records -> analysis -> output.

The descriptor owns acquisition and global transformation. The research index points to its fixed public release and records the experiment-specific selection and processing. A pointer is not evidence that a missing source partition has been released.

`manifests/result-index.json` holds planned research results. Status is PLANNED or BLOCKED until independently verified. Never assign DERIVED merely because an output file exists. Use SYNTHETIC for fixtures. Define denominators and estimands before interpretation.

## Tests Huaiyu must add

Missing inputs and bad hashes; duplicate or overlapping case splits; future/outcome leakage; malformed citation IDs and policy actions; all-deferral and zero-coverage cases; constant-score metrics; paired bootstrap grouping across runs; reference mismatch; guard clean-case behavior; full runner configuration errors.

The template tests cover only the scaffold's hash checks, evidence cutoff helper, fixture metrics, and rejection of an incomplete release manifest. They do not validate the experiment.

## Licenses and access

Record source/data/model licenses independently of the code license. Never commit credentials, confidential manuscripts, model weights without permission or private artifacts. Confirm the code license with the corresponding author; no release license is assumed here.
