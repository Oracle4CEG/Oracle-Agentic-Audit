# Current reproduction entry point

The implementation and necessary experiments are complete. Use [the public reproduction guide](PUBLIC_REPRODUCTION.md) and the pinned public-release manifest for the current executable commands and status. The earlier completion contract below is historical; its future-tense wording does not override recorded execution receipts. Hosted Colab and independent review still need separate receipts.

# Reproduction contract

> 2026-10-09 用户缩减范围（优先于下文的早期完整方案）：只保留验证集 483 条、已完成的 B2 两轮 320 条、A1 六场景 × 40 案例 × 开关对照 480 条，共 1,283 条。取消低温度实验，不再补齐 B2 五轮或 B1 防护消融。原 B1/A1 五轮主实验继续复用。B2 仅与 A1 的对应 repeat 3、4 配对；额外已完成记录单列保留。固定范围和修订理由见 `manifests/experiment-scope-20261009.json`。恢复期间更改服务配置、缩减时中断的尝试均单独保留；不按答案质量筛选。


Implementation update (9 October 2026): the scientific modules, strict saved-response replay and supplemental Qwen inference are implemented. See `docs/CAMERA_READY_RUNBOOK.md`, `docs/EXPERIMENT_PROTOCOL.md` and the machine-readable receipts in `manifests/`. The checklist below is retained as the original completion contract; its imperative wording is not the current execution status. Full inference completion and hosted Colab require their actual receipts.

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
