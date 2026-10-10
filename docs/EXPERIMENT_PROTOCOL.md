# Required camera-ready experiments

> 2026-10-09 用户缩减范围（优先于下文的早期完整方案）：只保留验证集 483 条、已完成的 B2 两轮 320 条、A1 六场景 × 40 案例 × 开关对照 480 条，共 1,283 条。取消低温度实验，不再补齐 B2 五轮或 B1 防护消融。原 B1/A1 五轮主实验继续复用。B2 仅与 A1 的对应 repeat 3、4 配对；额外已完成记录单列保留。固定范围和修订理由见 `manifests/experiment-scope-20261009.json`。恢复期间更改服务配置、缩减时中断的尝试均单独保留；不按答案质量筛选。


This is a retrospective supplementary analysis, not a prospective preregistration. Settings are fixed in `experiments/config.json`; no test-label tuning or answer-quality selection is performed. Infrastructure and scope amendments preserve all superseded attempts. Code hashes are stored with execution configurations.

## Recorded infrastructure amendment, 9 October 2026

After the first recovery still showed sustained decode slowdown, the two task-owned services were restarted with CUDA graphs and CPU-library thread limits of one. Tensor parallelism, weights, BF16 precision, sampling and evidence remain unchanged. The entire guard panel is restarted under this configuration, including prior successful guard responses, so all guard on/off and clean/corrupt conditions share the same runtime. Main-supplement successful responses are retained. Every superseded record, including the first recovery's interrupted requests, is archived. See `manifests/recovery-serving-runtime.json`. Latency across the original and recovered serving configurations is descriptive, not a controlled efficiency comparison.

The original local orchestration and SSH tunnels exited after a period of increasing timeouts. The recovery receipt preserves 289 completed transport failures and 20 interrupted attempt folders, together with file hashes and the exact original runner sources. Only infrastructure status selects retries; normal-return model/schema errors are retained unchanged. The recovered active panel is distinct from first-attempt-only evaluation. Archived observed tokens and latencies are reported separately; compute with no returned usage remains unknown. Generation parameters, cases, seeds, evidence and model weights are unchanged. A detached supervisor now runs the two jobs without the auxiliary matched batches, and a transport failure stops additional scheduling. See `manifests/infrastructure-recovery-20261009.json` and `analysis/infrastructure.py`; this amendment supersedes first-attempt-only descriptions for supplemental inference.

| Experiment | Design and denominator | Implementation |
| --- | --- | --- |
| Cohort/labels | 489 train, 161 validation, 160 test; 810 protocol labels and exact exclusion predicates | `analysis/cohort_lineage.py`, `analysis/economics.py` |
| Strong feature baseline | Fixed LightGBM, same 16 features as B0; train-prior predictor | `models/tabular.py` |
| Policy correction | Original intervals, cost-derived 0.1/0.9 thresholds, validation-selected thresholds, constant policies, costs 0.05–0.40 | `policies/decision.py`, `analysis/evaluate.py` |
| Comparable calibration | B1/A1/B2 each 161 validation cases, one inference per case; validation-only Platt | `models/run_required.py`, `policies/calibration.py` |
| Matched evidence | Reuse B2's 160 cases × 2 complete repeats (IDs 3,4); pair with those exact original A1 trajectories | `analysis/matching.py`, `analysis/evaluate.py` |
| Lower temperature | Cancelled optional sensitivity experiment; no additional inference | `manifests/experiment-scope-20261009.json` |
| A1 guard ablation | 40 cases × 6 scenarios × A1 × on/all_off = 480 runs; no new B1 ablation claim | `guards/run_experiment.py` |
| Enforcement replay | Saved predictions with only final override removed | `analysis/audits.py` |
| Contamination/availability | Checkpoint hashes, actor groups, fields, raw market reconstruction, lexical-guard review | `analysis/evidence_lineage.py`, `analysis/boundary.py` |
| Citations/costs | ID categories, completeness, eight-call cap, raw tokens, wall time | `analysis/citations.py`, `analysis/audits.py` |
| Economics | Role-specific activity, asset-specific recipients, capital-days, realized payoffs, five-protocol availability | `analysis/economics.py` |

Main comparisons use 160 unique test cases. Original B1/A1 paired bootstrap retains all five outputs within each resampled case. B2/A1 evidence-matched comparisons retain only corresponding repeat IDs 3 and 4 on both sides; deterministic baselines have one prediction per case. Two B2 repeats were selected because they were already complete when the user reduced scope, not for their performance. This post-start selection has less replication than the original five-repeat design and is not a preregistered sample-size choice. Intervals use 2,000 resamples, seed 20261009, without multiplicity adjustment. ECE has ten fixed-width bins. Undefined selective error is missing.

Intervals condition on the fixed training fit and validation calibration; they do not include retraining or recalibration uncertainty. Actor subgroup intervals retain the original time split. Hypothetical review-capacity scenarios count both Investigate and forced Abstain; budgets below mandatory review are declared infeasible. Six outcome-field perturbations and 100 shuffled train/validation-label fits are separate dataflow diagnostics, not competitive baselines or proof of no leakage.

Investigate/Abstain have terminal stylized costs 0.1/0.2, without measured human-review wages, error, delay or capacity. Always-Investigate has loss 0.1 and no automation when unconstrained. The model-conditioned enforcement replay has loss 0.1+0.1q, where q includes model contract failures; it is not an independently functioning no-model input-quality baseline. Canonical required evidence passes structural checks in all 160 test cases.

Calibration uses the fixed active validation panel, including normal-return invalid model outputs. Following the infrastructure amendment, timeout/interruption recoveries are explicit repeated attempts, so the supplemental panel is not first-attempt-only. A deadline exceeded can also reflect a long model trajectory; individual timeout causes are unverified. Missing numeric probability gets the original parser's 0.5 fallback, explicitly flagged. This is system fallback, not a model expression of 0.5. Calibration can transform that probability, while invalid-output constraints remain. Secondary valid-output-only metrics cannot replace the fixed denominator.

B2 receives exact tool-result content from the last actual A1 API request. Unconsumed results, A1 assistant reasoning, probabilities/actions and assistant tool arguments are excluded. Names, ordering, errors and validation responses remain. Retrieval choices are endogenous: the control holds delivered evidence fixed, not randomized selection. Completed retrievals are checked against actual received content.

Guard all_off disables source timestamp admission, evidence validation and final override. JSON/schema parsing, read-only scope and the eight-call cap remain. Paired conditions use identical deterministic fixtures. Future records use a fixed value at cutoff+1 without consulting labels. Plausible-wrong changes proposer history and recomputes internal digests/IDs, modeling incorrect upstream data with consistent metadata. Internal hashes cannot establish factual correctness under this threat model.

Actual intervention exposure is audited from B1 input packets and A1's last actual API request. Assignment to corruption does not guarantee A1 retrieves the changed record. Future-ID citation counts include index-only citations even if the associated content was never delivered. Raw logged exposure flags are retained separately. Injection admission recognizes the fixed synthetic string by construction; this tests a known-fixture protection, not general detection of unseen prompt injection.

On/off runs are separate generations. An equal configured seed does not establish bitwise deterministic output in batched serving. Initial-input and output equality are audited per pair; for A1, identical initial prompts can still lead to different tool-result contexts. Full-ablation contrasts contain generation variability, whereas saved-output final-enforcement replay isolates the deterministic action override.

The lexical-guard review covers all 76 flagged original main-Qwen case-runs. It is single AI-assisted contextual annotation, including explicit memory-cue search, not independent blinded expert review. Source hashes and annotations are in `analysis/reasoning_guard_annotations.json`. Historical predictions and guards are not changed to improve results. Positive-only review cannot estimate sensitivity or exclude unflagged memory use.

Qwen's training cutoff is unverified. Local hashes establish observed checkpoint bytes, not public revision or absence of contamination. Market responses were archived retrospectively. Point timestamps and features are checked; prospective publication, finality latency and independent exchange-fill reconstruction remain unestablished.

The main service uses 4×RTX 5090, TP4, SGLang 0.5.13.post1, no quantization, context 32768, Triton attention, PyTorch sampling, Qwen parsers and cache reporting. Guards use a separate 2×RTX 5090 TP2 service, memory fraction 0.91. The initial services disabled CUDA graphs; the current services enable them and limit CPU-library threads to one. Current orchestration uses eight main workers and four guard workers, with no auxiliary B2 batches. Timings include queues; cross-batch latency is descriptive. Local dollar cost is unmeasured.

The Qwen files in `models/reference` preserve archived algorithms with import-path adaptation. `cohort_source.py` is an exact copy of the original cohort builder. New orchestration and analysis live outside those sources. Source hashes, inputs and raw run records are authoritative.
