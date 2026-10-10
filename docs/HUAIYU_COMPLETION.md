> Current delivery note (10 October 2026): the implementation and fixed public inputs are available. See PUBLIC_REPRODUCTION.md and actual execution receipts for current status. The original instruction list below is retained as history. The user reduced the scope on 9 October: B2 uses repeats 3 and 4, A1 alone has a 480-run guard panel, and optional low-temperature work is cancelled. The notebook now runs the actual public replay; hosted Google Colab acceptance remains separate. The user confirmed MIT with the corresponding author on 10 October.

# Huaiyu: research repository completion instructions

This is the code project for the research paper. The Data Descriptor has an earlier, separate work plan. Your task here is to make the research experiment executable and align it with the revised research manuscript.

## Required handoff

1. **Inputs and cohort.** Fill `manifests/research-inputs.json`: exact public dataset revision, descriptor version when public, file checksums, case IDs, labels, features and evidence packets. Document the original 489/161/160 chronological split of the 810 linked disputed cases, all dates and exclusions, and the distinction from the 64 public examples. Verify these counts against the original records. New cohorts must be identified separately.
2. **Evidence available at the decision.** Publish request/proposal semantics, source locators, timestamps, retrieval snapshots and allow/deny rules. Remove outcome and future information from model input. Quantify the effects of late or incomplete evidence. Record evidence selection for every run.
3. **Model implementation.** Supply exact model IDs, revisions, training-cutoff evidence where available, serving software, quantization, hardware, prompts, sampling settings and seeds. Resolve inconsistent checkpoint descriptions in the draft. Do not use an inferred or invented model name.
4. **Policy correction.** Implement and disclose the complete loss matrix for uphold, reject, investigate and abstain. Investigate at cost 0.1 dominates abstain at 0.2 under the draft's fixed costs; include always-investigate. Clarify whether investigate ends the decision or adds later review loss. Derive thresholds from the actual costs and calibrated probabilities, and calibrate all systems on validation data only. Run a defensible cost sensitivity analysis.
5. **Comparisons.** Retain B0, B1 and A1 with matched cohorts and information boundaries. Add a stronger non-LLM baseline, a single-pass system receiving exactly the A1 evidence packet, comparable calibration, and a lower-temperature robustness run. Publish raw outputs and per-case predictions rather than only aggregated tables.
6. **Guards and contamination.** Implement guard-on/off comparisons, clean-case false abstention, future evidence, incorrect but in-cutoff evidence, citation-error categories and the eight-call cap. Separate model behavior from enforced safeguards. Audit possible training memorization and seen/unseen cases; a prompt instruction alone is not a contamination control.
7. **Analysis.** Compute AUROC, Brier, ECE with stated bins, loss, coverage, selective error with undefined denominators handled explicitly, evidence completeness and citation validity. Use a paired case-level bootstrap retaining all five runs per case. The original comparison has 160 unique test cases, not 800 independent cases: expect 800 B1 and 800 A1 case-runs plus deterministic B0 outputs. Track additional experiments separately.
8. **Costs and economic findings.** Reconcile main-text and appendix token counts and pricing assumptions from run records. Separate measured inference time, estimated API dollars and local hardware cost. Report call counts, failures and cap frequency. Add only the economic findings selected for the research manuscript, with the descriptor's definitions and appropriate denominators.
9. **Reproduction.** Provide an offline command that regenerates every reported table and figure from saved predictions; a separate full-inference command; locked dependencies; synthetic fixtures; exact output comparison tolerances; hardware/runtime measurements; and a clean-environment replay receipt. Update the result index so each claim maps to source, transformation, prediction file, script and output.
10. **Colab.** Replace the template notice with the working offline replay, pin this repository's commit, test every cell on an actual fresh hosted Colab runtime, and record runtime type, date, outputs and failures. The current notebook only demonstrates a synthetic smoke check. A local notebook execution is not a hosted Colab test.

## Research code and manuscript must agree

Resolve all differences between claimed scope and released assets. Do not say fully replicable, fully available, leakage-free or intrinsically safe unless the corresponding evidence exists. Guarded abstention is an enforced control; evidence retrieval improvement is not necessarily a predictive accuracy gain. Link every final claim to a checked result and retain negative or inconclusive findings.

Remove full Atlas construction, global variable inventories and full lifecycle reconstruction from the research manuscript; cite the descriptor. Retain the experiment cohort, labels, splits, admissible information, model methods, action policy, metrics and limitations in the research paper. Keep one scientific body and result set for both venue wrappers.

## Internal deadlines, Beijing time

| Deadline | Research deliverable |
| --- | --- |
| 6 October 2026 | Input inventory, missing-asset list and ownership map |
| 9 October | Corrected policy and analysis design; manuscript boundary agreed |
| 12 October | First compact research draft and executable replay skeleton |
| 15 October | Checkpoints, evidence controls, baselines and cost audit implemented |
| 19 October | Cite the descriptor's actual public arXiv version; this is an upstream dependency |
| 22 October | Required experiments, saved outputs and complete result index |
| 26 October | Independent clean replay and revised response/evidence register |
| 30 October | Freeze common scientific text, tables, figures and research code |

If a required input is unavailable, report the named gap and its consequence promptly; do not fill it with a synthetic value or silently omit the analysis. The corresponding author handles any change to the publication plan.

## Completion criteria

- Every paper result has a runnable command and an immutable input/reference record.
- The offline replay reproduces all reported outputs with documented tolerances.
- Full inference runs on the documented environment and its compute is recorded.
- Data semantics, labels, policy and leakage checks have scientific review.
- Paper numbers, code outputs, figures, tables and availability statements agree.
- Hosted Colab and CI are explicitly recorded as passed, failed, blocked or not run.
- License and citation metadata are confirmed by the author before a release claim.
