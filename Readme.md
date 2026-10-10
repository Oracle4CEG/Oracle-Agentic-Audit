# Oracle-Agentic-Audit

> 2026-10-09 用户缩减范围（优先于下文的早期完整方案）：只保留验证集 483 条、已完成的 B2 两轮 320 条、A1 六场景 × 40 案例 × 开关对照 480 条，共 1,283 条。取消低温度实验，不再补齐 B2 五轮或 B1 防护消融。原 B1/A1 五轮主实验继续复用。B2 仅与 A1 的对应 repeat 3、4 配对；额外已完成记录单列保留。固定范围和修订理由见 `manifests/experiment-scope-20261009.json`。恢复期间更改服务配置、缩减时中断的尝试均单独保留；不按答案质量筛选。


Evaluation of agentic auditing on 810 disputed UMA proposals: corrected policies, validation-only calibration, stronger feature baselines, matched-evidence single-pass inference, guard ablations and selected economic analyses.

The scientific implementation is present. Completion is checked from per-case run records, not inferred from this README. Public release and hosted Colab have separate requirements.

## Install and replay

Python 3.12.7, Linux:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m experiments.download
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m oracle_audit experiment-check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m analysis.replay --retrain --out outputs/replay
.venv/bin/python -m analysis.verify_replay --reference outputs/final --candidate outputs/replay
```

Replay uses public, immutable, checksum-pinned inputs and saved responses. It makes no inference requests. [The public-release manifest](manifests/public-release.json) binds separate Atlas selected-input and research-record archives to one Hugging Face commit. The downloader needs no authentication and refuses to overwrite differing existing records. Inputs and outputs are outside Git. See [the public reproduction guide](docs/PUBLIC_REPRODUCTION.md).

The complete replay recomputes 73 scientific CSV tables, renders 18 numeric LaTeX tables plus the exact prompts, and generates eight figures. [The output index](manifests/manuscript-outputs.json) maps each camera-ready table and figure to its source results and generator. Baselines are refitted on CPU; published pickles are never loaded. This does not rerun the LLM or certify the old manuscript's superseded numerical claims.

Strict replay rejects incomplete required panels. During inference, `python -m analysis.replay --partial --no-bootstrap --out outputs/progress` produces explicitly partial diagnostics. `outputs/final/RESULTS_zh.md` is the complete-results entry point only after strict replay passes. Tables have CSV/Parquet versions; original figures have PDF/PNG versions.

## Additional Qwen inference

The original 1,600 main Qwen runs and 320 original corruption runs are frozen, hash-verified inputs. These commands execute the additional experiments:

```bash
OPENAI_BASE_URL=http://127.0.0.1:33001/v1 .venv/bin/python -m models.run_required --suite all --workers 8
OPENAI_BASE_URL=http://127.0.0.1:33002/v1 .venv/bin/python -m guards.run_experiment --workers 4
```

Serve the pinned Qwen3.8-27B checkpoint. The main supplement uses TP4; the full guard experiment uses a separate TP2 service. Settings and hashes are in `manifests/*server-runtime.json` and [the protocol](docs/EXPERIMENT_PROTOCOL.md). The 9 October recovery preserves timeout/interruption attempts separately and labels the active panel as recovered, rather than first-attempt-only. Normal-return invalid model outputs remain in the analysis. New transport failures stop additional scheduling; changed attempts/configurations require archived provenance.

The 64 public demonstrations all overlap training. They are not an extra held-out cohort. Historical GPT-5.6-sol runs are analyzed separately and never called Qwen results.

## Scope and release

The research repository owns experiment selection, evidence, model/guard experiments, policies, evaluation and selected economics. Atlas construction belongs to the descriptor. [Project boundary](docs/PROJECT_BOUNDARY.md).

The public dataset revision is pinned in the release manifest. The descriptor arXiv version, author-confirmed research-code licensing, independent scientific review and actual hosted-Colab test remain separate requirements. `python -m oracle_audit release-check` reports blockers. Local replay and GitHub Actions are not hosted Colab.

The original `python -m oracle_audit smoke` example is synthetic and excluded from research findings. No new research-code license or descriptor arXiv identifier is asserted.
