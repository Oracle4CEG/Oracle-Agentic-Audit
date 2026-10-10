# Execution and replay

> 2026-10-09 用户缩减范围（优先于下文的早期完整方案）：只保留验证集 483 条、已完成的 B2 两轮 320 条、A1 六场景 × 40 案例 × 开关对照 480 条，共 1,283 条。取消低温度实验，不再补齐 B2 五轮或 B1 防护消融。原 B1/A1 五轮主实验继续复用。B2 仅与 A1 的对应 repeat 3、4 配对；额外已完成记录单列保留。固定范围和修订理由见 `manifests/experiment-scope-20261009.json`。恢复期间更改服务配置、缩减时中断的尝试均单独保留；不按答案质量筛选。


Work in the research repository. The original workspace and descriptor acquisition sources are read-only inputs.

1. Install `requirements.lock` into Python 3.12.7. It locks analysis/client dependencies; remote GPU serving is separate.
2. Restore `data/local`, `outputs/qwen_supplement`, and `outputs/guard_experiment` from the governed local research bundle. Never include `.env` or model weights.
3. Run `python -m oracle_audit experiment-check`. All required fixed-scope cells must be present without active transport failures. Normal-return model/schema failures remain in the denominator. Cancelled optional work is explicitly marked not required, not complete. Prior infrastructure failures and scope interruptions remain in their separate archives.
4. Run `python -m analysis.replay --retrain`. This verifies inputs/artifacts/lineage, trains the baseline, fits validation calibrators, computes uncertainty and audits, generates figures and writes `outputs/final/RESULTS_zh.md`.
5. After isolated replay, run `python -m analysis.verify_replay --reference outputs/final --candidate outputs/clean_replay`. CSV rows, columns and missingness must match; floats use absolute/relative tolerance 1e-10. Nonnumeric values match exactly. Wall-time receipts are not scientific numerical references.

Before inference completes, `python -m analysis.replay --partial --no-bootstrap --out outputs/progress` excludes incomplete panels and writes a clearly partial report. Do not treat it as a completed experiment.

`python -m experiments.finish_required --watch-pid <main-inference-pid> --watch-pid <guard-inference-pid>` waits for complete panels, runs the strict replay and independent locked-environment replay, compares tables, updates manifests, then packages and verifies the local bundle. It records progress or failure in `outputs/completion/status.json`. It never retries or replaces inference outputs and does not certify hosted Colab or public release. Both `.venv` and `.venv-replay` must already be installed from the lock file.

Supplementary inference uses OpenAI-compatible endpoints. Sun-lab authentication is read from the original workspace `.env` by `oracle_audit.remote`; credentials are not written to logs/argv. Use SSH loopback forwarding. Offline replay needs no secrets.

A completed `run_meta.json` is written last, after inputs, events, outputs and validation. Configuration mismatch is an error. Preserve interrupted/failed attempts before any explicitly versioned diagnostic rerun; never silently replace them to improve results.

The notebook is an offline replay entry point for a research bundle. An actual fresh hosted Colab session still needs a real runtime/output receipt. A local notebook run cannot certify Colab. Public dataset revision, descriptor arXiv version, licensing and scientific review likewise need their own evidence.

The 9 October recovery uses `experiments.durable_run`, launched in a new process session with stdin closed and file logs. Its SSH control master uses background forwarding and keepalives. The supervisor is independent of a chat tool session, but still requires the local machine/network. Failed/interrupted attempts are preserved under `outputs/infrastructure_attempts`; recovery provenance is included in each new run record and the final analysis. Do not launch a second supervisor or silently retry a stopped job.
