# Recorded public notebook execution

These are actual outputs from [GitHub Actions run 38053135664](https://github.com/virusLuke3/Oracle-Agentic-Audit/actions/runs/38053135664)
on 10 October 2026, not an unexecuted example notebook. The workflow ran all
6 code cells in a fresh environment, downloaded the public inputs
without authentication, ran 32 tests, refitted CPU baselines and recomputed results.
Notebook elapsed time was 762.12 seconds, including setup and downloads.

- Notebook source: `9edb0de1e1a28f98b5bc38d4c59e55cf679fb2fe`.
- Scientific implementation: `33ec7020e97fc6b8dd0005415d3184ce889d3e28`.
- Input dataset: `ee197fc6fd6b8d980e7c7dda4feae966efe5d28c`.
- 73 scientific CSV comparisons passed with absolute/relative tolerance 1e-10.
- 18 numeric LaTeX tables and the exact prompt listing matched byte for byte.
- Eight figures were regenerated; 0/8 PNGs matched the reference bytes.
  Numerical comparisons passed; pixel-identical rendering is not claimed.
- No LLM inference requests were made. Saved response replay and new generation
  are different reproducibility claims.
- A real hosted Google Colab run remains **NOT_RUN**.

`acceptance.json` summarizes the execution. `research.executed.ipynb` preserves all
actual cell outputs. `comparison_receipt.json` provides per-output checks.
`download_receipt.json` records archive/file verification. `validation-output.txt`
contains the real test and required-panel output. `environment.freeze.txt` records
the scientific environment; notebook orchestration dependencies are pinned in
`requirements-notebook.txt`.

`artifact-verification.json` binds these downloaded records to GitHub artifact
11670288235 and its SHA-256 digest. `notebook-ci.json` preserves job/step status.
`delivery-attempts.json` separately records earlier local setup/download interruptions,
which did not reach scientific analysis. They are not relabeled successful runs.

The generated data, tables and figures are in `generated-results.tar.gz` on Hugging
Face under `research/agentic-audit/2026-10-10/acceptance/github-notebook/`.
That large archive is deliberately excluded from Git; its checksum is in the
artifact-verification receipt. Original model/data artifacts remain in the separate,
immutable input release. This verification does not clear the complete Atlas,
certify unknown public model revision/training metadata, or replace scientific review.

The complete archive is deposited at [immutable Hugging Face revision `586e850c`](https://huggingface.co/datasets/Oracle4CEG/OracleEconLab-Oracle-Incentives-v1/tree/586e850ce504db0f4ffec9bea7b0f4db3fa45239/research/agentic-audit/2026-10-10/acceptance/github-notebook).
See `manifests/public-acceptance.json` for the public-download verification.
