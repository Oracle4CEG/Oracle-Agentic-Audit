# Folder guide for students

Put one kind of research artifact in each folder. The data descriptor continues under its existing instructions.

| Folder | What to put here |
| --- | --- |
| `docs/` | Research scope, methods/reproduction instructions, student tasks and validation receipts. |
| `manifests/` | Immutable data/model versions, input checksums, split references and result-to-command index. |
| `data/fixtures/` | Clearly labeled small synthetic examples for tests; never mix with observed results. |
| `data/local/` | Downloaded research inputs and local caches; ignored by Git. Publish needed real inputs on the governed data host and pin them in the manifest. |
| `experiments/` | Cohort selection, labels, split IDs, exclusions and decision-time evidence preparation. |
| `models/` | Model adapters, prompts, checkpoint/serving settings and response parsers. |
| `policies/` | Cost matrix, action baselines, thresholds and validation-only calibration. |
| `guards/` | Timestamp/citation checks, corruption fixtures and guard-on/off evaluation. |
| `analysis/` | Metrics, paired case bootstrap, comparisons, robustness and cost/economic analyses. |
| `figures/` | Scripts and styles that regenerate paper figures and tables from analysis outputs. |
| `notebooks/` | Reader walkthrough and actual hosted-Colab replay after the real pipeline works. |
| `tests/` | Scientific contract, leakage, checksum, policy, parser and reference-comparison tests. |
| `oracle_audit/` | Command-line entry points and shared pipeline helpers; replace the fixture-only commands with reviewed research implementations. |
| `.github/` | Automated checks and pull-request review template. |
| `outputs/` | Generated local results, logs, tables and figures; ignored by Git. Deposit governed raw runs/predictions/references on the data host and link them in the result index. |

`data/local/` and `outputs/` are created when needed, so they are absent from the initial Git tree. No credentials, private manuscripts, model weights or bulk Atlas files belong in this repository.

At the root keep the README, runtime version, dependency lock and Makefile. The code license and citation metadata need author confirmation before release.

**Boundary:** full Atlas extraction/ETL, global dictionaries, lifecycle reconstruction, dataset card, Croissant and resource-wide validation belong to the descriptor project. This repository consumes a pinned release and owns the research experiment. A missing input must remain an explicit blocker.
