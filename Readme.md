# Oracle-Agentic-Audit

Reproducible evaluation of agentic auditing for blockchain oracle adjudication.

**Status: research scaffold.** The executable example uses four synthetic cases. It does not reproduce the paper, establish scientific validity, or certify the hosted Colab runtime. The paper reproduction gate intentionally fails until Huaiyu supplies and validates the real artifacts.

## Quick start

Python 3.12.14; no third-party runtime dependencies or credentials are needed for the template.

```bash
python -m oracle_audit smoke
python -m unittest discover -s tests -v
python -m oracle_audit template-check
python -m oracle_audit release-check  # expected to fail while the research manifest is incomplete
```

The smoke command checks a hashed synthetic input and regenerates a small metrics CSV. Generated outputs are ignored by Git. A green template check means the scaffold works; it says nothing about the paper's reported results.

## The project boundary

| Product | Responsibility |
| --- | --- |
| Data Descriptor and its data/code release | Atlas acquisition, full protocol mapping, dictionary, provenance, lifecycle reconstruction, inventory, resource validation, dataset card and Croissant metadata. These follow the existing descriptor work plan. |
| This research code repository | Experiment cohort and split, decision-time evidence filters, labels used by the experiment, model runners and prompts, baselines, calibration, action policies, guard tests, ablations, raw run records, paired analysis and table/figure generation. |
| Research manuscript project | One scientific text, figures, tables and bibliography used by separate NeurIPS and ACM venue wrappers. Cite the public descriptor preprint and versioned data release. |

Consume the descriptor release through a fixed input manifest. Keep the full Atlas ingestion and data publication pipeline with the descriptor. Retain enough experimental detail here to reproduce the research. [Detailed boundary](docs/PROJECT_BOUNDARY.md).

## Start here, Huaiyu

1. Read [the completion instructions](docs/HUAIYU_COMPLETION.md).
2. Fill [the research input manifest](manifests/research-inputs.json) with immutable versions and real checksums.
3. Implement the experiment modules and fill [the result replication index](manifests/result-index.json).
4. Provide an offline saved-prediction replay and a separate full-inference path. See [the reproduction contract](docs/REPRODUCIBILITY.md).
5. Run the release gate and independent clean-environment replay before claiming paper replication.

[Concise folder guide: what students should put in each folder](docs/FOLDER_GUIDE.md).

## Planned research modules

| Area | Required implementation |
| --- | --- |
| `experiments/` | Cohort, chronological splits, evidence cutoffs and label validation |
| `models/` | Exact checkpoints, serving adapters, prompts and response parsing |
| `policies/` | Cost matrix, actions, thresholds and validation-only calibration |
| `guards/` | Citation/time validity and corruption controls |
| `analysis/` | Per-case metrics, paired case bootstrap, calibration, ablations and cost analysis |
| `figures/` | Deterministic publication figure and table generation |
| `notebooks/` | Hosted Colab walkthrough after the offline replay works |

Each directory currently contains requirements rather than a completed scientific implementation. [Result-to-code index](manifests/result-index.json).

## Data, citation and license

Canonical dataset: [OracleEconLab-Oracle-Incentives-v1](https://huggingface.co/datasets/Oracle4CEG/OracleEconLab-Oracle-Incentives-v1). Its current online presence alone does not establish that every experimental input is available. Pin the exact revision and file hashes after validation.

The descriptor arXiv identifier, research citation, code license and applicable input licenses are pending author confirmation. No license is granted by this scaffold. Do not create an invented DOI, model identifier or arXiv citation.

## Validation and limitations

See [the template validation receipt](docs/TEMPLATE_VALIDATION.md). Full scientific replication, real model inference, GPU requirements, independent replay, hosted CI and hosted Colab remain unverified. Synthetic examples must remain clearly separated from observed research outputs.
