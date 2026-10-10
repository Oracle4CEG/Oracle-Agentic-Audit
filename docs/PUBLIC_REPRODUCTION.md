# Public research reproduction

[Open the commit-pinned Colab notebook](https://colab.research.google.com/github/virusLuke3/Oracle-Agentic-Audit/blob/86a900e6a8add44cabd15b11b3f61ce2206addcb/notebooks/Oracle_Agentic_Audit_Replay.ipynb).
Its implementation commit is `33ec7020e97fc6b8dd0005415d3184ce889d3e28`; the notebook
itself is frozen at `86a900e6a8add44cabd15b11b3f61ce2206addcb`.

The fixed Hugging Face commit and archive hashes are in
[public-release.json](../manifests/public-release.json). Archives have separate
`atlas/selected-inputs/2026-10-10/` and `research/agentic-audit/2026-10-10/` prefixes.
The former contains four curated UMA source tables; the latter contains the
experiment-specific cohort, evidence and run records, recovery history and reference
outputs. This is not a publication or semantic clearance of the full Atlas.

## Fresh CPU run

Use a fresh checkout, Python 3.12.7 and a CPU runtime. Allow approximately 2 GB of
disk for code, downloaded archives and extracted records, plus the Python environment
and new outputs. The two archives total approximately 188 MB. The earlier complete
local replay took about 13 minutes; machine speed and package downloads vary. No
model endpoint, GPU, account, API key or server credential is required.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip check
.venv/bin/python -m experiments.download
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m oracle_audit experiment-check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m analysis.replay --retrain --out outputs/replay
.venv/bin/python -m analysis.verify_replay --reference outputs/final --candidate outputs/replay
```

The downloader checks archive hashes, every indexed file hash and size, unique safe
paths and regular-file types before extraction. Different existing scientific
records are never overwritten. Delete nothing to resolve that error: use another
fresh checkout. Download/extraction is resumable if existing files match.

Inspect `outputs/downloads/download_receipt.json`, the replay's `replay_receipt.json`
and `comparison_receipt.json`. Required results are 73 CSV comparisons within
absolute/relative tolerance 1e-10, 18 identical numeric LaTeX tables plus one prompt
listing, and eight regenerated figures. Figure hashes are reported separately:
rendering may differ with platform fonts while the numeric inputs agree.

The saved research notebook runs these same commands. A completed notebook with
its actual outputs is an execution record. A hosted-Colab PASS requires execution
in a fresh Google Colab runtime, with a timestamped receipt; preparing a Colab link,
running locally or passing GitHub Actions does not satisfy that condition.

## Scope and provenance

- 810 protocol-adjudicated disputed cases: 489 training, 161 validation, 160 test.
- Original main Qwen panel: 1,600 case-runs; original corruption panel: 320.
- Necessary supplemental panel: 483 validation, 320 matched B2 (repeats 3 and 4),
  480 A1 guard runs. Optional low-temperature experiments were cancelled.
- Archive failed, interrupted and superseded attempts separately from active
  panel means. Missing compute usage is unknown rather than zero.
- Historical GPT-5.6-sol records are preserved separately and never relabeled Qwen.
- Source paths in historical receipts are original non-secret provenance. Replay
  uses relative paths and does not need access to those machines.
- Local checkpoint hashes are recorded; the immutable public checkpoint revision
  and training-data cutoff have not been independently verified. The replay makes
  no fresh-model-generation claim.

## Licensing and final acceptance

The established dataset license covers original curated data; source snapshots
and third-party content retain their original terms. No model weights or private
manuscript are included. The author confirmed MIT for original research code on 10 October 2026; see
[LICENSE](../LICENSE) and [the confirmation record](../manifests/license-approval.json).
This includes the original scientific implementation at commit
`33ec7020e97fc6b8dd0005415d3184ce889d3e28`. Third-party rights are unchanged. The descriptor's actual arXiv version, independent review and hosted-Colab
receipt remain final publication gates, separate from an executable draft PR.
