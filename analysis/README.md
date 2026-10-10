# Research analysis

`replay.py` is the offline entry point. It verifies frozen records, reconstructs the cohort/features, refits feature baselines and recomputes calibration, policies, paired case-bootstrap intervals, citation/guard diagnostics, workload and selected economics. `verify_replay.py` compares 73 reference CSVs and the rendered table/prompt text, recording figure hashes separately.

Run `python -m analysis.replay --retrain --out outputs/replay` after `python -m experiments.download`. This makes no model requests. Full Atlas construction and global semantic validation remain in Oracle4SD-Atlas.
