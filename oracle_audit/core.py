"""Small checked examples; no scientific experiment is implemented here."""
import csv
import hashlib
from pathlib import Path


def checked_file(path, expected_sha256):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing input: {path}")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("A complete SHA-256 input checksum is required")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError(f"Checksum mismatch: {path}")
    return path


def evidence_is_admissible(source_time, cutoff_time, is_outcome=False):
    """Illustration using integer UTC times; real provenance checks remain to implement."""
    return not is_outcome and source_time <= cutoff_time


def metrics(rows):
    """Illustrative fixed costs; investigate ends the decision in this fixture only."""
    if not rows:
        raise ValueError("No predictions")
    seen = set()
    total_brier = total_loss = 0.0
    automatic = errors = 0
    for row in rows:
        key = (row['case_id'], row['run_id'])
        if key in seen:
            raise ValueError("Duplicate case/run")
        seen.add(key)
        y = int(row['label'])
        probability = float(row['p_reject'])
        action = row['action']
        if y not in (0, 1) or not 0 <= probability <= 1:
            raise ValueError("Invalid label or probability")
        if action not in ('uphold', 'reject', 'investigate', 'abstain'):
            raise ValueError("Invalid action")
        total_brier += (probability-y)**2
        if action in ('uphold', 'reject'):
            automatic += 1
            wrong = int((action == 'reject') != bool(y))
            errors += wrong
            total_loss += wrong
        else:
            total_loss += 0.1 if action == 'investigate' else 0.2
    n = len(rows)
    return {'case_runs': n, 'unique_cases': len({r['case_id'] for r in rows}),
            'brier': total_brier/n, 'coverage': automatic/n,
            'selective_error': errors/automatic if automatic else None,
            'mean_action_loss': total_loss/n}


def read_predictions(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))
