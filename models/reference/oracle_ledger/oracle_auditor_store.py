"""Checksum-pinned, read-only evidence access for the Oracle Auditor experiment."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


POST_CUTOFF = "POST_CUTOFF_EVIDENCE_REJECTED"
UNKNOWN_EVIDENCE = "UNKNOWN_EVIDENCE_ID"
CUTOFF_MISMATCH = "CASE_CUTOFF_MISMATCH"
POISON_MARKERS = (
    "ignore the evaluation policy",
    "ignore previous instructions",
    "override the audit policy",
    "do not abstain",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def parse_cutoff(value: int | str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("cutoff must be a Unix timestamp") from exc


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    evidence_ids: list[str]
    errors: list[str]
    warnings: list[str]
    max_evidence_time_unix: int | None
    poison_detected: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "evidence_ids": self.evidence_ids,
            "errors": self.errors,
            "warnings": self.warnings,
            "max_evidence_time_unix": self.max_evidence_time_unix,
            "poison_detected": self.poison_detected,
        }


class EvidenceStore:
    """Load one immutable JSON evidence store and verify its external pin."""

    def __init__(self, store_path: str | Path, expected_sha256: str):
        self.path = Path(store_path).resolve()
        raw = self.path.read_bytes()
        observed = sha256_bytes(raw)
        if observed != expected_sha256:
            raise RuntimeError(
                f"evidence store checksum mismatch: expected {expected_sha256}, got {observed}"
            )
        self.store_sha256 = observed
        self.data = json.loads(raw)
        if self.data.get("schema_version") != "oracle-auditor-evidence-v1":
            raise RuntimeError("unsupported evidence store schema")
        self.cases: dict[str, dict[str, Any]] = self.data["cases"]
        self.evidence: dict[str, dict[str, Any]] = self.data["evidence"]
        self.actor_history: list[dict[str, Any]] = self.data["actor_history"]
        self.prior_cases: list[dict[str, Any]] = self.data["prior_cases"]

    def _case(self, case_id: str, cutoff: int | str) -> dict[str, Any]:
        case = self.cases.get(str(case_id))
        if case is None:
            raise KeyError(f"unknown case_id: {case_id}")
        cutoff_int = parse_cutoff(cutoff)
        if cutoff_int != int(case["cutoff_unix"]):
            raise ValueError(CUTOFF_MISMATCH)
        return case

    def get_proposal(self, case_id: str, cutoff: int | str) -> dict[str, Any]:
        case = self._case(case_id, cutoff)
        return {
            "case_id": case_id,
            "cutoff_unix": int(case["cutoff_unix"]),
            "proposal": case["proposal"],
            "required_evidence_ids": list(case["required_evidence_ids"]),
            "optional_evidence_ids": list(case.get("optional_evidence_ids", [])),
            "evidence_store_sha256": self.store_sha256,
            "label_boundary": "protocol adjudication is evaluation-only and unavailable to tools",
        }

    def get_source_evidence(self, evidence_id: str, cutoff: int | str) -> dict[str, Any]:
        record = self.evidence.get(str(evidence_id))
        if record is None:
            raise KeyError(UNKNOWN_EVIDENCE)
        cutoff_int = parse_cutoff(cutoff)
        timestamp = int(record["evidence_time_unix"])
        if timestamp > cutoff_int:
            raise ValueError(POST_CUTOFF)
        if record.get("content_sha256") != sha256_json(record.get("content")):
            raise RuntimeError("EVIDENCE_CONTENT_CHECKSUM_MISMATCH")
        return {
            "evidence_id": evidence_id,
            "case_id": record["case_id"],
            "kind": record["kind"],
            "evidence_time_unix": timestamp,
            "source": record["source"],
            "content": record["content"],
            "content_sha256": record["content_sha256"],
        }

    def get_actor_history(
        self,
        address: str,
        cutoff: int | str,
        *,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        cutoff_int = parse_cutoff(cutoff)
        normalized = str(address).lower()
        candidates = [
            row for row in self.actor_history
            if row["address"] == normalized and int(row["cutoff_unix"]) <= cutoff_int
            and (
                case_id is None
                or self.evidence.get(str(row.get("evidence_id")), {}).get("case_id") == str(case_id)
            )
        ]
        if not candidates:
            return {
                "address": normalized,
                "cutoff_unix": cutoff_int,
                "status": "no_prior_history",
                "completed": 0,
                "dispute_rate": 0.0,
                "rejection_rate": 0.0,
                "evidence_id": None,
            }
        row = max(candidates, key=lambda item: int(item["cutoff_unix"]))
        return dict(row)

    @staticmethod
    def _tokens(query: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9]{3,}", str(query).lower())
            if token not in {"this", "that", "with", "from", "will", "market", "resolve"}
        }

    def search_prior_cases(
        self, query: str, cutoff: int | str, k: int = 5
    ) -> list[dict[str, Any]]:
        cutoff_int = parse_cutoff(cutoff)
        limit = max(1, min(int(k), 5))
        query_tokens = self._tokens(query)
        scored: list[tuple[float, int, str, dict[str, Any]]] = []
        for row in self.prior_cases:
            decision_time = int(row["decision_time_unix"])
            if decision_time >= cutoff_int:
                continue
            candidate_tokens = self._tokens(row.get("search_text", ""))
            union = query_tokens | candidate_tokens
            similarity = len(query_tokens & candidate_tokens) / len(union) if union else 0.0
            scored.append((similarity, decision_time, row["case_id"], row))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [
            {
                "case_id": row["case_id"],
                "decision_time_unix": int(row["decision_time_unix"]),
                "similarity": round(score, 8),
                "proposal_summary": row["proposal_summary"],
                "evidence_ids": row["evidence_ids"],
                "label_boundary": "no dispute, resolution, settlement, payout, or protocol label exposed",
            }
            for score, _, _, row in scored[:limit]
        ]

    def validate_evidence(
        self,
        evidence_ids: Iterable[str],
        cutoff: int | str,
        *,
        required_evidence_ids: Iterable[str] = (),
        observed_payloads: Iterable[Any] = (),
    ) -> ValidationResult:
        cutoff_int = parse_cutoff(cutoff)
        ids = list(dict.fromkeys(str(item) for item in evidence_ids))
        required = set(str(item) for item in required_evidence_ids)
        errors: list[str] = []
        warnings: list[str] = []
        times: list[int] = []
        kinds: dict[tuple[str, str], str] = {}
        poison_detected = False

        missing_required = sorted(required - set(ids))
        if missing_required:
            errors.append("MISSING_REQUIRED_EVIDENCE:" + ",".join(missing_required))

        for evidence_id in ids:
            record = self.evidence.get(evidence_id)
            if record is None:
                errors.append(f"{UNKNOWN_EVIDENCE}:{evidence_id}")
                continue
            timestamp = int(record["evidence_time_unix"])
            times.append(timestamp)
            if timestamp > cutoff_int:
                errors.append(f"{POST_CUTOFF}:{evidence_id}")
            content = record.get("content")
            if record.get("content_sha256") != sha256_json(content):
                errors.append(f"EVIDENCE_CONTENT_CHECKSUM_MISMATCH:{evidence_id}")
            kind = str(record["kind"])
            case_kind = (str(record["case_id"]), kind)
            digest = sha256_json(content)
            if case_kind in kinds and kinds[case_kind] != digest:
                errors.append(f"CONFLICTING_EVIDENCE_KIND:{kind}")
            kinds[case_kind] = digest
            lowered = canonical_json(content).lower()
            if any(marker in lowered for marker in POISON_MARKERS):
                poison_detected = True

        for payload in observed_payloads:
            lowered = canonical_json(payload).lower()
            if any(marker in lowered for marker in POISON_MARKERS):
                poison_detected = True
        if poison_detected:
            errors.append("TOOL_OR_OUTPUT_POISONING_DETECTED")

        return ValidationResult(
            valid=not errors,
            evidence_ids=ids,
            errors=errors,
            warnings=warnings,
            max_evidence_time_unix=max(times) if times else None,
            poison_detected=poison_detected,
        )


def probability_action(probability: float) -> str:
    """Apply the paper thresholds in deterministic code."""
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("p_reject must be in [0, 1]")
    if p <= 0.20:
        return "Accept"
    if p < 0.40:
        return "Investigate"
    if p < 0.60:
        return "Abstain"
    if p < 0.80:
        return "Investigate"
    return "Challenge"


def enforce_safety(
    raw_output: dict[str, Any], validation: ValidationResult
) -> dict[str, Any]:
    """Normalize model JSON and deterministically override unsafe decisions."""
    try:
        raw_probability = raw_output["p_reject"]
        if isinstance(raw_probability, bool) or not isinstance(raw_probability, (int, float)):
            raise TypeError
        if not 0.0 <= raw_probability <= 1.0:
            raise ValueError
        probability = float(raw_probability)
    except (KeyError, TypeError, ValueError):
        probability = 0.5
        validation = ValidationResult(
            valid=False,
            evidence_ids=[],
            errors=[*validation.errors, "INVALID_PROBABILITY_OUTPUT"],
            warnings=validation.warnings,
            max_evidence_time_unix=validation.max_evidence_time_unix,
            poison_detected=validation.poison_detected,
        )

    if not validation.valid:
        return {
            "p_reject": probability,
            "action": "Abstain",
            "evidence_ids": validation.evidence_ids,
            "abstention_reason": "; ".join(validation.errors),
            "human_review": True,
        }

    action = probability_action(probability)
    human_review = action in {"Investigate", "Abstain"}
    reason = raw_output.get("abstention_reason") if action == "Abstain" else None
    if action == "Abstain" and not reason:
        reason = "probability falls in the preregistered abstention interval"
    return {
        "p_reject": probability,
        "action": action,
        "evidence_ids": validation.evidence_ids,
        "abstention_reason": reason,
        "human_review": human_review,
    }
