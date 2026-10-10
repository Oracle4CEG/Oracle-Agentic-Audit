# Decision policies

`decision.py` defines the explicit terminal action-cost matrix, original thresholds, cost-derived thresholds, mandatory guard overrides and validation-only threshold selection. `calibration.py` fits Platt scaling only on validation cases and prevents repeated runs from multiplying case weight. Always-Investigate is an action-only reference; it has no invented probability AUROC.
