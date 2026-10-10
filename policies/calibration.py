"""Regularized Platt calibration, fit on chronological validation cases only."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1, 1)


def fit(validation):
    if set(validation.split) != {'validation'}:
        raise ValueError('Calibration requires validation rows only')
    if validation.label.nunique() != 2:
        raise ValueError('Calibration needs both labels')
    # Repeats do not increase the regularization weight of a case.
    weight = 1 / validation.groupby('case_id').case_id.transform('size').to_numpy()
    estimator = LogisticRegression(C=1., random_state=20261009, max_iter=2000)
    estimator.fit(logit(validation.p_reject), validation.label, sample_weight=weight)
    return estimator


def apply(estimator, p):
    return estimator.predict_proba(logit(p))[:, 1]
