"""Metrics use repeated-run means and case-cluster paired uncertainty."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from policies.decision import loss, Costs


METRICS = ['auroc','brier','ece10','decision_loss','coverage','selective_error']


def one(y, p, action, costs=Costs()):
    y, p, a = np.asarray(y), np.asarray(p), np.asarray(action)
    auto = np.isin(a, ['Accept', 'Challenge'])
    error = ((a == 'Accept') & (y == 1)) | ((a == 'Challenge') & (y == 0))
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.abs(y[bins == b].sum()-p[bins == b].sum()) for b in range(10))/len(y)
    return dict(auroc=roc_auc_score(y,p) if len(np.unique(y)) == 2 else np.nan,
                brier=float(np.mean((p-y)**2)), ece10=float(ece),
                decision_loss=float(loss(y,a,costs).mean()), coverage=float(auto.mean()),
                selective_error=float(error[auto].mean()) if auto.any() else np.nan)


def repeated(frame, costs=Costs()):
    values = [one(g.label, g.p_reject, g.action, costs) for _,g in frame.groupby('repeat')]
    return {m: float(pd.Series([v[m] for v in values]).mean()) for m in METRICS}


def validate_panel(frame):
    if frame.duplicated(['case_id','repeat']).any():
        raise ValueError('Duplicate case/run')
    counts = frame.groupby('case_id')['repeat'].agg(lambda x: tuple(sorted(x)))
    if counts.nunique() != 1:
        raise ValueError('Incomplete repeated-measures panel')
    if frame.groupby('case_id').label.nunique().max() > 1:
        raise ValueError('Inconsistent repeated labels')


def paired_bootstrap(left, right, n=2000, seed=20261009):
    for frame in [left, right]:
        validate_panel(frame)
    cases = sorted(left.case_id.unique())
    if cases != sorted(right.case_id.unique()):
        raise ValueError('Paired comparison requires identical case sets')
    if left.groupby('case_id').label.first().sort_index().equals(right.groupby('case_id').label.first().sort_index()) is False:
        raise ValueError('Paired labels differ')
    def arrays(frame):
        return [(g.set_index('case_id').loc[cases].label.to_numpy(),
                 g.set_index('case_id').loc[cases].p_reject.to_numpy(),
                 g.set_index('case_id').loc[cases].action.to_numpy()) for _,g in frame.groupby('repeat')]
    lparts, rparts = arrays(left), arrays(right)
    rng = np.random.default_rng(seed)
    draws = {m: [] for m in METRICS}
    for _ in range(n):
        idx = rng.integers(0, len(cases), len(cases))
        sides = []
        for parts in [lparts, rparts]:
            vals = [one(y[idx],p[idx],a[idx]) for y,p,a in parts]
            sides.append({m: np.mean([v[m] for v in vals if np.isfinite(v[m])])
                          if any(np.isfinite(v[m]) for v in vals) else np.nan for m in METRICS})
        for m in METRICS:
            draws[m].append(sides[1][m]-sides[0][m])
    lm, rm = repeated(left), repeated(right)
    return pd.DataFrame([dict(metric=m, difference=rm[m]-lm[m],
                              ci_low=np.nanquantile(draws[m],.025) if np.isfinite(draws[m]).any() else np.nan,
                              ci_high=np.nanquantile(draws[m],.975) if np.isfinite(draws[m]).any() else np.nan,
                              resamples=n,valid_resamples=int(np.isfinite(draws[m]).sum()),cases=len(cases),unit='case; all runs retained') for m in METRICS])
