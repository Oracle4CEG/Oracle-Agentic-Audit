"""Outcome-conditional losses; deferral is terminal in this stylized evaluation."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Costs:
    missed_rejection: float = 1.0
    false_challenge: float = 1.0
    investigate: float = 0.1
    abstain: float = 0.2

    def __post_init__(self):
        if any(not np.isfinite(v) or v < 0 for v in self.__dict__.values()):
            raise ValueError("Costs must be finite and nonnegative")


def probabilities(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    return p


def registered(p):
    p = probabilities(p)
    return np.select([p <= .2, p < .4, p < .6, p < .8],
                     ['Accept', 'Investigate', 'Abstain', 'Investigate'], 'Challenge')


def optimal(p, costs=Costs(), blocked=None):
    """Prefer deferral on a tie; handles asymmetric costs without heuristic thresholds."""
    p = probabilities(p)
    risk = np.column_stack([np.full(p.size, costs.investigate),
                            np.full(p.size, costs.abstain),
                            p * costs.missed_rejection,
                            (1-p) * costs.false_challenge])
    best=np.isclose(risk,risk.min(axis=1,keepdims=True),rtol=0,atol=1e-12).argmax(axis=1)
    action = np.array(['Investigate', 'Abstain', 'Accept', 'Challenge'])[best]
    if blocked is not None:
        action[np.asarray(blocked, dtype=bool)] = 'Abstain'
    return action


def loss(y, action, costs=Costs()):
    y = np.asarray(y, dtype=int)
    a = np.asarray(action)
    if not np.isin(y, [0, 1]).all() or not np.isin(a, ['Accept','Challenge','Investigate','Abstain']).all():
        raise ValueError('Invalid label or action')
    return np.select([a == 'Accept', a == 'Challenge', a == 'Investigate'],
                     [y * costs.missed_rejection, (1-y)*costs.false_challenge,
                      np.full(y.size, costs.investigate)], costs.abstain)


def threshold_policy(p, lower, upper, blocked=None):
    p = probabilities(p)
    if not 0 <= lower <= upper <= 1:
        raise ValueError('Invalid thresholds')
    a = np.select([p < lower, p > upper], ['Accept', 'Challenge'], 'Investigate')
    if blocked is not None:
        a[np.asarray(blocked, dtype=bool)] = 'Abstain'
    return a


def tune_thresholds(validation, costs=Costs()):
    if set(validation['split']) != {'validation'}:
        raise ValueError('Threshold fitting requires validation rows only')
    # Small fixed grid; tie-break by less automation, then smaller lower bound.
    choices = []
    for lo in [0, .025, .05, .1, .15, .2, .3]:
        for hi in [.7, .8, .85, .9, .95, .975, 1.0]:
            a = threshold_policy(validation.p_reject, lo, hi, validation.guard_blocked)
            choices.append((float(loss(validation.label, a, costs).mean()),
                            float(np.isin(a, ['Accept','Challenge']).mean()), lo, hi))
    _, _, lo, hi = min(choices)
    return lo, hi
