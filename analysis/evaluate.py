"""Saved-probability policy replay, validation calibration, and paired inference."""
import json
import numpy as np
import pandas as pd
from oracle_audit.io import ROOT,table,save_json
from policies.decision import Costs,registered,optimal,threshold_policy,tune_thresholds,loss
from policies.calibration import fit,apply
from analysis.metrics import repeated,paired_bootstrap


def align_matched_repeats(b2,a1):
    """Keep exactly the source A1 case/seed pairs, never its unmatched repeats."""
    right=a1[a1.repeat.isin(b2.repeat.unique())].copy()
    keys=['case_id','repeat']
    if b2.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError('Duplicate matched-evidence pair')
    if set(b2[keys].itertuples(index=False,name=None))!=set(right[keys].itertuples(index=False,name=None)):
        raise ValueError('Matched evidence requires identical case and repeat IDs')
    return b2,right


def calibrated_llm(frame, out):
    pieces=[];receipts=[]
    for system,g in frame.groupby('system'):
        g=g.copy();g['calibration']='raw';pieces.append(g)
        val=g[g.split.eq('validation')]
        if val.case_id.nunique()!=161:
            receipts.append(dict(system=system,status='MISSING_VALIDATION_PREDICTIONS',cases=val.case_id.nunique()));continue
        estimator=fit(val)
        pieces.append(g.assign(calibration='platt',p_reject=apply(estimator,g.p_reject)))
        receipts.append(dict(system=system,status='FITTED_VALIDATION_ONLY',cases=val.case_id.nunique(),
                             case_runs=len(val),coefficient=estimator.coef_.tolist(),intercept=estimator.intercept_.tolist()))
    save_json(out/'llm_calibration_receipts.json',receipts)
    return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()


def policies(predictions, out, bootstrap=False):
    summaries=[];sensitivity=[];panels=[];thresholds=[]
    for (model,cal),g in predictions.groupby(['system','calibration']):
        test=g[g.split.eq('test')].copy();valid=g[g.split.eq('validation')]
        if test.empty:continue
        p=test.p_reject.to_numpy();blocked=test.guard_blocked.to_numpy(dtype=bool)
        actions={'registered':registered(p),'cost_derived':optimal(p),
                 'always_investigate':np.full(len(test),'Investigate',dtype='<U12'),
                 'always_accept':np.full(len(test),'Accept',dtype='<U12'),
                 'always_challenge':np.full(len(test),'Challenge',dtype='<U12'),
                 'always_abstain':np.full(len(test),'Abstain',dtype='<U12')}
        if not valid.empty and cal=='raw':
            lo,hi=tune_thresholds(valid);actions['validation_tuned']=threshold_policy(p,lo,hi)
            thresholds.append(dict(system=model,calibration=cal,lower=lo,upper=hi,selection_split='validation'))
        for policy,action in actions.items():
            for guard in ['off','on']:
                a=action.copy()
                if guard=='on':a[blocked]='Abstain'
                panel=test.assign(action=a,policy=policy,guard=guard)
                panels.append(panel)
                summaries.append(dict(system=model,calibration=cal,policy=policy,guard=guard,
                                      cases=test.case_id.nunique(),case_runs=len(test),guard_block_rate=blocked.mean(),**repeated(panel)))
        for ci in [.05,.1,.15,.2,.3,.4]:
            # Abstain remains an available terminal option at .2; it becomes optimal
            # over Investigate above .2. Never silently change the cost matrix.
            costs=Costs(investigate=ci)
            for guard in ['off','on']:
                a=optimal(p,costs,blocked if guard=='on' else None)
                panel=test.assign(action=a)
                sensitivity.append(dict(system=model,calibration=cal,guard=guard,investigate_cost=ci,
                                        abstain_cost=costs.abstain,**repeated(panel,costs)))
    summary=pd.DataFrame(summaries);panel=pd.concat(panels,ignore_index=True)
    table(out,'policy_metrics',summary);table(out,'policy_predictions',panel)
    table(out,'cost_sensitivity',pd.DataFrame(sensitivity));table(out,'validation_thresholds',pd.DataFrame(thresholds))
    matched=[]
    if {'B2','A1'}<=set(panel.system):
        for (calibration,policy,guard),selected in panel.groupby(['calibration','policy','guard']):
            left=selected[selected.system.eq('B2')];right=selected[selected.system.eq('A1')]
            if left.empty or right.empty:continue
            left,right=align_matched_repeats(left,right)
            for system,side in [('B2',left),('A1',right)]:
                matched.append(dict(system=system,calibration=calibration,policy=policy,guard=guard,
                    repeat_ids=','.join(map(str,sorted(side.repeat.unique()))),cases=side.case_id.nunique(),
                    case_runs=len(side),**repeated(side)))
        table(out,'matched_evidence_comparison_metrics',pd.DataFrame(matched))
    if bootstrap:
        cis=[]
        systems=set(panel.system)
        pairs=[('B0','B0plus'),('B1','A1'),('B2','A1'),('B0plus','A1')]
        for l,r in pairs:
            if not {l,r}<=systems:continue
            for calibration in ['raw','platt']:
                for policy in ['registered','cost_derived']:
                    selected=panel[(panel.calibration==calibration)&(panel.policy==policy)&(panel.guard=='on')]
                    left=selected[selected.system==l];right=selected[selected.system==r]
                    if left.empty or right.empty:continue
                    if l=='B2':left,right=align_matched_repeats(left,right)
                    print(f'Bootstrap {r}-{l} {calibration} {policy}',flush=True)
                    ci=paired_bootstrap(left,right)
                    cis.append(ci.assign(left=l,right=r,calibration=calibration,policy=policy,guard='on',
                        left_repeat_ids=','.join(map(str,sorted(left.repeat.unique()))),
                        right_repeat_ids=','.join(map(str,sorted(right.repeat.unique())))))
        if cis:table(out,'paired_bootstrap',pd.concat(cis,ignore_index=True))
    return summary
