"""Capacity scenarios, case-level uncertainty, timing windows and budget diagnostics."""
import numpy as np
import pandas as pd
from oracle_audit.io import ROOT,table,save_json
from analysis.metrics import one,repeated
from policies.decision import Costs,registered


def limited_review(p,blocked,slots,costs=Costs()):
    p=np.asarray(p);blocked=np.asarray(blocked,dtype=bool)
    if blocked.sum()>slots:return None
    action=np.where(p*costs.missed_rejection <= (1-p)*costs.false_challenge,'Accept','Challenge').astype('<U12')
    action[blocked]='Abstain'
    benefit=np.minimum(p*costs.missed_rejection,(1-p)*costs.false_challenge)-costs.investigate
    order=np.argsort(-benefit,kind='stable')
    candidates=[i for i in order if not blocked[i] and benefit[i]>0]
    action[candidates[:max(0,int(slots-blocked.sum()))]]='Investigate'
    return action


def capacity(predictions,out):
    rows=[]
    for (system,cal),frame in predictions[predictions.split.eq('test')].groupby(['system','calibration']):
        for budget in [0,.1,.25,.5,.75,1.]:
            panels=[];infeasible=0;required=[]
            for repeat,g in frame.groupby('repeat'):
                g=g.sort_values('case_id');slots=int(np.floor(len(g)*budget));required.append(g.guard_blocked.mean())
                action=limited_review(g.p_reject,g.guard_blocked,slots)
                if action is None:infeasible+=1
                else:panels.append(g.assign(action=action))
            row=dict(system=system,calibration=cal,budget_fraction=budget,cases=frame.case_id.nunique(),
                     repeats=frame.repeat.nunique(),infeasible_repeats=infeasible,mandatory_review_mean=np.mean(required))
            if infeasible==0:row.update(repeated(pd.concat(panels,ignore_index=True)))
            rows.append(row)
    table(out,'capacity_scenarios',pd.DataFrame(rows))
    save_json(out/'capacity_scope.json',dict(
        scenario='Hypothetical capacity over each batch of 160 test cases, not an observed operational constraint.',
        human_slots='Both Investigate and guard-forced Abstain consume a slot; if mandatory guard review exceeds capacity, the scenario is infeasible and metrics are missing.',
        allocation='Allocate remaining review slots by predicted reduction in terminal decision loss, with case-ID ordering for ties; never use test labels.',
        limitation='No measured human accuracy, arrival process, queue duration or staffing cost.'))


def absolute_ci(frame,n=2000,seed=20261009):
    cases=sorted(frame.case_id.unique());parts=[]
    for _,g in frame.groupby('repeat'):
        g=g.set_index('case_id').loc[cases];parts.append((g.label.to_numpy(),g.p_reject.to_numpy(),g.action.to_numpy()))
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(n):
        idx=rng.integers(0,len(cases),len(cases));vals=[one(y[idx],p[idx],a[idx]) for y,p,a in parts]
        draws.append({key:pd.Series([v[key] for v in vals]).mean() for key in vals[0]})
    draws=pd.DataFrame(draws);point=repeated(frame)
    return pd.DataFrame([dict(metric=key,estimate=value,ci_low=draws[key].quantile(.025),ci_high=draws[key].quantile(.975),
                             cases=len(cases),resamples=n,valid_resamples=int(draws[key].notna().sum())) for key,value in point.items()])


def group_intervals(predictions,cohort,out):
    cis=[];full=[]
    for (system,cal),g in predictions[predictions.split.eq('test')].groupby(['system','calibration']):
        if system=='TrainPrior':continue
        # Report comparable calibrated models once complete; until then use raw LLMs.
        preferred='platt' if not predictions[(predictions.system==system)&(predictions.calibration=='platt')].empty else 'raw'
        g=g.merge(cohort[['sample_id','proposer_seen_train','requester_seen_train']],left_on='case_id',right_on='sample_id',validate='many_to_one')
        g['action']=registered(g.p_reject);g.loc[g.guard_blocked,'action']='Abstain'
        print(f'Case bootstrap: {system} {cal}',flush=True)
        full.append(absolute_ci(g).assign(system=system,calibration=cal,policy='registered',guard='on'))
        if cal!=preferred:continue
        for role in ['proposer','requester']:
            for seen,sub in g.groupby(role+'_seen_train'):
                cis.append(absolute_ci(sub).assign(system=system,calibration=cal,role=role,seen_in_train=seen))
    table(out,'actor_group_confidence_intervals',pd.concat(cis,ignore_index=True))
    table(out,'performance_confidence_intervals',pd.concat(full,ignore_index=True))


def diagnostics(runs,out,bootstrap=True):
    samples=pd.read_parquet(ROOT/'data/local/samples.parquet').set_index('sample_id')
    frame=runs[runs.scenario.eq('clean')&runs.split.eq('test')].copy()
    frame['challenge_window_seconds']=frame.case_id.map((samples.challenge_deadline_unix-samples.decision_time_unix).to_dict())
    frame['latency_fraction_of_window']=frame.latency_seconds/frame.challenge_window_seconds
    frame['latency_exceeds_window']=frame.latency_seconds>frame.challenge_window_seconds
    table(out,'inference_window_records',frame[['case_id','repeat','system','latency_seconds','challenge_window_seconds','latency_fraction_of_window','latency_exceeds_window']])
    table(out,'inference_window_summary',frame.groupby('system').agg(case_runs=('case_id','size'),cases=('case_id','nunique'),
        median_window_seconds=('challenge_window_seconds','median'),median_latency_fraction=('latency_fraction_of_window','median'),
        p95_latency_fraction=('latency_fraction_of_window',lambda x:x.quantile(.95)),over_window_rate=('latency_exceeds_window','mean')).reset_index())
    budget=[]
    for (system,capped),g in frame.groupby(['system','cap_reached']):
        budget.append(dict(system=system,cap_reached=capped,cases=g.case_id.nunique(),case_runs=len(g),
                           complete_citation_rate=g.required_citation_complete.mean(),citation_valid_rate=g.citation_valid.mean(),
                           mean_latency=g.latency_seconds.mean(),**repeated(g)))
    table(out,'tool_cap_performance',pd.DataFrame(budget))
    validity=[]
    for (system,valid),g in frame.groupby(['system','output_valid']):
        validity.append(dict(system=system,output_valid=valid,cases=g.case_id.nunique(),case_runs=len(g),
                             selection='Descriptive output-selected subgroup; not a replacement for the intention-to-evaluate panel',**repeated(g)))
    table(out,'output_validity_sensitivity',pd.DataFrame(validity))
    if bootstrap:
        summaries=[];rng=np.random.default_rng(20261009)
        fields=['required_citation_complete','citation_valid','output_valid','guard_blocked','cap_reached','latency_seconds','total_tokens']
        # Mean over repeated runs within each case, then resample cases.
        for system,g in frame.groupby('system'):
            case=g.groupby('case_id')[fields].mean(numeric_only=True)
            idx=rng.integers(0,len(case),size=(2000,len(case)))
            for field in fields:
                vals=case[field].to_numpy();draws=np.nanmean(vals[idx],axis=1)
                summaries.append(dict(system=system,metric=field,estimate=np.nanmean(vals),ci_low=np.quantile(draws,.025),
                                      ci_high=np.quantile(draws,.975),cases=len(case),case_runs=len(g),resamples=2000))
        table(out,'citation_cost_confidence_intervals',pd.DataFrame(summaries))
    save_json(out/'latency_scope.json',dict(
        comparison='Observed inference wall time versus configured proposal-to-challenge-deadline interval. Assumes inference starts exactly at proposal, solely as a timing scenario.',
        unmeasured='Event publication, finality, indexing latency, waiting before inference and human review time are not included.',
        tool_cap='Cap subgroups are descriptive: reaching the cap is an endogenous consequence of the trajectory, not randomized treatment.'))
