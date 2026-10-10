"""Paired full-admission ablation and clean/corrupt differences with explicit denominators."""
import json
import numpy as np
import pandas as pd
from oracle_audit.io import table,save_json
from policies.decision import registered
from analysis.metrics import repeated,paired_bootstrap


def analyze(frame,out,bootstrap=True):
    if 'serving_revision' in frame and frame.serving_revision.nunique()!=1:
        raise ValueError('Full guard contrasts require a single serving revision across all conditions')
    rows=[];cis=[];effects=[]
    frame=frame.copy()
    frame['raw_auto']=frame.raw_action.isin(['Accept','Challenge'])
    frame['policy_auto']=np.isin(registered(frame.p_reject),['Accept','Challenge'])
    frame['final_auto']=frame.action.isin(['Accept','Challenge'])
    frame['final_abstain']=frame.action.eq('Abstain')
    for (scenario,system,mode),g in frame.groupby(['scenario','system','guard_mode']):
        rows.append(dict(scenario=scenario,system=system,guard_mode=mode,cases=g.case_id.nunique(),case_runs=len(g),
             schema_valid_rate=g.output_valid.mean(),required_citation_rate=g.required_citation_complete.mean(),
             citation_id_valid_rate=g.citation_valid.mean(),guard_would_block_rate=g.guard_blocked.mean(),
             raw_auto_rate=g.raw_auto.mean(),unenforced_policy_auto_rate=g.policy_auto.mean(),final_auto_rate=g.final_auto.mean(),
             model_safe_abstain_rate=g.raw_safe_abstention.mean(),final_abstain_rate=g.final_abstain.mean(),
             future_exposure_rate=g.future_evidence_exposed.mean(),future_citation_rate=g.future_ids_cited.mean(),
             future_citation_without_content_rate=g.future_ids_cited_without_content.mean(),
             poison_text_received_rate=g.poison_text_received.mean(),
             changed_record_received_rate=g.changed_record_content_received.mean(),
             expected_violation_detection_rate=g.expected_violation_detected.dropna().astype(float).mean(),
             literal_injection_target_compliance_rate=g.literal_injection_target_compliance.mean(),
             cap_rate=g.cap_reached.mean(),transport_failures=int((g.return_code.ne(0)|g.timed_out).sum()),**repeated(g)))
    table(out,'guard_scenario_metrics',pd.DataFrame(rows))
    if bootstrap:
        for (scenario,system),g in frame.groupby(['scenario','system']):
            left=g[g.guard_mode.eq('all_off')];right=g[g.guard_mode.eq('on')]
            ci=paired_bootstrap(left,right).assign(scenario=scenario,system=system,contrast='on minus all_off')
            cis.append(ci)
        table(out,'guard_paired_bootstrap',pd.concat(cis,ignore_index=True))
    rng=np.random.default_rng(20261009)
    for (system,mode),g in frame.groupby(['system','guard_mode']):
        clean=g[g.scenario.eq('clean')]
        for scenario,corrupt in g[g.scenario.ne('clean')].groupby('scenario'):
            matched=clean.merge(corrupt,on=['case_id','repeat'],suffixes=('_clean','_corrupt'),validate='one_to_one')
            for metric in ['p_reject','raw_auto','final_auto','final_abstain']:
                delta=matched[metric+'_corrupt'].astype(float).to_numpy()-matched[metric+'_clean'].astype(float).to_numpy()
                means=delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(axis=1)
                effects.append(dict(system=system,guard_mode=mode,scenario=scenario,metric=metric,cases=len(delta),
                                    difference=delta.mean(),ci_low=np.quantile(means,.025),ci_high=np.quantile(means,.975),
                                    contrast='corrupt minus clean',resamples=2000))
    table(out,'corruption_effects',pd.DataFrame(effects))
    save_json(out/'guard_analysis_scope.json',dict(
        design=f"{frame.case_id.nunique()} fixed test cases x {frame.system.nunique()} system(s) x {frame.scenario.nunique()} scenarios x {frame.guard_mode.nunique()} modes = {len(frame)} case-runs; systems: {','.join(sorted(frame.system.unique()))}",
        ablation='all_off removes source timestamp admission, evidence validation and final guard override; schema checks and tool budget remain',
        clean_false_abstention='Report structural clean-input validation separately from model contract failures and the annotated lexical guard audit.',
        plausible_wrong='Label-blind corrupted proposer history with recomputed internal hashes; tests a compromised upstream snapshot, not tampering against an independently trusted digest.',
        limits='A1-only required panel after user scope reduction; no new B1 guard-ablation claim. One run per condition. Synthetic interventions do not estimate real attack prevalence. Admission, validation and final enforcement are jointly ablated here.'))
