"""Cohort, evidence, citation, budget, and accounting audits of saved records."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from oracle_audit.io import ROOT,save_json,table
from models.tabular import FEATURES
from policies.decision import registered
from analysis.metrics import repeated


def cohort(out):
    local=ROOT/'data/local';store=json.loads((local/'evidence_store.json').read_text())
    samples=pd.read_parquet(local/'samples.parquet');splits=pd.read_parquet(local/'splits.parquet')
    if samples.sample_id.duplicated().any() or splits.sample_id.duplicated().any():raise ValueError('Duplicate cohort cases')
    frame=samples.merge(splits,on=['sample_id','decision_time_unix'],validate='one_to_one')
    if frame.split.value_counts().to_dict()!={'train':489,'validation':161,'test':160}:raise ValueError('Cohort counts changed')
    for col in ['proposer','requester']:
        frame[col]=frame.sample_id.map(lambda x:store['cases'][x]['proposal'][col])
        seen=set(frame.loc[frame.split.eq('train'),col]);frame[col+'_seen_train']=frame[col].isin(seen)
    frame['decision_time_utc']=pd.to_datetime(frame.decision_time_unix,unit='s',utc=True).astype(str)
    summaries=[]
    for split,g in frame.groupby('split',sort=False):
        summaries.append(dict(split=split,cases=len(g),start_utc=g.decision_time_utc.min(),end_utc=g.decision_time_utc.max(),
                             rejected=int(g.proposal_rejected_by_protocol.sum()),rejection_rate=g.proposal_rejected_by_protocol.mean(),
                             proposers=g.proposer.nunique(),requesters=g.requester.nunique(),
                             unseen_proposer_cases=int((~g.proposer_seen_train).sum()),unseen_requester_cases=int((~g.requester_seen_train).sum())))
    train_end=frame.loc[frame.split.eq('train'),'decision_time_unix'].max()
    val_start=frame.loc[frame.split.eq('validation'),'decision_time_unix'].min()
    val_end=frame.loc[frame.split.eq('validation'),'decision_time_unix'].max()
    test_start=frame.loc[frame.split.eq('test'),'decision_time_unix'].min()
    if not train_end<val_start<=val_end<test_start:raise ValueError('Temporal split overlap')
    public=set(pd.read_parquet(local/'public64.parquet').sample_id)
    overlap=frame[frame.sample_id.isin(public)]
    table(out,'cohort_manifest',frame);table(out,'cohort_summary',pd.DataFrame(summaries))
    table(out,'request_composition',frame.groupby(['split','adapter_version','proposed_price_class']).size().rename('cases').reset_index())
    table(out,'public64_overlap',overlap[['sample_id','split']])
    evidence=[];fields=[];forbidden=set(json.loads((local/'evidence_store_manifest.json').read_text())['forbidden_fields'])
    for case_id,case in store['cases'].items():
        for eid in case['required_evidence_ids']+case.get('optional_evidence_ids',[]):
            record=store['evidence'].get(eid)
            if record is None:raise ValueError('Missing pinned evidence')
            content=record['content'];kind=record['kind'];source_type=record['source'].get('type')
            from hashlib import sha256
            checksum=sha256(json.dumps(content,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
            evidence.append(dict(case_id=case_id,split=case['split'],evidence_id=eid,kind=kind,required=eid in case['required_evidence_ids'],
                                 evidence_time=record['evidence_time_unix'],cutoff=case['cutoff_unix'],source_type=source_type,
                                 source=json.dumps(record['source']),content_hash_valid=checksum==record['content_sha256'],
                                 at_or_before_cutoff=record['evidence_time_unix']<=case['cutoff_unix'],
                                 forbidden_top_level_fields=json.dumps(sorted(set(content)&forbidden)),
                                 availability_basis='on-chain inclusion timestamp; finality delay not measured' if source_type=='polygon_transaction' else 'derived timestamp; source lineage needs separate validation',
                                 independently_measured_publication_time=None))
            for name in content:fields.append(dict(kind=kind,field=name,source_type=source_type))
    ed=pd.DataFrame(evidence);table(out,'evidence_manifest',ed)
    table(out,'model_visible_fields',pd.DataFrame(fields).drop_duplicates())
    table(out,'evidence_availability',ed.groupby(['split','kind','required']).agg(records=('evidence_id','size'),valid_hash=('content_hash_valid','sum'),within_cutoff=('at_or_before_cutoff','sum')).reset_index())
    save_json(out/'cohort_checks.json',dict(counts_verified=True,chronology_verified=True,label_definition='Later UMA adjudication rejects proposal; conditional on disputed Grade-A linked cohort',
               evidence_hashes_pass=bool(ed.content_hash_valid.all()),declared_timestamps_pass=bool(ed.at_or_before_cutoff.all()),
               public64_overlap_counts=overlap.split.value_counts().to_dict(),
               model_features=FEATURES,training_cutoff='unverified',
               limitations=['Model pretraining memorization is not excluded by a prompt or timestamp filter.',
                            'Evidence event timestamps do not independently establish publication/finality availability.',
                            'A source reconstruction audit is needed for derived history and market snapshots.']))
    return frame


def log_audits(runs,out,store=None):
    if store is None:store=json.loads((ROOT/'data/local/evidence_store.json').read_text())
    citations=[];missing=[];counts=[];guard=[]
    for row in runs.itertuples(index=False):
        case=store['cases'][row.case_id];cutoff=case['cutoff_unix'];ids=json.loads(row.evidence_ids)
        for eid in ids:
            record=store['evidence'].get(eid)
            kind='unknown_id' if record is None else ('future_id' if record['evidence_time_unix']>cutoff else 'structurally_valid')
            # Cross-case historical records are allowed; do not call them unauthorized.
            citations.append(dict(case_id=row.case_id,repeat=row.repeat,system=row.system,scenario=row.scenario,evidence_id=eid,
                                  category=kind,semantic_support='not_established_by_id_validity'))
        for eid in sorted(set(case['required_evidence_ids'])-set(ids)):
            missing.append(dict(case_id=row.case_id,repeat=row.repeat,system=row.system,scenario=row.scenario,evidence_id=eid,category='missing_required_citation'))
        raw_policy=registered([row.p_reject])[0]
        guard.append(dict(case_id=row.case_id,repeat=row.repeat,system=row.system,scenario=row.scenario,guard_blocked=row.guard_blocked,
                          model_raw_action=row.raw_action,registered_without_enforcement=raw_policy,guarded_action=row.action,
                          raw_auto=row.raw_action in ['Accept','Challenge'],policy_auto=raw_policy in ['Accept','Challenge'],
                          final_auto=row.action in ['Accept','Challenge']))
    citation_frame=pd.DataFrame(citations);table(out,'citation_records',citation_frame)
    table(out,'missing_required_citations',pd.DataFrame(missing));table(out,'guard_replay_records',pd.DataFrame(guard))
    table(out,'citation_id_breakdown',citation_frame.groupby(['system','scenario','category']).size().rename('citations').reset_index())
    clean=runs[runs.scenario.eq('clean')]
    table(out,'tool_budget_distribution',clean.groupby(['system','tool_calls','cap_reached']).agg(case_runs=('case_id','size'),
          complete_citations=('required_citation_complete','mean'),valid_citations=('citation_valid','mean'),latency_mean=('latency_seconds','mean')).reset_index())
    gr=pd.DataFrame(guard)
    table(out,'guard_replay_summary',gr.groupby(['system','scenario']).agg(case_runs=('case_id','size'),unique_cases=('case_id','nunique'),
          forced_abstention_rate=('guard_blocked','mean'),model_raw_auto_rate=('raw_auto','mean'),unenforced_policy_auto_rate=('policy_auto','mean'),final_auto_rate=('final_auto','mean')).reset_index())
    costs=[]
    for system,g in clean.groupby('system'):
        row=dict(system=system,unique_cases=g.case_id.nunique(),case_runs=len(g),tool_calls=g.tool_calls.sum(),
                 tool_successes=g.tool_successes.sum(),cap_rate=g.cap_reached.mean(),failed_runs=(g.return_code.ne(0)|g.timed_out).sum())
        row['exactly_eight_attempts']=int(g.tool_calls.eq(8).sum())
        row['over_eight_attempts']=int(g.tool_calls.gt(8).sum())
        row['over_eight_successes']=int(g.tool_successes.gt(8).sum())
        if row['over_eight_successes']:raise ValueError('More than eight successful tool calls')
        for metric in ['input_tokens','cached_input_tokens','output_tokens','total_tokens','latency_seconds','api_equivalent_usd']:
            if metric not in g:continue
            row[metric+'_observed']=int(g[metric].notna().sum())
            for name,fn in [('mean',lambda s:s.mean()),('median',lambda s:s.median()),('p95',lambda s:s.quantile(.95))]:
                row[metric+'_'+name]=fn(g[metric]) if g[metric].notna().any() else np.nan
        costs.append(row)
    table(out,'cost_accounting',pd.DataFrame(costs))
    save_json(out/'accounting_and_guard_scope.json',dict(
        cost=('Dollar estimates, where recorded, are frozen API-list-price equivalents. Qwen self-hosted '
              'runs have no measured dollar cost; absent costs remain missing, never zero. Neither '
              'token counts nor API equivalents constitute a measured local GPU bill.'),
        tokens='Input includes cached input. Total = input + output; do not add cache again.',
        guard_replay='Only final enforcement is removed from saved predictions. Input-admission-off inference is a separate experiment.',
        false_abstention='Forced abstention on a clean case is not automatically false positive; missing/invalid model citations can warrant enforcement.',
        citation='Structural ID validity does not establish claim support; semantic relevance requires a separate annotation.',
        latency='Historical end-to-end wall time; queue/model/retrieval components were not separately logged.'))


def actor_groups(predictions,cohort_frame,out):
    rows=[]
    for (system,cal),g in predictions[predictions.split.eq('test')].groupby(['system','calibration']):
        g=g.merge(cohort_frame[['sample_id','proposer_seen_train','requester_seen_train']],left_on='case_id',right_on='sample_id',validate='many_to_one')
        g['action']=registered(g.p_reject);g.loc[g.guard_blocked,'action']='Abstain'
        for role in ['proposer','requester']:
            for seen,group in g.groupby(role+'_seen_train'):
                rows.append(dict(system=system,calibration=cal,role=role,seen_in_train=seen,cases=group.case_id.nunique(),case_runs=len(group),**repeated(group)))
    table(out,'seen_unseen_actor_metrics',pd.DataFrame(rows))
