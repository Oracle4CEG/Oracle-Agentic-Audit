"""Verify and normalize archived Qwen outputs without converting failed outputs to successes."""
import json
import numpy as np
import pandas as pd
from oracle_audit.io import ROOT,sha256,save_json,table
from experiments.prepare import read_events

ORIGINAL=ROOT/'data/local/qwen_original/Results/AgenticOracleAuditor_Qwen38_27B_20260901_formal_v1'


def matched_evidence(folder):
    events,_=read_events(folder/'events.jsonl')
    # The request logs are authoritative for what actually entered the context.
    # Tool calls/results not followed by another model request are not model-visible.
    requests=[e for e in events if e.get('type')=='api_request']
    if not requests:return []
    messages=requests[-1]['request']['messages']
    names={}
    for m in messages:
        for call in m.get('tool_calls') or []:
            names[call['id']]=dict(name=call['function']['name'],arguments=call['function'].get('arguments','{}'))
    return [dict(**names.get(m.get('tool_call_id'),{}),content=m['content']) for m in messages if m.get('role')=='tool']


def row_from_folder(folder, labels, store, split=None):
    m=json.loads((folder/'run_meta.json').read_text());v=json.loads((folder/'validation.json').read_text())
    raw=json.loads((folder/'raw_output.json').read_text()) or {};case=m['case_id']
    final=m['final'];valid=m['raw_output_valid'];rp=raw.get('p_reject')
    raw_ids=raw.get('evidence_ids',[])
    citation_schema_invalid=not isinstance(raw_ids,list) or any(not isinstance(x,str) for x in raw_ids)
    normalized_ids=[x for x in raw_ids if isinstance(x,str)] if isinstance(raw_ids,list) else []
    probability_valid=isinstance(rp,(int,float)) and not isinstance(rp,bool) and np.isfinite(rp) and 0<=rp<=1
    return dict(case_id=case,repeat=m['repeat'],system=m['system'],model_id=m['model_id'],
        split=split or store['cases'][case]['split'],scenario=m['scenario'],label=int(labels[case]),
        p_reject=final['p_reject'],probability_imputed=not probability_valid,action=final['action'],raw_action=raw.get('action'),
        guard_blocked=not v['valid'],output_valid=valid,required_citation_complete=v['required_evidence_citation_complete'],
        citation_valid=v['evidence_identifier_citation_valid'],evidence_ids=json.dumps(normalized_ids),citation_schema_invalid=citation_schema_invalid,
        guard_errors=json.dumps(v['errors']),tool_calls=m['tool_call_count'],tool_successes=m['tool_call_successes'],
        cap_reached=m['tool_call_count']>=8,input_tokens=m['input_tokens'],output_tokens=m['output_tokens'],
        cached_input_tokens=m['cached_input_tokens'],total_tokens=m['total_tokens'],latency_seconds=m['latency_seconds'],
        api_equivalent_usd=np.nan,timed_out=m['timed_out'],return_code=m['return_code'],
        reasoning_boundary_violation=m.get('reasoning_boundary_violation',False),source=str(folder.relative_to(ROOT)),
        attempt_number=m.get('attempt_number',1),infrastructure_recovery=m.get('infrastructure_recovery',False),
        prior_attempt_path=m.get('prior_attempt_path'),
        serving_revision=m.get('serving_revision','original_formal_manifest' if folder.is_relative_to(ORIGINAL) else 'initial_no_cuda_graph'))


def import_original(out=None):
    manifest=json.loads((ORIGINAL/'experiment_manifest.json').read_text())
    index=ORIGINAL/'artifact_index.jsonl'
    if sha256(index)!=manifest['artifact_index']['sha256']:raise ValueError('Original Qwen artifact index hash mismatch')
    verified=0
    for line in index.read_text().splitlines():
        entry=json.loads(line);p=ORIGINAL/entry['path']
        if not p.is_file() or sha256(p)!=entry['sha256']:raise ValueError(f'Original Qwen artifact mismatch: {p}')
        verified+=1
    store=json.loads((ROOT/'data/local/evidence_store.json').read_text())
    if sha256(ROOT/'data/local/evidence_store.json')!=manifest['evidence_store']['sha256']:raise ValueError('Qwen evidence mismatch')
    labels=pd.read_parquet(ROOT/'data/local/samples.parquet').set_index('sample_id').proposal_rejected_by_protocol.to_dict()
    rows=[];matched=[]
    for group in ['runs','robustness']:
        for meta in sorted((ORIGINAL/group).rglob('run_meta.json')):
            folder=meta.parent;row=row_from_folder(folder,labels,store);rows.append(row)
            if group=='runs' and row['system']=='A1':
                matched.append(dict(case_id=row['case_id'],repeat=row['repeat'],model_id=row['model_id'],
                                    source_events_sha256=sha256(folder/'events.jsonl'),
                                    observed_tool_responses=matched_evidence(folder)))
    destination=out or ROOT/'data/local'
    frame=pd.DataFrame(rows);table(destination,'qwen_original_runs',frame)
    p=destination/'qwen_matched_evidence.jsonl'
    p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in matched))
    save_json((out or ROOT/'outputs/audit')/'qwen_original_integrity.json',dict(verified_files=verified,expected_files=manifest['artifact_index']['indexed_per_run_files'],
              evidence_hash_verified=True,main_case_runs=int(frame.scenario.eq('clean').sum()),robust_case_runs=int(frame.scenario.ne('clean').sum()),
              invalid_outputs=int((~frame.output_valid).sum()),
              probability_fallback='Original final-output p=.5 on parse/probability failure; marked probability_imputed, kept in ITT denominators.'))
    print('Original Qwen files verified:',verified,'normalized runs:',len(frame),flush=True)
    return frame
