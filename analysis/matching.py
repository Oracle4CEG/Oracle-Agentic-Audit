"""Check evidence delivery against the exact paired A1 API context."""
import json
from pathlib import Path
from oracle_audit.io import ROOT,table,save_json,sha256
from experiments.qwen_records import matched_evidence
from experiments.prepare import read_events
from models.reference import run_agentic_oracle_auditor as ref
import pandas as pd


def visible_ids(context):
    ids=set()
    for item in context:
        if item.get('name',item.get('tool')) not in {'get_source_evidence','get_actor_history'}:continue
        try:payload=json.loads(item['content'])
        except (ValueError,TypeError):continue
        if isinstance(payload,dict) and isinstance(payload.get('evidence_id'),str):ids.add(payload['evidence_id'])
    return ids


def audit(a1,b2,out):
    rows=[];delivery=[]
    for row in a1.itertuples():
        folder=ROOT/row.source;context=matched_evidence(folder)
        events,_=read_events(folder/'events.jsonl');info=ref.event_metrics(events)
        observed=visible_ids(context)
        rows.append(dict(case_id=row.case_id,repeat=row.repeat,split=row.split,
                         actual_received_content_ids=json.dumps(sorted(observed)),
                         completed_retrieval_not_in_context=len(set(info['retrieved_evidence_ids'])-observed),
                         source_events_sha256=sha256(folder/'events.jsonl')))
    for row in b2.itertuples():
        folder=ROOT/row.source;meta=json.loads((folder/'run_meta.json').read_text())
        source=a1[(a1.case_id==row.case_id)&(a1.repeat==row.repeat)]
        if len(source)!=1:raise ValueError('B2 source A1 pairing is not unique')
        source_path=ROOT/source.iloc[0].source
        expected=[{'tool':t.get('name'),'content':t['content']} for t in matched_evidence(source_path)]
        messages=json.loads((folder/'input_messages.json').read_text())
        packet=json.loads(messages[1]['content'])
        received=packet['observed_tool_responses']
        passed=received==expected and meta['source_a1_events_sha256']==sha256(source_path/'events.jsonl')
        if not passed:raise ValueError('B2 exact-evidence delivery check failed')
        delivery.append(dict(case_id=row.case_id,repeat=row.repeat,split=row.split,exact_content_match=passed,
                             source_events_sha256=meta['source_a1_events_sha256'],tool_responses=len(received)))
    table(out,'a1_actual_received_evidence',pd.DataFrame(rows))
    table(out,'b2_exact_delivery_checks',pd.DataFrame(delivery))
    if rows and any(r['completed_retrieval_not_in_context'] for r in rows):
        raise ValueError('A1 tool completion differs from actual context; B2 guard availability needs explicit correction')
    save_json(out/'matched_evidence_scope.json',dict(
        control='B2 receives only tool name and unmodified tool-result content from the final actual A1 API request; no A1 private reasoning, predictions or assistant/tool-call arguments.',
        source_pairs=len(delivery),all_exact=all(r['exact_content_match'] for r in delivery),
        limitation='The source A1 trajectory is endogenous. Tool validation responses and evidence ordering are part of its received evidence. This comparison isolates repackaged single-pass reasoning conditional on that trajectory, not randomized evidence selection.'))
