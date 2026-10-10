"""Import frozen experiment inputs and normalize original saved LLM runs."""
import json
import shutil
from pathlib import Path
from collections import Counter
import pandas as pd
from oracle_audit.io import ROOT, sha256, save_json, table

INPUTS = {
 'samples.parquet':'data/applications/trustworthy_ai_challenge/decision_samples.parquet',
 'splits.parquet':'data/applications/trustworthy_ai_challenge/splits.parquet',
 'original_b0.parquet':'data/applications/trustworthy_ai_challenge/predictions.parquet',
 'evidence_provenance.parquet':'data/applications/trustworthy_ai_challenge/evidence_provenance.parquet',
 'evidence_store.json':'data/applications/agentic_oracle_auditor/evidence_store.json',
 'evidence_store_manifest.json':'data/applications/agentic_oracle_auditor/evidence_store_manifest.json',
 'public64.parquet':'data/applications/trustworthy_ai_public_source_demo/processed/decision_samples.parquet',
 'reward_concentration.parquet':'data/applications/accountability_economics/reward_concentration.parquet',
 'capital_lock.parquet':'data/applications/accountability_economics/capital_lock.parquet',
 'economic_outcomes.parquet':'data/applications/trustworthy_ai_requirements_audit/observed_economic_consequences.parquet',
 'economic_dictionary.csv':'data/dictionaries/abc_economic_measure_dictionary.csv',
 'old_experiment_manifest.json':'Results/AgenticOracleAuditor/experiment_manifest.json',
}


def read_events(path):
    events, noise = [], []
    for line in path.read_text().splitlines():
        try: events.append(json.loads(line))
        except json.JSONDecodeError: noise.append(line)
    return events, noise


def observed_tools(events):
    outputs=[]
    for event in events:
        item=event.get('item') or {}
        if event.get('type')=='item.completed' and item.get('type')=='mcp_tool_call':
            result=item.get('result')
            payload=None
            if isinstance(result,dict):
                payload=result.get('structured_content',result.get('structuredContent'))
                if payload is None:
                    texts=[c.get('text','') for c in result.get('content',[]) if c.get('type')=='text']
                    try: payload=json.loads('\n'.join(texts))
                    except (ValueError,TypeError): payload='\n'.join(texts)
            outputs.append(dict(tool=item.get('tool'),arguments=item.get('arguments'),
                                content=result,parsed_content=payload,error=item.get('error'),status=item.get('status')))
    return outputs


def prepare(workspace):
    workspace=Path(workspace).resolve(); local=ROOT/'data/local'; local.mkdir(parents=True,exist_ok=True)
    files=[]
    for name, relative in INPUTS.items():
        src=workspace/relative; dst=local/name
        if not src.exists(): raise FileNotFoundError(src)
        if dst.exists() and sha256(dst)!=sha256(src): raise ValueError(f'Input changed: {dst}; use a separately versioned experiment')
        if not dst.exists(): shutil.copy2(src,dst)
        files.append(dict(path=str(dst.relative_to(ROOT)),source=str(src),sha256=sha256(dst),bytes=dst.stat().st_size))
    store=json.loads((local/'evidence_store.json').read_text())
    expected=json.loads((local/'evidence_store_manifest.json').read_text())['evidence_store']['sha256']
    if sha256(local/'evidence_store.json')!=expected: raise ValueError('Evidence store pin mismatch')
    labels=pd.read_parquet(local/'samples.parquet').set_index('sample_id').proposal_rejected_by_protocol.to_dict()
    rows=[]; sources=[]; matched=[]; diagnostic=[]
    for group in ['runs','robustness']:
        for p in sorted((workspace/'Results/AgenticOracleAuditor'/group).rglob('run_meta.json')):
            folder=p.parent; meta=json.loads(p.read_text())
            raw=json.loads((folder/'raw_output.json').read_text())
            validation=json.loads((folder/'validation.json').read_text())
            events,noise=read_events(folder/'events.jsonl'); tools=observed_tools(events)
            usage=[e['usage'] for e in events if e.get('type')=='turn.completed' and 'usage' in e]
            usage=usage[-1] if usage else {}
            case=meta['case_id']; final=meta['final']; ids=raw.get('evidence_ids',[]) if isinstance(raw,dict) else []
            row=dict(case_id=case,repeat=meta['repeat'],system=meta['system'],model_id=meta['model_id'],
                     split=store['cases'][case]['split'],scenario=meta['scenario'],label=int(labels[case]),
                     p_reject=raw.get('p_reject',.5),action=final['action'],raw_action=raw.get('action'),
                     guard_blocked=not validation['valid'],output_valid=meta['raw_output_valid'],
                     required_citation_complete=validation['required_evidence_citation_complete'],
                     citation_valid=validation['evidence_citation_valid'],evidence_ids=json.dumps(ids),
                     guard_errors=json.dumps(validation['errors']),tool_calls=meta['tool_call_count'],
                     tool_successes=meta['tool_call_successes'],cap_reached=meta['tool_call_count']>=8,
                     input_tokens=meta['input_tokens'],output_tokens=meta['output_tokens'],
                     cached_input_tokens=meta['cached_input_tokens'],total_tokens=meta['total_tokens'],
                     latency_seconds=meta['latency_seconds'],api_equivalent_usd=meta['cost_usd'],
                     timed_out=meta['timed_out'],return_code=meta['return_code'],source=str(folder))
            rows.append(row)
            diagnostic.append(dict(case_id=case,system=meta['system'],repeat=meta['repeat'],scenario=meta['scenario'],
                                   non_json_log_lines=len(noise),tool_count_match=len(tools)==meta['tool_call_count'],
                                   token_count_match=all(usage.get(k)==meta[k] for k in ['input_tokens','output_tokens','cached_input_tokens']),
                                   total_identity=meta['input_tokens']+meta['output_tokens']==meta['total_tokens'],
                                   input_includes_cache=meta['cached_input_tokens']<=meta['input_tokens']))
            for name in ['run_meta.json','raw_output.json','validation.json','events.jsonl','prompt.txt']:
                s=folder/name
                sources.append(dict(path=str(s),sha256=sha256(s),bytes=s.stat().st_size))
            if group=='runs' and meta['system']=='A1':
                matched.append(dict(case_id=case,repeat=meta['repeat'],model_id=meta['model_id'],
                                    cutoff_unix=meta['cutoff_unix'],source_events_sha256=sha256(folder/'events.jsonl'),
                                    observed_tool_responses=tools))
    table(local,'historical_runs',pd.DataFrame(rows)); table(local,'historical_log_checks',pd.DataFrame(diagnostic))
    matched_path=local/'historical_matched_evidence.jsonl'
    with matched_path.open('w') as f:
        for row in matched: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    save_json(local/'historical_source_checksums.json',sources)
    for name in ['historical_runs.parquet','historical_log_checks.parquet','historical_matched_evidence.jsonl','historical_source_checksums.json']:
        p=local/name;files.append(dict(path=str(p.relative_to(ROOT)),source='Normalized from checksum-indexed original run records',sha256=sha256(p),bytes=p.stat().st_size))
    save_json(ROOT/'manifests/local-inputs.json',dict(status='LOCAL_INPUTS_PINNED_PUBLIC_REVISION_UNVERIFIED',files=files,
              historical_run_count=len(rows),historical_models=dict(Counter(r['model_id'] for r in rows)),
              selected_new_model='Qwen3.8-27B',public_dataset_revision=None))
    print(json.dumps(dict(imported_files=len(files),historical_runs=len(rows),matched_evidence_packets=len(matched))))
