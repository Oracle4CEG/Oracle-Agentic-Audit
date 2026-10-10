"""Resumable Qwen supplementation; saved raw responses are the scientific record."""
import argparse
import json
import os
import time
import hashlib
import threading
import signal
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import requests
from oracle_audit.io import ROOT,save_json,sha256
from models.reference import run_agentic_oracle_auditor as ref
from experiments.qwen_records import ORIGINAL,matched_evidence
from experiments.infrastructure_recovery import provenance,MANIFEST as RECOVERY_MANIFEST
from experiments.scope import load as load_scope, provenance as scope_provenance, MANIFEST as SCOPE_MANIFEST

RUNS=ROOT/'outputs/qwen_supplement'
TRANSPORT_STOP=threading.Event()
SCHEDULING_STOP=threading.Event()
ref.STORE=ROOT/'data/local/evidence_store.json'
ref.STORE_MANIFEST=ROOT/'data/local/evidence_store_manifest.json'
ref.API_BASE_URL=os.environ.get('OPENAI_BASE_URL','http://127.0.0.1:33001/v1')
ref.API_KEY=os.environ.get('OPENAI_API_KEY','EMPTY')


def path_for(suite,system,repeat,case):
    return RUNS/suite/system/f'repeat_{repeat}'/case.removeprefix('0x')


def evidence_context(folder):
    return matched_evidence(folder)


def run_one(suite,system,repeat,case,temperature,timeout=600):
    if SCHEDULING_STOP.is_set():return 'skipped_after_requested_stop',system,case
    if TRANSPORT_STOP.is_set():return 'skipped_after_transport_failure',system,case
    out=path_for(suite,system,repeat,case);out.mkdir(parents=True,exist_ok=True)
    if (out/'run_meta.json').exists():return 'cached',system,case
    store,_=ref.load_store();cutoff=store.cases[case]['cutoff_unix']
    packet=None;source=None
    if system=='B2':
        source=(ORIGINAL/'runs/A1'/f'repeat_{repeat}'/case.removeprefix('0x')) if suite=='matched_test' else path_for('validation','A1',repeat,case)
        if not (source/'run_meta.json').exists():raise FileNotFoundError(f'A1 must complete before B2: {source}')
        context=[{'tool':item.get('name'),'content':item['content']} for item in evidence_context(source)]
        messages=[{'role':'system','content':ref.B1_INSTRUCTION},
                  {'role':'user','content':json.dumps(dict(case_id=case,cutoff_unix=cutoff,
                   evidence_delivery='The following tool responses are the exact evidence received by the paired auditor. No additional retrieval is permitted.',
                   observed_tool_responses=context),ensure_ascii=False)}]
        backend='B1'
    elif system=='B1':
        packet=ref.clean_packet(store,case);messages=ref.initial_messages('B1',packet,case,cutoff);backend='B1'
    else:
        messages=ref.initial_messages('A1',None,case,cutoff);backend='A1'
    save_json(out/'input_messages.json',messages)
    (out/'prompt.txt').write_text(ref.render_prompt(messages))
    started=time.perf_counter();start_utc=datetime.now(timezone.utc).isoformat()
    raw_text,events,code,timed_out,error=ref.run_inference(backend,repeat,messages,case,cutoff,ref.STORE,store.store_sha256,'clean',timeout)
    elapsed=time.perf_counter()-started
    (out/'events.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in events))
    (out/'raw_output.txt').write_text(raw_text or '')
    raw=ref.parse_json_object(raw_text);info=ref.event_metrics(events)
    validation,diagnostics=ref.validation_for_run(store,case,raw,backend,'clean',packet,info)
    if system=='B2':
        # Replay has no agent's private reasoning, proposed answer or model probability.
        # Evidence IDs in the proposal index alone do not count as acquired content.
        source_events,_=__import__('experiments.prepare',fromlist=['read_events']).read_events(source/'events.jsonl')
        source_info=ref.event_metrics(source_events)
        available=set(source_info['retrieved_evidence_ids'])
        missing=set(store.cases[case]['required_evidence_ids'])-available
        if missing:
            errors=list(validation.errors)+['MATCHED_REQUIRED_CONTENT_UNAVAILABLE:'+','.join(sorted(missing))]
            validation=ref.ValidationResult(False,validation.evidence_ids,errors,validation.warnings,validation.max_evidence_time_unix,validation.poison_detected)
            diagnostics.update(validation.as_dict())
        diagnostics['matched_source_events_sha256']=sha256(source/'events.jsonl')
        diagnostics['matched_context_sha256']=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    final=ref.enforce_safety(raw or {},validation)
    usage=info['usage']
    meta=dict(case_id=case,repeat=repeat,system=system,scenario='clean',suite=suite,split=store.cases[case]['split'],
              model_id='Qwen3.8-27B',temperature=temperature,request_seed=ref.request_seed(repeat),
              top_p=ref.TOP_P,top_k=ref.TOP_K,max_completion_tokens=ref.MAX_COMPLETION_TOKENS,
              enable_thinking=ref.ENABLE_THINKING,preserve_thinking=ref.PRESERVE_THINKING,
              evidence_store_sha256=store.store_sha256,raw_output_valid=ref.valid_raw_output(raw),
              final=final,return_code=code,timed_out=timed_out,inference_error=error,
              started_utc=start_utc,latency_seconds=elapsed,tool_call_count=info['tool_call_count'],
              tool_call_successes=info['tool_call_successes'],**usage,
              cost_usd=None,cost_status='unmeasured_self_hosted_resource_cost',
              reasoning_boundary_violation=diagnostics.get('reasoning_boundary_violation',False),
              source_a1_events_sha256=sha256(source/'events.jsonl') if source else None,
              **provenance(out),**scope_provenance(out))
    save_json(out/'raw_output.json',raw);save_json(out/'validation.json',diagnostics);save_json(out/'final_output.json',final)
    save_json(out/'run_meta.json',meta)
    if code!=0 or timed_out:TRANSPORT_STOP.set()
    return 'ok' if code==0 else 'failed',system,case


def run_batch(tasks,suite,temperature,workers):
    ref.TEMPERATURE=temperature
    total=len(tasks);completed=0;failures=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(run_one,suite,system,repeat,case,temperature) for system,repeat,case in tasks]
        for future in as_completed(jobs):
            status,system,case=future.result();completed+=1;failures+=status=='failed'
            if completed%10==0 or status=='failed' or completed==total:
                print(json.dumps(dict(suite=suite,completed=completed,total=total,transport_failures=failures,last_system=system,status=status)),flush=True)
    if TRANSPORT_STOP.is_set():raise RuntimeError('Transport failure: pending tasks stopped; completed attempts retained. Inspect infrastructure before resuming.')
    if SCHEDULING_STOP.is_set():raise SystemExit(75)
    return failures


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',choices=['pilot','validation','matched_test','low_temperature','all'],default='all')
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:SCHEDULING_STOP.set())
    plan=load_scope()
    if args.suite=='low_temperature' and not plan['low_temperature']['required']:
        raise ValueError('Low-temperature inference was cancelled by the user scope amendment')
    r=requests.get(ref.API_BASE_URL+'/models',timeout=10);r.raise_for_status()
    if 'Qwen3.8-27B' not in [m['id'] for m in r.json()['data']]:raise RuntimeError('Wrong model service')
    store,_=ref.load_store();test=[c for c,v in sorted(store.cases.items()) if v['split']=='test'];val=[c for c,v in sorted(store.cases.items()) if v['split']=='validation']
    RUNS.mkdir(parents=True,exist_ok=True)
    config=dict(endpoint=ref.endpoint_without_secrets(ref.API_BASE_URL),workers=args.workers,
                model='Qwen3.8-27B',validation_repeats=1,test_matched_repeat_ids=plan['matched_test']['repeats'],low_temperature_repeats=0,
                scope_manifest_sha256=sha256(SCOPE_MANIFEST),scope_module_sha256=sha256(ROOT/'experiments/scope.py'),
                failure_policy='Retain every response; stop scheduling after a transport failure. Any infrastructure recovery is separately archived and disclosed.',
                recovery_manifest_sha256=sha256(RECOVERY_MANIFEST) if RECOVERY_MANIFEST.exists() else None,
                original_qwen_manifest_sha256=sha256(ORIGINAL/'experiment_manifest.json'),
                runner_sha256=sha256(Path(__file__)),reference_runner_sha256=sha256(Path(ref.__file__)))
    config_path=RUNS/f'execution_config_{args.suite}.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise RuntimeError('Changed execution configuration; use separately versioned output root')
    save_json(config_path,config)
    if args.suite=='pilot':run_batch([('B1',0,val[0]),('A1',0,val[0])],'pilot',1.,args.workers);return
    if args.suite in ['validation','all']:
        run_batch([(s,0,c) for c in val for s in ['B1','A1']],'validation',1.,args.workers)
        run_batch([('B2',0,c) for c in val],'validation',1.,args.workers)
    if args.suite in ['matched_test','all']:
        run_batch([('B2',i,c) for i in plan['matched_test']['repeats'] for c in test],'matched_test',1.,args.workers)
    if args.suite in ['low_temperature','all'] and plan['low_temperature']['required']:
        run_batch([(s,0,c) for c in test for s in ['B1','A1']],'low_temperature',0.,args.workers)


if __name__=='__main__':main()
