"""Full paired corruption experiment, distinct from replaying final enforcement."""
import json
import time
import os
import threading
import signal
from datetime import datetime,timezone
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor,as_completed
import pandas as pd
from oracle_audit.io import ROOT,save_json,sha256
from models.run_required import ref
from guards.fixtures import apply_fixture,packet,SCENARIOS
from policies.decision import registered
from experiments.infrastructure_recovery import provenance,MANIFEST as RECOVERY_MANIFEST
from experiments.scope import load as load_scope, provenance as scope_provenance, MANIFEST as SCOPE_MANIFEST

OUT=ROOT/'outputs/guard_experiment'
TRANSPORT_STOP=threading.Event()
SCHEDULING_STOP=threading.Event()
ref.MCP_SERVER=ROOT/'guards/experimental_mcp_server.py'


def run_one(case,system,scenario,mode):
    if SCHEDULING_STOP.is_set():return 'skipped_after_requested_stop'
    if TRANSPORT_STOP.is_set():return 'skipped_after_transport_failure'
    folder=OUT/'runs'/scenario/mode/system/case.removeprefix('0x');folder.mkdir(parents=True,exist_ok=True)
    if (folder/'run_meta.json').exists():return 'cached'
    store,_=ref.load_store();fixture=apply_fixture(store,case,scenario);cutoff=store.cases[case]['cutoff_unix']
    pack=packet(store,case,scenario,mode=='on') if system=='B1' else None
    messages=ref.initial_messages(system,pack,case,cutoff)
    save_json(folder/'input_messages.json',messages);save_json(folder/'fixture.json',fixture)
    (folder/'prompt.txt').write_text(ref.render_prompt(messages))
    start=time.perf_counter();started_utc=datetime.now(timezone.utc).isoformat()
    text,events,code,timed_out,error=ref.run_inference(system,0,messages,case,cutoff,ref.STORE,store.store_sha256,scenario+':'+mode,600)
    elapsed=time.perf_counter()-start
    (folder/'events.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in events))
    (folder/'raw_output.txt').write_text(text or '')
    raw=ref.parse_json_object(text);info=ref.event_metrics(events)
    validation,diagnostics=ref.validation_for_run(store,case,raw,system,scenario,pack,info)
    if pack and pack.get('admission_errors'):
        errors=list(dict.fromkeys(validation.errors+pack['admission_errors']))
        validation=ref.ValidationResult(False,validation.evidence_ids,errors,validation.warnings,validation.max_evidence_time_unix,validation.poison_detected)
        diagnostics.update(validation.as_dict())
    guarded=ref.enforce_safety(raw or {},validation)
    final=guarded.copy()
    if mode=='all_off' and ref.valid_raw_output(raw):
        final['action']=str(registered([raw['p_reject']])[0]);final['human_review']=final['action'] in ['Abstain','Investigate']
    save_json(folder/'raw_output.json',raw);save_json(folder/'validation.json',diagnostics);save_json(folder/'final_output.json',final)
    all_payloads=[pack,info['observed_payloads']]
    future=ref.post_cutoff_evidence_ids(cutoff,*all_payloads)
    save_json(folder/'run_meta.json',dict(case_id=case,repeat=0,system=system,scenario=scenario,guard_mode=mode,model_id='Qwen3.8-27B',
       final=final,guarded_counterfactual=guarded,raw_output_valid=ref.valid_raw_output(raw),guard_blocked=not validation.valid,
       raw_safe_abstention=ref.safe_abstention(raw),future_evidence_exposed=bool(future),future_ids_cited=bool(future&set(ref.raw_evidence_ids(raw))),
       tool_call_count=info['tool_call_count'],tool_call_successes=info['tool_call_successes'],
       started_utc=started_utc,temperature=ref.TEMPERATURE,request_seed=ref.request_seed(0),
       top_p=ref.TOP_P,top_k=ref.TOP_K,max_completion_tokens=ref.MAX_COMPLETION_TOKENS,
       evidence_store_sha256=store.store_sha256,reasoning_boundary_violation=diagnostics.get('reasoning_boundary_violation',False),
       cost_usd=None,cost_status='unmeasured_self_hosted_resource_cost',
       latency_seconds=elapsed,return_code=code,timed_out=timed_out,inference_error=error,
       **info['usage'],**provenance(folder),**scope_provenance(folder)))
    if code!=0 or timed_out:TRANSPORT_STOP.set()
    return 'ok' if code==0 else 'failed'


def main():
    global OUT
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--pilot',action='store_true');args=p.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:SCHEDULING_STOP.set())
    spec=load_scope()['guards']
    store,_=ref.load_store();panel=store.data['cohort']['robustness_panel_case_ids']
    if args.pilot:
        panel=panel[:1];OUT=ROOT/'outputs/guard_pilot'
    response=requests.get(ref.API_BASE_URL+'/models',timeout=10);response.raise_for_status()
    if 'Qwen3.8-27B' not in [m['id'] for m in response.json()['data']]:raise RuntimeError('Wrong guard experiment model')
    config=dict(endpoint=ref.endpoint_without_secrets(ref.API_BASE_URL),workers=args.workers,
                scope_manifest_sha256=sha256(SCOPE_MANIFEST),scope_module_sha256=sha256(ROOT/'experiments/scope.py'),
                failure_policy='Retain every response; stop scheduling after a transport failure. Any infrastructure recovery is separately archived and disclosed.',
                recovery_manifest_sha256=sha256(RECOVERY_MANIFEST) if RECOVERY_MANIFEST.exists() else None,
                runner_sha256=sha256(Path(__file__)),fixtures_sha256=sha256(ROOT/'guards/fixtures.py'),
                mcp_server_sha256=sha256(ref.MCP_SERVER),reference_runner_sha256=sha256(Path(ref.__file__)))
    config_path=OUT/'execution_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:raise RuntimeError('Changed guard configuration; use separately versioned output root')
    save_json(config_path,config)
    tasks=[(case,system,scenario,mode) for scenario in spec['scenarios'] for case in panel for system in spec['systems'] for mode in spec['modes']]
    save_json(OUT/'execution_plan.json',dict(cases=panel,scenarios=spec['scenarios'],modes=spec['modes'],systems=spec['systems'],repeats=1,
             seed=ref.request_seed(0),case_runs=len(tasks),scope='Full admission and validation ablation; no actual on-chain action'))
    done=0;failures=0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(run_one,*task) for task in tasks]
        for future in as_completed(jobs):
            status=future.result();done+=1;failures+=status=='failed'
            if done%10==0 or done==len(tasks):print(json.dumps(dict(completed=done,total=len(tasks),transport_failures=failures)),flush=True)
    if TRANSPORT_STOP.is_set():raise RuntimeError('Transport failure: pending tasks stopped; completed attempts retained. Inspect infrastructure before resuming.')
    if SCHEDULING_STOP.is_set():raise SystemExit(75)


if __name__=='__main__':main()
