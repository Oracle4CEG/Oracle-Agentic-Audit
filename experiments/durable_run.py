"""Detached orchestration; no SSH-session lifetime or automatic inference retry."""
import argparse
import json
import os
import fcntl
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone
from analysis.collect import collect
from oracle_audit.io import ROOT,save_json
from experiments.scope import required_complete, load as load_scope, audit as audit_scope


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--env-file',required=True);parser.add_argument('--round',type=int,default=1)
    parser.add_argument('--run-label');args=parser.parse_args()
    label=args.run_label or f'recovery{args.round}'
    plan=audit_scope()
    output=ROOT/'outputs/completion';output.mkdir(parents=True,exist_ok=True)
    lock=(output/'durable_run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    python=str(ROOT/'.venv/bin/python')
    specs=[('main','http://127.0.0.1:33001/v1',['-u','-m','models.run_required','--suite','all','--workers','8'],
            ROOT/f'outputs/qwen_supplement/{label}_run.log'),
           ('guards','http://127.0.0.1:33002/v1',['-u','-m','guards.run_experiment','--workers','4'],
            ROOT/f'outputs/guard_experiment_{label}.log')]
    children={};streams=[]
    for name,endpoint,command,log in specs:
        stream=log.open('a');streams.append(stream)
        env=os.environ.copy();env['OPENAI_BASE_URL']=endpoint
        env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        env['ORACLE_SERVING_REVISION']='cuda_graph_cpu_threads_1' if args.round>=2 else 'initial_no_cuda_graph'
        children[name]=subprocess.Popen([python,*command],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                       stdout=stream,stderr=subprocess.STDOUT)
    save_json(output/'active_processes.json',dict(supervisor_pid=os.getpid(),session_id=os.getsid(0),scope_id=plan['scope_id'],
               processes={name:child.pid for name,child in children.items()},
               launch='Detached process with stdin closed and file logs; no auxiliary overlapping batches.',
               policy='No automatic retries. A transport failure stops further scheduling in that job.'))
    previous=None
    while True:
        _,_,panels=collect(output/'collection')
        codes={name:child.poll() for name,child in children.items()}
        failed={name:code for name,code in codes.items() if code is not None and code!=0}
        status=dict(stage='NEEDS_ATTENTION' if failed else 'INFERENCE_RUNNING_NECESSARY_ONLY',scope_id=plan['scope_id'],
                    updated_utc=datetime.now(timezone.utc).isoformat(),panels=panels,process_exit_codes=codes,
                    recovery_receipt='manifests/infrastructure-recovery-20261009.json',
                    recovery_round=args.round,
                    archived_attempts_excluded_from_active_panel_but_retained=True)
        save_json(output/'status.json',status)
        current={k:v['completed'] for k,v in panels.items()}
        if current!=previous:print(json.dumps(status),flush=True);previous=current
        if all(code is not None for code in codes.values()):break
        time.sleep(30)
    if failed or not required_complete(panels,successful=True):
        raise RuntimeError('Recovery stopped before a complete transport-successful panel; records preserved.')
    # Offline analysis does not require the GPU services. Stop only task-owned PIDs.
    try:
        from oracle_audit.remote import run
        import shlex
        owned=json.loads((ROOT/'manifests/owned-experiment-services.json').read_text())['services']
        script='''import json,hashlib,os,signal
from pathlib import Path
rows=OWNED
actions=[]
for row in rows:
 p=Path(f"/proc/{row['pid']}")
 if not p.exists():
  actions.append(dict(pid=row['pid'],status='already_exited'));continue
 if row['port'] not in (30001,30002):raise RuntimeError('Not a task-owned port')
 cmd=p.joinpath('cmdline').read_bytes()
 if p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]!=row['start_ticks'] or hashlib.sha256(cmd).hexdigest()!=row['command_sha256']:
  raise RuntimeError('Process ownership changed; refusing shutdown')
 os.kill(row['pid'],signal.SIGTERM)
 actions.append(dict(pid=row['pid'],port=row['port'],status='SIGTERM_SENT'))
print(json.dumps(actions))
'''.replace('OWNED',repr(owned))
        actions=json.loads(run('python3 -c '+shlex.quote(script),args.env_file))
        save_json(output/'owned_services_shutdown.json',dict(actions=actions,original_port_30000_untouched=True))
    except Exception as error:
        save_json(output/'owned_services_shutdown.json',dict(status='NEEDS_ATTENTION',error=str(error)))
    with (ROOT/f'outputs/finish_required_{label}.log').open('w') as stream:
        subprocess.run([python,'-u','-m','experiments.finish_required'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
