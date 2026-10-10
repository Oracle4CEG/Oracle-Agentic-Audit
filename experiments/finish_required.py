"""Finish the authorized experiment deliverables when recorded inference is complete."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from analysis.collect import collect
from oracle_audit.io import ROOT, save_json, sha256
from experiments.scope import required_complete, load as load_scope


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch-pid',type=int,action='append',default=[])
    args=parser.parse_args()
    output=ROOT/'outputs/completion';output.mkdir(parents=True,exist_ok=True)
    def status(stage,**kwargs):
        save_json(output/'status.json',dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs))
        print(stage,flush=True)
    def command(name,argv):
        status(name,command=argv)
        with (output/(name+'.log')).open('w') as stream:
            subprocess.run(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    python=str(ROOT/'.venv/bin/python');clean=str(ROOT/'.venv-replay/bin/python')
    previous=None
    try:
        while True:
            _,_,panels=collect(output/'collection')
            if required_complete(panels,successful=True):break
            current={k:v['completed'] for k,v in panels.items()}
            if current!=previous:
                status('INFERENCE_RUNNING',panels=panels);previous=current
            if args.watch_pid and not any(Path(f'/proc/{pid}').exists() for pid in args.watch_pid):
                raise RuntimeError('All watched inference processes exited with incomplete panels; no automatic replacement attempted.')
            time.sleep(30)
        command('scientific_tests',[python,'-m','unittest','discover','-s','tests'])
        command('final_replay',[python,'-u','-m','analysis.replay','--retrain'])
        command('isolated_replay',[clean,'-u','-m','analysis.replay','--retrain','--out','outputs/clean_replay'])
        command('compare_replays',[clean,'-m','analysis.verify_replay','--reference','outputs/final','--candidate','outputs/clean_replay'])
        receipt=json.loads((ROOT/'outputs/clean_replay/comparison_receipt.json').read_text())
        if receipt['status']!='PASS':raise ValueError('Complete independent replay has not passed')
        save_json(ROOT/'manifests/clean_replay_receipt.json',receipt)
        command('update_manifest',[python,'-m','experiments.update_manifest','--analysis-dir','outputs/final'])
        p=ROOT/'manifests/research-inputs.json';manifest=json.loads(p.read_text())
        manifest['reference_outputs']=[dict(path=str(f.relative_to(ROOT)),sha256=sha256(f))
          for f in sorted((ROOT/'outputs/final').rglob('*')) if f.suffix in ['.csv','.png','.pdf']]
        save_json(p,manifest)
        command('package',[python,'-m','experiments.package'])
        command('verify_package',[python,'-m','experiments.verify_bundle','--archive',
                                 'outputs/delivery/oracle-camera-ready-20261009.tar.gz'])
        status('COMPLETE_LOCAL_EXPERIMENTS',panels=panels,
               scope_id=load_scope()['scope_id'],optional_low_temperature='CANCELLED_NOT_RUN',
               report='outputs/final/RESULTS_zh.md',bundle='outputs/delivery/oracle-camera-ready-20261009.tar.gz',
               isolated_replay=receipt['status'],compared_tables=receipt['compared_tables'],
               hosted_colab='NOT_RUN',public_release='UNVERIFIED')
    except Exception as error:
        status('NEEDS_ATTENTION',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':main()
