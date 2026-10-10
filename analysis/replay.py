"""Offline regeneration. Completeness is a checked condition, never a filled-in claim."""
import argparse
import json
import platform
import time
from pathlib import Path
import pandas as pd
from oracle_audit.io import ROOT,verify_inputs,save_json,table,sha256
from experiments.qwen_records import import_original
from models.tabular import run as tabular
from analysis import audits,evaluate,economics,citations,boundary,matching,evidence_lineage,cohort_lineage,supplementary,negative_controls
from analysis.collect import collect
from analysis.metrics import paired_bootstrap
from experiments.scope import required_complete, audit as audit_scope, MANIFEST as SCOPE_MANIFEST


def run(out,partial=False,bootstrap=True,retrain=False):
    started=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plan=audit_scope()
    verify_inputs();original=import_original(out/'normalized')
    supplement,guards,status=collect(out,require_complete=not partial)
    from analysis.infrastructure import audit as infrastructure_audit
    infrastructure_audit(out,supplement,guards)
    source=ROOT/'outputs/tabular/predictions.parquet'
    fresh_baseline=retrain or not source.exists()
    base=tabular(out/'tabular') if fresh_baseline else pd.read_parquet(source)
    main=original[original.scenario.eq('clean')].copy()
    calibration_ready=status['validation']['complete']
    if calibration_ready:main=pd.concat([main,supplement[supplement.suite.eq('validation')]],ignore_index=True)
    if status['matched_test']['complete']:main=pd.concat([main,supplement[supplement.suite.eq('matched_test')]],ignore_index=True)
    llm=evaluate.calibrated_llm(main,out)
    predictions=pd.concat([base,llm],ignore_index=True,sort=False)
    table(out,'all_predictions',predictions)
    evaluate.policies(predictions,out,bootstrap=bootstrap)
    cohort=audits.cohort(out);audits.log_audits(main[main.split.eq('test')],out)
    audits.actor_groups(predictions,cohort,out)
    supplementary.capacity(predictions,out)
    supplementary.diagnostics(main,out,bootstrap=bootstrap)
    if bootstrap:supplementary.group_intervals(predictions,cohort,out)
    citations.analyze(main[main.split.eq('test')],out)
    review=boundary.extract(original,out);boundary.reviewed_summary(review,out)
    matching.audit(main[main.system.eq('A1')],main[main.system.eq('B2')],out)
    evidence_lineage.audit(out)
    cohort_lineage.audit(out)
    negative_controls.future_fields(out)
    negative_controls.shuffled_labels(out,model_path=(out/'tabular' if fresh_baseline else ROOT/'outputs/tabular')/'lightgbm.joblib')
    economics.run(out/'economics')
    if status['guards']['complete']:
        from analysis.guard_results import analyze
        from analysis.guard_delivery import audit as audit_delivery
        analyze(audit_delivery(guards,out),out,bootstrap=bootstrap)
    if status['low_temperature']['complete']:
        low=supplement[supplement.suite.eq('low_temperature')].copy();low['calibration']='raw'
        evaluate.policies(low,out/'low_temperature',bootstrap=False)
        if bootstrap:
            cis=[]
            for system in ['B1','A1']:
                high=original[(original.system==system)&(original.scenario=='clean')&(original.repeat==0)]
                ci=paired_bootstrap(high,low[low.system.eq(system)]).assign(system=system,contrast='temperature 0 minus 1; repeat 0')
                cis.append(ci)
            table(out/'low_temperature','paired_temperature_bootstrap',pd.concat(cis,ignore_index=True))
    from figures.render import render
    render(predictions,out,out/'economics')
    from figures.tables import render as render_tables
    render_tables(out)
    save_json(out/'replay_receipt.json',dict(
        status='COMPLETE_OFFLINE_REPLAY' if required_complete(status,successful=True) else 'PARTIAL_ANALYSIS_ONLY',
        scope_id=plan['scope_id'],scope_manifest_sha256=sha256(SCOPE_MANIFEST),
        cancelled_optional_experiments=['low_temperature'],
        python=platform.python_version(),platform=platform.platform(),elapsed_seconds=time.perf_counter()-started,
        original_main_case_runs=int(original.scenario.eq('clean').sum()),
        completion=status,bootstrap_resamples=2000 if bootstrap else 0,
        input_manifest_sha256=sha256(ROOT/'manifests/local-inputs.json'),
        lock_sha256=sha256(ROOT/'requirements.lock'),
        incomplete_panels_excluded_from_reported_metrics=True,hosted_colab='NOT_RUN',
        inference_attempt_policy='Recovered fixed-case panel; archived timeout/interruption history reported separately' if (ROOT/'manifests/infrastructure-recovery-20261009.json').exists() else 'Recorded first attempts',
        public_dataset_revision=(json.loads((ROOT/'manifests/public-release.json').read_text())['dataset_revision']
                                 if (ROOT/'manifests/public-release.json').exists() else 'UNVERIFIED'),
        training_cutoff='UNVERIFIED'))
    from analysis.report import write
    write(out)
    print(f'Offline analysis written to {out}',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'outputs/final')
    p.add_argument('--partial',action='store_true');p.add_argument('--no-bootstrap',action='store_true');p.add_argument('--retrain',action='store_true')
    args=p.parse_args();run(args.out,partial=args.partial,bootstrap=not args.no_bootstrap,retrain=args.retrain)


if __name__=='__main__':main()
