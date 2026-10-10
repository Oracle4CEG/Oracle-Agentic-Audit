"""Record checked local capabilities without claiming a public release."""
import json
from oracle_audit.io import ROOT,save_json,sha256


def update(analysis_dir=None):
    analysis_dir=analysis_dir or ROOT/'outputs/progress'
    p=ROOT/'manifests/research-inputs.json';manifest=json.loads(p.read_text())
    pins=json.loads((ROOT/'manifests/local-inputs.json').read_text())['files']
    for name in ['economic-source-inputs.json','evidence-lineage-inputs.json','cohort-lineage-inputs.json']:
        pins+=json.loads((ROOT/'manifests'/name).read_text())
    unique={v['path']:v for v in pins}
    manifest.update(status='LOCAL_EXPERIMENTS_IMPLEMENTED_PUBLIC_RELEASE_UNVERIFIED',
        input_files=list(unique.values()),
        model_specification=dict(model='Qwen3.8-27B',training_cutoff=None,
             local_model_card_title='Qwen3.8-27B',architecture='Qwen3_5ForConditionalGeneration',
             config_model_type='qwen3_5',text_dtype='bfloat16',
             source_revision_metadata='Revision:master,CreatedAt:1786731894',
             immutable_public_checkpoint_revision=None,
             checkpoint_hash_manifest='manifests/model-snapshot-verification.json',
             checkpoint_receipt_sha256=sha256(ROOT/'manifests/model-snapshot-verification.json'),
             main_serving='manifests/supplement-server-runtime.json',guard_serving='manifests/guard-server-runtime.json',
             sampling='experiments/config.json',quantization=None),
        replay_command='python -m analysis.replay --retrain',
        inference_command=['OPENAI_BASE_URL=http://127.0.0.1:33001/v1 python -m models.run_required --suite all --workers 8',
                           'OPENAI_BASE_URL=http://127.0.0.1:33002/v1 python -m guards.run_experiment --workers 4'],
        inference_scope='User-reduced necessary-only scope: 483 validation + 320 existing B2 (repeat IDs 3,4) + 480 A1-only guard. Low temperature cancelled. Original 1600 main + 320 robustness are frozen inputs.',
        scope_amendment=dict(path='manifests/experiment-scope-20261009.json',sha256=sha256(ROOT/'manifests/experiment-scope-20261009.json')),
        original_model_conflict='Legacy local GPT-5.6-sol logs are excluded from Qwen comparisons.')
    recovery=ROOT/'manifests/infrastructure-recovery-20261009.json'
    if recovery.exists():
        manifest['infrastructure_recovery']=dict(receipt=str(recovery.relative_to(ROOT)),sha256=sha256(recovery),
          analysis='Recovered active panel with preserved timeout/interruption history; not first-attempt-only performance.',
          accounting='See infrastructure_attempt_audit.csv; unreported compute is unknown.')
    save_json(p,manifest)
    index=json.loads((ROOT/'manifests/result-index.json').read_text())
    code={
        'R01':['policies/decision.py','policies/calibration.py','analysis/evaluate.py','analysis/metrics.py'],
        'R02':['analysis/citations.py','analysis/matching.py','analysis/audits.py'],
        'R03':['models/tabular.py','models/run_required.py','analysis/collect.py'],
        'R04':['guards/fixtures.py','guards/experimental_mcp_server.py','guards/run_experiment.py','analysis/guard_delivery.py','analysis/guard_results.py','analysis/boundary.py'],
        'R05':['analysis/audits.py','experiments/qwen_records.py'],
        'R06':['analysis/economics.py']}
    outputs={'R01':['policy_metrics.csv','paired_bootstrap.csv','cost_sensitivity.csv'],
             'R02':['citation_error_rates.csv','citation_taxonomy_counts.csv','b2_exact_delivery_checks.csv'],
             'R03':['policy_metrics.csv','matched_evidence_comparison_metrics.csv'],
             'R04':['guard_scenario_metrics.csv','guard_paired_bootstrap.csv','guard_delivery_checks.csv','reasoning_guard_review_summary.csv'],
             'R05':['cost_accounting.csv','tool_budget_distribution.csv'],
             'R06':['economics/verification_activity_concentration.csv','economics/incentives_and_realized_outcomes.csv','economics/cross_protocol_reward_concentration.csv','economics/protocol_variable_availability.csv']}
    ready=(analysis_dir/'replay_receipt.json').exists() and json.loads((analysis_dir/'replay_receipt.json').read_text())['status']=='COMPLETE_OFFLINE_REPLAY'
    for record in index:
        key=record['result_id']
        if key=='R03':
            record['reported_result_or_boundary']='Matched-evidence two-repeat comparison and stronger feature baseline; optional lower-temperature evaluation cancelled'
        record.update(original_sources='Checksum-pinned curated partitions and archived model outputs; see research-inputs.json and original Qwen artifact index.',
            query_code='Descriptor acquisition receipts are upstream; the experimental cohort and raw price snapshots are locally pinned.',
            queried_data='data/local; local SHA-256 revision, public dataset revision unverified',
            process_code=['experiments/prepare.py','experiments/qwen_records.py','analysis/cohort_lineage.py','analysis/evidence_lineage.py'],
            processed_data=['data/local/samples.parquet','data/local/evidence_store.json','data/local/qwen_original_runs.parquet'],
            analysis_code=code[key],result_outputs=[str((analysis_dir/x).relative_to(ROOT)) for x in outputs[key]],
            commands=['python -m analysis.replay --retrain'],
            evidence_status='DERIVED_LOCAL' if ready or key in ['R05','R06'] else 'IN_PROGRESS',
            interpretation_boundary='Observed local results with source checks; consult RESULTS_zh.md for each estimand and limitation. Public availability, hosted Colab and independent scientific review are separate.',
            data_revision='SHA-256 manifests/local-inputs.json; public revision not verified')
    save_json(ROOT/'manifests/result-index.json',index)


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--analysis-dir',type=Path)
    args=parser.parse_args()
    update(args.analysis_dir.resolve() if args.analysis_dir else None)
