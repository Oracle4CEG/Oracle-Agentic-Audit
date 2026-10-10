"""Keep infrastructure attempts visible alongside the recovered scientific panel."""
import json
from pathlib import Path
import pandas as pd
from oracle_audit.io import ROOT,table,save_json,sha256
from experiments.infrastructure_recovery import MANIFEST


def audit(out,supplement=None,guards=None):
    from experiments.scope import INTERRUPTIONS, load as load_scope
    if INTERRUPTIONS.exists():
        amendment=json.loads(INTERRUPTIONS.read_text())
        cancelled=pd.DataFrame([dict(active_path=a['active_path'],archive_path=a['archive_path'],
            cause=a['cause'],usage_status=a['usage_status']) for a in amendment['interrupted_attempts']])
        table(out,'scope_interrupted_attempts',cancelled)
        save_json(Path(out)/'scope_reduction_accounting.json',dict(scope_id=load_scope()['scope_id'],
            interrupted_attempts=len(cancelled),
            explanation='User-requested scope reduction stopped old schedulers; unfinished attempts retained separately. Completed extra runs remain in *_runs_outside_scope.csv.',
            costs='Interrupted requests without returned usage have unknown token/compute cost, not zero.'))
    if not MANIFEST.exists():return
    receipt=json.loads(MANIFEST.read_text());rows=[]
    for config in receipt['configurations']:
        if sha256(ROOT/config['archive_path'])!=config['sha256']:raise ValueError('Modified archived execution configuration')
    for attempt in receipt['attempts']:
        folder=ROOT/attempt['archive_path']
        for file in attempt['files']:
            if sha256(folder/file['path'])!=file['sha256']:raise ValueError('Modified archived infrastructure attempt')
        original=json.loads((folder/'run_meta.json').read_text()) if (folder/'run_meta.json').exists() else {}
        active_path=ROOT/attempt['active_path']/'run_meta.json'
        active=json.loads(active_path.read_text()) if active_path.exists() else {}
        rows.append(dict(active_path=attempt['active_path'],archive_path=attempt['archive_path'],cause=attempt['cause'],
          original_completion_record=bool(original),original_return_code=original.get('return_code'),
          original_observed_input_tokens=original.get('input_tokens'),original_observed_output_tokens=original.get('output_tokens'),
          original_observed_total_tokens=original.get('total_tokens'),original_recorded_latency_seconds=original.get('latency_seconds'),
          recovery_completion_record=bool(active),recovery_return_code=active.get('return_code'),
          recovery_output_schema_valid=active.get('raw_output_valid')))
    frame=pd.DataFrame(rows);table(out,'infrastructure_attempt_audit',frame)
    summary=frame.groupby('cause').agg(attempts=('active_path','size'),
              original_completion_records=('original_completion_record','sum'),
              recovery_completion_records=('recovery_completion_record','sum')).reset_index()
    table(out,'infrastructure_attempt_summary',summary)
    frames=[f for f in [supplement,guards] if f is not None and len(f)]
    if frames:
        active=pd.concat(frames,ignore_index=True)
        table(out,'serving_runtime_cohorts',active.groupby(['suite','system','serving_revision']).agg(
            case_runs=('case_id','size'),unique_cases=('case_id','nunique'),
            recovered_case_runs=('infrastructure_recovery','sum'),mean_latency_seconds=('latency_seconds','mean'),
            mean_observed_total_tokens=('total_tokens','mean')).reset_index())
    save_json(Path(out)/'infrastructure_scope.json',dict(
        active_panel='Successful original responses plus explicitly recorded infrastructure recoveries. This is not a first-attempt-only analysis.',
        archived_attempts=len(frame),selection=receipt['selection'],
        guard_panel_amendment=receipt.get('guard_panel_amendment'),
        timeout_attribution=receipt.get('timeout_attribution'),
        costs='Primary cost tables cover active panel requests. Archived observed usage is additional; actual compute for requests without returned usage is unknown. These tables are not total end-to-end resource consumption.',
        limitations='Recovery changes execution time and batching. Generation is not guaranteed bitwise reproducible. Main-supplement normal-return model/schema errors remain; the entire guard panel is restarted to keep all conditions under the same runtime.'))
