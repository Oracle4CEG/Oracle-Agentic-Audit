"""Validate completed panels before using additional inference in paper analyses."""
import json
import time
from pathlib import Path
import pandas as pd
from oracle_audit.io import ROOT,table,save_json
from experiments.qwen_records import row_from_folder
from experiments.scope import load as load_scope, select, required_complete


def read_runs(root,guard=False):
    store=json.loads((ROOT/'data/local/evidence_store.json').read_text())
    labels=pd.read_parquet(ROOT/'data/local/samples.parquet').set_index('sample_id').proposal_rejected_by_protocol.to_dict()
    rows=[]
    for path in sorted(Path(root).rglob('run_meta.json')):
        try:
            meta=json.loads(path.read_text())
        except json.JSONDecodeError:
            # Completion markers from an active writer can be observed during write.
            # Never accept them as complete; older malformed records remain errors.
            if time.time()-path.stat().st_mtime < 10:
                continue
            raise
        row=row_from_folder(path.parent,labels,store)
        row['suite']=meta.get('suite','guard_experiment')
        if guard:
            for key in ['guard_mode','raw_safe_abstention','future_evidence_exposed','future_ids_cited']:
                row[key]=meta[key]
        rows.append(row)
    return pd.DataFrame(rows)


def check_panel(frame,systems,cases,repeats,extra=None):
    keys=['system','case_id','repeat']+list((extra or {}).keys())
    import itertools
    expected=set(itertools.product(systems,cases,repeats,*list((extra or {}).values())))
    observed=set(frame[keys].itertuples(index=False,name=None)) if len(frame) else set()
    duplicates=int(frame.duplicated(keys).sum()) if len(frame) else 0
    return dict(expected=len(expected),completed=len(observed),missing=len(expected-observed),
                unexpected=len(observed-expected),duplicates=duplicates,
                complete=expected==observed and duplicates==0,
                transport_failures=int((frame.return_code.ne(0)|frame.timed_out).sum()) if len(frame) else 0)


def collect(out,require_complete=False):
    store=json.loads((ROOT/'data/local/evidence_store.json').read_text())
    val=sorted(c for c,v in store['cases'].items() if v['split']=='validation')
    test=sorted(c for c,v in store['cases'].items() if v['split']=='test')
    panel=store['cohort']['robustness_panel_case_ids']
    if not set(panel)<=set(test) or len(set(panel))!=40:raise ValueError('Guard panel must contain 40 distinct test cases')
    plan=load_scope()
    supplement,extra_supplement=select(read_runs(ROOT/'outputs/qwen_supplement'),plan)
    guards,extra_guards=select(read_runs(ROOT/'outputs/guard_experiment',guard=True),plan,guard=True)
    status={}
    for suite,systems,cases,repeats in [('validation',['B1','A1','B2'],val,[0]),
                                      ('matched_test',['B2'],test,plan['matched_test']['repeats'])]:
        sub=supplement[supplement.suite.eq(suite)] if len(supplement) else supplement
        status[suite]=check_panel(sub,systems,cases,repeats)
        status[suite]['required']=True
    spec=plan['guards']
    status['guards']=check_panel(guards,spec['systems'],panel,spec['repeats'],dict(scenario=spec['scenarios'],guard_mode=spec['modes']))
    status['guards']['required']=True
    status['low_temperature']=dict(required=False,complete=False,expected=0,completed=0,missing=0,
                                   unexpected=0,duplicates=0,transport_failures=0,
                                   status=plan['low_temperature']['status'])
    save_json(out/'inference_completion.json',status)
    if len(supplement):table(out,'supplement_runs',supplement)
    if len(guards):table(out,'guard_runs',guards)
    if len(extra_supplement):table(out,'supplement_runs_outside_scope',extra_supplement)
    if len(extra_guards):table(out,'guard_runs_outside_scope',extra_guards)
    save_json(out/'experiment_scope.json',dict(scope_id=plan['scope_id'],required_case_runs=plan['required_case_runs'],
       excluded_completed_supplement=len(extra_supplement),excluded_completed_guards=len(extra_guards),
       exclusion='Fixed scope amendment; additional records retained unchanged, not screened by prediction quality.'))
    if require_complete and not required_complete(status):
        raise ValueError('Additional inference panels are incomplete; see inference_completion.json. Partial outputs are not paper results.')
    return supplement,guards,status
