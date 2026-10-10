"""Review material for the original lexical reasoning guard, not a leakage detector."""
import json
import re
from pathlib import Path
import pandas as pd
from oracle_audit.io import ROOT, table, save_json, sha256
from experiments.prepare import read_events
from models.reference import run_agentic_oracle_auditor as ref
from policies.decision import registered


def extract(runs, out):
    rows=[]
    for row in runs[runs.scenario.eq('clean') & runs.reasoning_boundary_violation].sort_values(['system','repeat','case_id']).itertuples():
        events,_=read_events(ROOT/row.source/'events.jsonl')
        info=ref.event_metrics(events)
        snippets=[]
        for text in info['assistant_generated_texts']:
            for name,pattern in ref.UNGROUNDED_OUTCOME_PATTERNS.items():
                for match in re.finditer(pattern,text,re.I):
                    snippets.append(dict(pattern=name,context=text[max(0,match.start()-180):match.end()+240]))
        rows.append(dict(review_id=len(rows),system=row.system,repeat=row.repeat,case_id=row.case_id,
                         snippets=json.dumps(snippets),guard_errors=row.guard_errors,raw_action=row.raw_action,
                         final_action=row.action,p_reject=row.p_reject,
                         source_events_sha256=sha256(ROOT/row.source/'events.jsonl')))
    frame=pd.DataFrame(rows);table(out,'reasoning_guard_review_material',frame)
    return frame


def reviewed_summary(frame,out):
    path=ROOT/'analysis/reasoning_guard_annotations.json'
    if not path.exists():return
    annotations=json.loads(path.read_text())
    review=pd.DataFrame(annotations['reviews'])
    merged=frame.merge(review,on='review_id',validate='one_to_one')
    if len(merged)!=len(frame):raise ValueError('Incomplete guard annotation coverage')
    if not merged.source_events_sha256.eq(merged.reviewed_source_sha256).all():raise ValueError('Guard annotation source mismatch')
    merged['only_lexical_error']=merged.guard_errors.map(lambda x:json.loads(x)==['UNGROUNDED_POST_CUTOFF_KNOWLEDGE_CLAIM'])
    merged['confirmed_guard_false_alarm']=merged.assessment.eq('lexical_false_positive')
    merged['policy_without_lexical_guard']=registered(merged.p_reject)
    merged['potentially_avoidable_abstention']=(merged.confirmed_guard_false_alarm & merged.only_lexical_error
                                                & merged.policy_without_lexical_guard.ne('Abstain'))
    merged['potentially_lost_automatic_action']=merged.potentially_avoidable_abstention & merged.policy_without_lexical_guard.isin(['Accept','Challenge'])
    table(out,'reasoning_guard_review',merged)
    table(out,'reasoning_guard_review_summary',merged.groupby(['system','assessment']).agg(
        flagged_runs=('case_id','size'),unique_cases=('case_id','nunique'),sole_error_runs=('only_lexical_error','sum'),
        potentially_avoidable_abstentions=('potentially_avoidable_abstention','sum'),
        potentially_lost_automatic_actions=('potentially_lost_automatic_action','sum')).reset_index())
    save_json(out/'reasoning_guard_review_scope.json',dict(
        method=annotations['method'],denominator='76 lexical-guard-positive main Qwen case-runs; not all 1600 outputs',
        limitations=['Single AI-assisted review, not independent blinded expert annotation.',
                     'False-positive classifications concern the lexical match, not proof that the whole output is grounded.',
                     'A model claim of memory is not proof that the claimed fact is true or was in training data.',
                     'This positive-only review cannot estimate recall or exclude unflagged world-knowledge use.']))
