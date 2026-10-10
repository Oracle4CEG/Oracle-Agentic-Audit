"""Structural citation taxonomy with delivery and semantic limits kept distinct."""
import json
from pathlib import Path
import pandas as pd
from oracle_audit.io import ROOT,table,save_json
from experiments.qwen_records import matched_evidence
from analysis.matching import visible_ids


def analyze(runs,out):
    store=json.loads((ROOT/'data/local/evidence_store.json').read_text());records=[];run_rows=[]
    for row in runs[runs.scenario.eq('clean')].itertuples():
        case=store['cases'][row.case_id];required=set(case['required_evidence_ids'])
        folder=ROOT/row.source;ids=json.loads(row.evidence_ids)
        if not isinstance(ids,list):ids=[]
        if row.system=='A1':delivered=visible_ids(matched_evidence(folder))
        elif row.system=='B2':
            packet=json.loads(json.loads((folder/'input_messages.json').read_text())[1]['content'])
            delivered=visible_ids(packet['observed_tool_responses'])
        else:
            # Canonical B1 input contains required and optional evidence records.
            delivered=required|set(case.get('optional_evidence_ids',[]))
        unknown=future=not_delivered=cross_case=0
        for eid in ids:
            evidence=store['evidence'].get(eid)
            if evidence is None:category='nonexistent_id';unknown+=1
            elif evidence['evidence_time_unix']>case['cutoff_unix']:category='post_cutoff_id';future+=1
            elif eid not in delivered:category='valid_id_content_not_received';not_delivered+=1
            elif evidence['case_id']!=row.case_id:category='valid_received_other_case';cross_case+=1
            else:category='valid_received_current_case'
            records.append(dict(case_id=row.case_id,repeat=row.repeat,system=row.system,evidence_id=eid,category=category,
                                semantic_entailment='not_measured'))
        run_rows.append(dict(case_id=row.case_id,repeat=row.repeat,system=row.system,
                             required_content_received=required<=delivered,required_ids_cited=required<=set(ids),
                             unknown_id=unknown>0,future_id=future>0,
                             valid_id_not_received=not_delivered>0,other_case_cited=cross_case>0,missing_required=bool(required-set(ids)),
                             output_invalid=not row.output_valid,no_citations=len(ids)==0))
    frame=pd.DataFrame(records);runframe=pd.DataFrame(run_rows)
    table(out,'citation_taxonomy',frame);table(out,'citation_run_errors',runframe)
    table(out,'citation_taxonomy_counts',frame.groupby(['system','category']).size().rename('citation_occurrences').reset_index())
    counts=frame.groupby(['system','category']).size().rename('citation_occurrences').reset_index()
    counts['system_citation_occurrences']=counts.groupby('system').citation_occurrences.transform('sum')
    counts['fraction_of_citation_occurrences']=counts.citation_occurrences/counts.system_citation_occurrences
    table(out,'citation_occurrence_rates',counts)
    error_fields=['unknown_id','future_id','valid_id_not_received','other_case_cited',
                  'missing_required','output_invalid','no_citations']
    per_case=runframe.groupby(['system','case_id'])[error_fields].max().reset_index()
    table(out,'citation_case_errors',per_case)
    table(out,'citation_case_error_rates',per_case.groupby('system')[error_fields].mean().reset_index())
    table(out,'citation_error_rates',runframe.groupby('system').agg(
        case_runs=('case_id','size'),cases=('case_id','nunique'),unknown_id_rate=('unknown_id','mean'),
        required_content_received_rate=('required_content_received','mean'),required_ids_cited_rate=('required_ids_cited','mean'),
        future_id_rate=('future_id','mean'),valid_id_not_received_rate=('valid_id_not_received','mean'),
        other_case_cited_rate=('other_case_cited','mean'),missing_required_rate=('missing_required','mean'),
        output_invalid_rate=('output_invalid','mean'),no_citations_rate=('no_citations','mean')).reset_index())
    save_json(out/'citation_taxonomy_scope.json',dict(
        interpretation='Categories concern ID existence, event time and actual content delivery. They do not certify support for a prediction.',
        other_case='Admissible retrieved historical context is permitted; a different case ID alone is not irrelevance.',
        semantic_relevance='Not established: final schema has evidence IDs and an optional abstention reason, without claim-to-source entailment annotations. No valid-but-irrelevant rate is invented.',
        denominators='Occurrence rates count every cited ID; case-run rates count each output once; case-level rates mean at least one failure across all five repeats, not the average repeat rate.',
        missing_required='A separate run-level completeness error, not a nonexistent citation. Categories may overlap at run level.'))
