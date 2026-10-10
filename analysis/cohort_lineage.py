"""Independent rerun of the archived benchmark cohort/feature construction."""
import json
import numpy as np
import pandas as pd
from oracle_audit.io import ROOT,sha256,save_json,table
from models.reference import cohort_source as source
from models.tabular import FEATURES


def audit(out):
    for name in ['cohort-lineage-inputs.json','economic-source-inputs.json']:
        for pin in json.loads((ROOT/'manifests'/name).read_text()):
            if sha256(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Modified cohort lineage input')
    source.DATA_ROOT=ROOT/'data/local'
    candidates=source.extract_rows();rebuilt,_=source.build_samples(candidates)
    expected=pd.read_parquet(ROOT/'data/local/samples.parquet')
    if set(rebuilt.sample_id)!=set(expected.sample_id):raise ValueError('Rebuilt strict cohort differs')
    a=rebuilt.set_index('sample_id').sort_index();b=expected.set_index('sample_id').sort_index()
    comparisons=[]
    for name in FEATURES+['decision_time_unix','proposal_rejected_by_protocol']:
        if pd.api.types.is_numeric_dtype(a[name]):match=np.isclose(a[name],b[name],rtol=0,atol=1e-12,equal_nan=True)
        else:match=a[name].eq(b[name]).to_numpy()
        comparisons.append(dict(field=name,cases=len(a),matches=int(match.sum()),pass_all=bool(match.all())))
    splits=source.assign_temporal_splits(rebuilt).set_index('sample_id').sort_index()
    pinned=pd.read_parquet(ROOT/'data/local/splits.parquet').set_index('sample_id').sort_index()
    if not splits['split'].equals(pinned['split']):raise ValueError('Rebuilt chronological splits differ')
    checks=pd.DataFrame(comparisons);table(out,'feature_lineage_checks',checks)
    if not checks.pass_all.all():raise ValueError('Feature or label reconstruction differs')
    # All candidate exclusions are reconstructible from the exact original predicates.
    reasons=[]
    for row in candidates.itertuples(index=False):
        failed=[]
        for condition,label in [(pd.notna(row.dispute_tx),'dispute_transaction_missing'),(pd.notna(row.dispute_time),'dispute_time_missing'),
              (row.cross_chain_match_grade=='A','not_grade_a'),(pd.notna(row.resolved_price_consistent) and bool(row.resolved_price_consistent),'resolution_price_unverified'),
              (pd.notna(row.settlement_flow_exact) and bool(row.settlement_flow_exact),'settlement_flow_not_exact'),
              (pd.notna(row.proposed_price_raw) and pd.notna(row.resolved_price_raw),'price_missing'),
              (pd.notna(row.dispute_time) and row.proposal_time<=row.dispute_time,'dispute_precedes_proposal'),
              (pd.notna(row.dispute_time) and row.dispute_time<=float(row.expiration_time),'dispute_after_expiration')]:
            if not condition:failed.append(label)
        reasons.append(dict(case_id=row.oo_request_id,included=not failed,failed_predicates=json.dumps(failed)))
    table(out,'exact_candidate_exclusions',pd.DataFrame(reasons))
    save_json(out/'feature_lineage_reconstruction.json',dict(
        source_code_sha256=sha256(source.__file__),candidate_settled_primary_proposals=len(candidates),
        reconstructed_cases=len(rebuilt),matched_fields=FEATURES+['decision_time_unix','proposal_rejected_by_protocol'],
        temporal_splits_match=True,all_fields_match=True,
        history_rule='Only earlier proposed episodes with settlement_time <= current proposal time contribute to lagged actor counts/rates.',
        limitation='Reconstruction from pinned curated event/flow/link tables is checked; upstream chain extraction is owned by the descriptor.'))
