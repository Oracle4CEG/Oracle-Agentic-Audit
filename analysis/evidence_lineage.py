"""Reconstruct model-visible market features from archived, timestamped price points."""
import gzip
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from oracle_audit.io import ROOT,sha256,save_json,table


def prepare(workspace):
    workspace=Path(workspace);local=ROOT/'data/local';pins=[]
    mapping={
        'market_feature_snapshot.parquet':workspace/'data/applications/trustworthy_ai_market_evidence/market_feature_snapshot.parquet',
        'price_provenance.parquet':workspace/'data/curated/parquet/polymarket_decision_time_price_provenance.parquet',
        'price_source_code.py':workspace/'scripts/ingest_polymarket_decision_prices.py',
        'market_feature_source_code.py':workspace/'scripts/applications/trustworthy_ai/run_market_evidence_benchmark.py'}
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists() and sha256(dst)!=sha256(src):raise ValueError('Evidence lineage source changed')
        if not dst.exists():shutil.copy2(src,dst)
        pins.append(dict(path=str(dst.relative_to(ROOT)),sha256=sha256(dst),source=str(src)))
    for name,src in mapping.items():copy(src,local/name)
    for row in pd.read_parquet(local/'price_provenance.parquet').itertuples():
        src=Path(row.raw_snapshot)
        if sha256(src)!=row.raw_sha256:raise ValueError('Raw price archive mismatch')
        copy(src,local/'price_histories'/src.name)
    save_json(ROOT/'manifests/evidence-lineage-inputs.json',pins)


def history_features(history,decision):
    if not history:return dict(last_price=None,price_age_seconds=None,momentum_1h=None,momentum_6h=None,
                               momentum_24h=None,volatility_24h=None,range_24h=None,points_1h=0,points_6h=0,points_24h=0,points_168h=0)
    last=history[-1];out=dict(last_price=last['p'],price_age_seconds=decision-last['t'])
    for h in [1,6,24]:
        before=[p['p'] for p in history if p['t']<=decision-h*3600]
        out[f'momentum_{h}h']=last['p']-before[-1] if before else None
    recent=[p['p'] for p in history if p['t']>=decision-86400]
    out['volatility_24h']=float(np.std(np.diff(recent))) if len(recent)>=2 else None
    out['range_24h']=max(recent)-min(recent) if recent else None
    for h in [1,6,24,168]:out[f'points_{h}h']=sum(p['t']>=decision-h*3600 for p in history)
    return out


def audit(out):
    for pin in json.loads((ROOT/'manifests/evidence-lineage-inputs.json').read_text()):
        if sha256(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Modified price lineage input')
    store=json.loads((ROOT/'data/local/evidence_store.json').read_text())
    provenance=pd.read_parquet(ROOT/'data/local/price_provenance.parquet').set_index('sample_id')
    records=[];checked_fields=0
    for case_id,case in sorted(store['cases'].items()):
        path=ROOT/'data/local/price_histories'/Path(provenance.loc[case_id,'raw_snapshot']).name
        with gzip.open(path,'rt') as stream:raw=json.load(stream)
        cutoff=case['cutoff_unix'];assert raw['sample_id']==case_id and raw['decision_time_unix']==cutoff
        kind=case['proposal']['proposed_price_class'];present=bool(case['optional_evidence_ids'])
        future=0;pointcount=0;latest=None;match=True
        if raw['status']=='complete':
            assert raw['request_parameters']['endTs']==cutoff
            assert set(raw['gamma_token_ids'])=={raw['primary_yes']['token_id'],raw['secondary_no']['token_id']}
            ys=raw['primary_yes']['history'];ns=raw['secondary_no']['history']
            points=ys+ns;future=sum(p['t']>cutoff for p in points);pointcount=len(points)
            latest=max((p['t'] for p in points),default=None)
            y=history_features(ys,cutoff);n=history_features(ns,cutoff)
            aligned=y if kind=='binary_one' else n if kind=='binary_zero' else {}
            opposite=n if kind=='binary_one' else y if kind=='binary_zero' else {}
            expected=dict(coverage_status='complete',aligned_probability=aligned.get('last_price'),
                          opposing_probability=opposite.get('last_price'),
                          price_sum=y['last_price']+n['last_price'] if y['last_price'] is not None and n['last_price'] is not None else None,
                          market_confidence=abs(y['last_price']-.5)*2 if y['last_price'] is not None else None)
            for k in ['price_age_seconds','momentum_1h','momentum_6h','momentum_24h','volatility_24h','range_24h']:expected['aligned_'+k]=aligned.get(k)
            for h in [1,6,24,168]:expected[f'total_points_{h}h']=y[f'points_{h}h']+n[f'points_{h}h']
            if not present:raise ValueError('Complete market evidence missing from store')
            content=store['evidence'][case['optional_evidence_ids'][0]]['content']
            for key,value in expected.items():
                actual=content[key];checked_fields+=1
                if value is None:equal=actual is None
                elif isinstance(value,str):equal=actual==value
                else:equal=actual is not None and np.isclose(actual,value,rtol=0,atol=1e-12)
                match=match and equal
        elif present:raise ValueError('Unavailable market evidence invented in store')
        retrieval=pd.Timestamp(raw['retrieved_at_utc'])
        decision=pd.Timestamp(cutoff,unit='s',tz='UTC')
        records.append(dict(case_id=case_id,split=case['split'],market_evidence_present=present,
                            raw_sha256=sha256(path),feature_reconstruction_pass=match,price_points=pointcount,
                            post_cutoff_points=future,latest_price_age_seconds=cutoff-latest if latest is not None else None,
                            retrieval_time_utc=str(retrieval),archived_after_decision=retrieval>decision,
                            archive_delay_days=(retrieval-decision).total_seconds()/86400))
    frame=pd.DataFrame(records);table(out,'market_lineage_audit',frame)
    if not frame.feature_reconstruction_pass.all() or frame.post_cutoff_points.any():raise ValueError('Price evidence reconstruction or cutoff check failed')
    table(out,'market_availability_summary',frame.groupby('split').agg(cases=('case_id','size'),market_available=('market_evidence_present','sum'),
          retrospective_archives=('archived_after_decision','sum'),price_points=('price_points','sum'),
          median_last_price_age_seconds=('latest_price_age_seconds','median'),p95_last_price_age_seconds=('latest_price_age_seconds',lambda x:x.quantile(.95))).reset_index())
    save_json(out/'market_lineage_receipt.json',dict(cases=len(frame),fields_checked=checked_fields,
         reconstructed_market_records=int(frame.market_evidence_present.sum()),price_points=int(frame.price_points.sum()),
         post_cutoff_points=int(frame.post_cutoff_points.sum()),
         limitation='Archived retrospective official price-index responses with cutoff-bounded event timestamps. These checks do not prove prospective publication, finality latency, immutable API history, or independently decoded exchange fills.'))
    return frame
