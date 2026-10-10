"""Future-field perturbations and shuffled training labels as pipeline diagnostics."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score,brier_score_loss
from oracle_audit.io import ROOT,save_json,table
from models.reference import cohort_source as source
from models.tabular import FEATURES
from policies.calibration import fit,apply


def future_fields(out):
    source.DATA_ROOT=ROOT/'data/local';frame=source.extract_rows()
    original,_=source.build_samples(frame);original=original.set_index('sample_id')
    splits=pd.read_parquet(ROOT/'data/local/splits.parquet')
    selected=[]
    for _,g in splits.groupby('split'):selected+=sorted(g.sample_id)[:2]
    checks=[]
    for case in selected:
        changed=frame.copy();index=changed.index[changed.oo_request_id.eq(case)]
        if len(index)!=1:raise ValueError('Negative-control case is not unique')
        # Hold eligibility fixed: this is a dataflow perturbation, not a new chain record.
        proposed=str(changed.loc[index[0],'proposed_price_raw']);resolved=str(changed.loc[index[0],'resolved_price_raw'])
        changed.loc[index,'resolved_price_raw']=proposed if proposed!=resolved else ('0' if proposed!='0' else '1000000000000000000')
        changed.loc[index,'gross_payout_raw']='123456789012345'
        changed.loc[index,'economic_status']='settled_disputed_disputer_wins'
        rebuilt,_=source.build_samples(changed);row=rebuilt.set_index('sample_id').loc[case]
        equal=True
        for field in FEATURES:
            a,b=original.loc[case,field],row[field]
            equal=equal and (bool(np.isclose(a,b,rtol=0,atol=1e-12)) if isinstance(a,(float,int,np.number)) else a==b)
        checks.append(dict(case_id=case,split=splits.set_index('sample_id').loc[case,'split'],features_unchanged=equal))
    result=pd.DataFrame(checks);table(out,'future_field_negative_controls',result)
    if not result.features_unchanged.all():raise ValueError('Future-only perturbation changes own-case input features')


def shuffled_labels(out,repeats=100,model_path=None):
    samples=pd.read_parquet(ROOT/'data/local/samples.parquet').merge(pd.read_parquet(ROOT/'data/local/splits.parquet'),on=['sample_id','decision_time_unix'],validate='one_to_one')
    train=samples[samples.split.eq('train')];val=samples[samples.split.eq('validation')];test=samples[samples.split.eq('test')]
    template=joblib.load(model_path or ROOT/'outputs/tabular/lightgbm.joblib')['model']
    rng=np.random.default_rng(20261009);rows=[]
    for i in range(repeats):
        model=clone(template)
        model.fit(train[FEATURES],rng.permutation(train.proposal_rejected_by_protocol))
        vp=model.predict_proba(val[FEATURES])[:,1]
        calibration=pd.DataFrame(dict(case_id=val.sample_id.to_numpy(),split='validation',
            p_reject=vp,label=rng.permutation(val.proposal_rejected_by_protocol)))
        raw=model.predict_proba(test[FEATURES])[:,1];cal=apply(fit(calibration),raw)
        rows.append(dict(permutation=i,train_cases=len(train),validation_cases=len(val),test_cases=len(test),
             raw_auroc=roc_auc_score(test.proposal_rejected_by_protocol,raw),calibrated_auroc=roc_auc_score(test.proposal_rejected_by_protocol,cal),
             raw_brier=brier_score_loss(test.proposal_rejected_by_protocol,raw),calibrated_brier=brier_score_loss(test.proposal_rejected_by_protocol,cal)))
    table(out,'shuffled_label_negative_controls',pd.DataFrame(rows))
    save_json(out/'negative_control_scope.json',dict(
        future_perturbation='Six fixed cases, two per chronological split. Own resolved price, payout and terminal-status fields changed while eligibility is held fixed. Own proposal-time features must stay invariant.',
        label_control=f'{repeats} independent train/validation label permutations; fixed LightGBM settings; actual test labels retained only for evaluation.',
        limitation='Diagnostics of these dataflow and label paths, not proof that upstream histories, semantics or model pretraining are free of leakage. Not competitive baselines.'))
