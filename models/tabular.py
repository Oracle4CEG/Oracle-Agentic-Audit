"""Frozen small-data baselines; fit preprocessing and models on training only."""
import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from oracle_audit.io import ROOT,save_json,table
from policies.calibration import fit,apply,logit

NUMERIC=['log_bond_raw','log_final_fee_raw','log_reward_plus1_raw','reward_to_bond_ratio',
         'liveness_hours','proposal_latency_hours','ancillary_bytes','requester_is_polymarket_adapter',
         'proposer_prior_completed','proposer_prior_dispute_rate','proposer_prior_rejection_rate',
         'requester_prior_completed','requester_prior_dispute_rate','requester_prior_rejection_rate']
CATEGORICAL=['proposed_price_class','adapter_version']
FEATURES=NUMERIC+CATEGORICAL


def run(out):
    local=ROOT/'data/local'; out.mkdir(parents=True,exist_ok=True)
    samples=pd.read_parquet(local/'samples.parquet').merge(pd.read_parquet(local/'splits.parquet'),on=['sample_id','decision_time_unix'],validate='one_to_one')
    train=samples.split.eq('train'); val=samples.split.eq('validation'); test=samples.split.eq('test')
    base=samples.loc[val|test,['sample_id','split','proposal_rejected_by_protocol']].rename(columns={'sample_id':'case_id','proposal_rejected_by_protocol':'label'}).reset_index(drop=True)
    base['repeat']=0; base['guard_blocked']=False; base['scenario']='clean';base['model_id']='non_llm'
    pre=ColumnTransformer([('numeric',StandardScaler(),NUMERIC),('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL)])
    config=json.loads((ROOT/'experiments/config.json').read_text())['strong_baseline'].copy();config.pop('family')
    strong=Pipeline([('preprocess',pre),('model',LGBMClassifier(**config,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True))])
    strong.fit(samples.loc[train,FEATURES],samples.loc[train,'proposal_rejected_by_protocol'])
    raw=base.assign(system='B0plus',calibration='raw',p_reject=strong.predict_proba(samples.loc[val|test,FEATURES])[:,1])
    cal=fit(raw[raw.split.eq('validation')]);calrows=raw.assign(calibration='platt',p_reject=apply(cal,raw.p_reject))
    joblib.dump(dict(model=strong,calibrator=cal,features=FEATURES),out/'lightgbm.joblib')
    # Reconstruct the original bootstrap logistic ensemble, including its five calibrators.
    p_raw=[];p_cal=[];rng_master=np.random.default_rng(20260806);train_idx=np.flatnonzero(train)
    for seed in [7,17,27,37,47]:
        rng=np.random.default_rng(seed+int(rng_master.integers(0,10000)))
        idx=rng.choice(train_idx,size=len(train_idx),replace=True)
        pipeline=Pipeline([('preprocess',ColumnTransformer([('numeric',StandardScaler(),NUMERIC),('categorical',OneHotEncoder(handle_unknown='ignore'),CATEGORICAL)])),
                           ('model',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=seed))])
        pipeline.fit(samples.iloc[idx][FEATURES],samples.iloc[idx].proposal_rejected_by_protocol)
        vp=pipeline.predict_proba(samples.loc[val,FEATURES])[:,1]
        cp=LogisticRegression(random_state=seed).fit(logit(vp),samples.loc[val,'proposal_rejected_by_protocol'])
        pred=pipeline.predict_proba(samples.loc[val|test,FEATURES])[:,1]
        p_raw.append(pred);p_cal.append(apply(cp,pred))
    b0raw=base.assign(system='B0',calibration='raw',p_reject=np.mean(p_raw,axis=0))
    b0cal=base.assign(system='B0',calibration='platt',p_reject=np.mean(p_cal,axis=0))
    prev=float(samples.loc[train,'proposal_rejected_by_protocol'].mean())
    constant=base.assign(system='TrainPrior',calibration='raw',p_reject=prev)
    predictions=pd.concat([raw,calrows,b0raw,b0cal,constant],ignore_index=True)
    original=pd.read_parquet(local/'original_b0.parquet').query("model == 'calibrated_logit_ensemble'")
    cmp=b0cal.query("split == 'test'").merge(original,left_on='case_id',right_on='sample_id',validate='one_to_one')
    error=float(abs(cmp.p_reject-cmp.probability_proposal_rejected).max())
    if error>1e-8: raise ValueError(f'Original B0 failed replication: max absolute difference {error}')
    table(out,'predictions',predictions)
    save_json(out/'training_receipt.json',dict(train_cases=int(train.sum()),validation_cases=int(val.sum()),test_cases=int(test.sum()),
              features=FEATURES,lightgbm_config=config,hyperparameter_search=False,calibration='validation only',
              train_prior=prev,b0_max_absolute_reference_error=error,
              lightgbm_calibrator=dict(coefficient=cal.coef_.tolist(),intercept=cal.intercept_.tolist()),
              model_file='lightgbm.joblib'))
    print(f'Trained B0plus; reconstructed B0 (max reference error {error:.3g})',flush=True)
    return predictions
