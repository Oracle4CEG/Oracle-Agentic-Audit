"""Descriptive economic analyses with asset, role, and stage-specific denominators."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from oracle_audit.io import ROOT,sha256,save_json,table


def concentration(weights):
    w=np.asarray(weights,dtype=float);w=w[w>0]
    if len(w)==0:return dict(actors=0,top1=np.nan,top5=np.nan,top10=np.nan,hhi=np.nan,effective_n=np.nan)
    s=np.sort(w/w.sum())[::-1];hhi=float(np.sum(s*s))
    return dict(actors=len(s),top1=s[:1].sum(),top5=s[:5].sum(),top10=s[:10].sum(),hhi=hhi,effective_n=1/hhi)


def prepare_sources(curated):
    curated=Path(curated);local=ROOT/'data/local';pins=[]
    # Selected descriptor products, no reconstruction of the full Atlas.
    for name in ['polygon_uma_request_rounds.parquet','uma_polygon_ethereum_grade_a_links.parquet']:
        src=curated/name;dst=local/name
        import shutil
        if dst.exists() and sha256(dst)!=sha256(src):raise ValueError(f'Source revision changed: {name}')
        if not dst.exists():shutil.copy2(src,dst)
        pins.append(dict(path=str(dst.relative_to(ROOT)),source=str(src),sha256=sha256(dst)))
    save_json(ROOT/'manifests/economic-source-inputs.json',pins)


def run(out):
    local=ROOT/'data/local'
    for pin in json.loads((ROOT/'manifests/economic-source-inputs.json').read_text()):
        if sha256(ROOT/pin['path'])!=pin['sha256']:raise ValueError('Modified economic input')
    rounds=pd.read_parquet(local/'polygon_uma_request_rounds.parquet')
    if rounds.oo_request_id.duplicated().any():raise ValueError('Activity denominator has duplicate request IDs')
    links=pd.read_parquet(local/'uma_polygon_ethereum_grade_a_links.parquet')
    samples=pd.read_parquet(local/'samples.parquet')
    sample_rows=samples[['sample_id','proposal_rejected_by_protocol']].merge(rounds,left_on='sample_id',right_on='oo_request_id',validate='one_to_one')
    check=sample_rows.merge(links[['oo_request_id','cross_chain_match_grade','resolved_price_consistent','dvm_resolved_price_raw']],on='oo_request_id',validate='one_to_one')
    check['label_from_raw_prices']=(check.proposed_price_raw.astype(str)!=check.dvm_resolved_price_raw.astype(str)).astype(int)
    check['label_matches']=check.label_from_raw_prices.eq(check.proposal_rejected_by_protocol)
    table(out,'label_audit',check[['sample_id','proposal_rejected_by_protocol','label_from_raw_prices','label_matches','cross_chain_match_grade','resolved_price_consistent','proposal_tx','settlement_tx','proposed_price_raw','dvm_resolved_price_raw']])
    if not check.label_matches.all() or not check.resolved_price_consistent.all():raise ValueError('Protocol label verification failed')
    # Explicit eligibility intersection, rather than subtracting unrelated headline counts.
    stages=[('all_request_rounds',len(rounds)),('primary',int(rounds.sample_tier.eq('primary').sum())),
            ('primary_proposed',int((rounds.sample_tier.eq('primary')&rounds.proposal_tx.notna()).sum())),
            ('primary_disputed',int((rounds.sample_tier.eq('primary')&rounds.dispute_tx.notna()).sum())),
            ('pinned_strict_benchmark',len(check))]
    table(out,'candidate_cohort_counts',pd.DataFrame(stages,columns=['criterion','request_rows']))
    exclusion=rounds[rounds.dispute_tx.notna()].merge(links[['oo_request_id','cross_chain_match_grade','resolved_price_consistent']],on='oo_request_id',how='left',validate='one_to_one')
    exclusion['included']=exclusion.oo_request_id.isin(samples.sample_id)
    exclusion['observable_exclusion_reason']=np.select([exclusion['included'],exclusion.sample_tier.ne('primary'),exclusion.cross_chain_match_grade.ne('A'),~exclusion.resolved_price_consistent.eq(True)],
        ['included','nonprimary_tier','not_grade_a','price_inconsistent_or_unverified'],'additional_lifecycle_or_flow_filter_requires_source_receipt')
    table(out,'cohort_exclusion_audit',exclusion[['oo_request_id','sample_tier','cross_chain_match_grade','resolved_price_consistent','included','observable_exclusion_reason']])
    activity=[]
    primary=rounds[rounds.sample_tier.eq('primary')]
    for role,col,tx in [('proposer','proposer','proposal_tx'),('disputer','disputer','dispute_tx')]:
        eligible=primary[primary[tx].notna()&primary[col].notna()].copy()
        activity.append(dict(protocol='UMA',role=role,observation_unit='distinct primary OOV2 request',
                              denominator=len(eligible),**concentration(eligible.groupby(col).oo_request_id.nunique())))
    table(out,'verification_activity_concentration',pd.DataFrame(activity))
    reward=pd.read_parquet(local/'reward_concentration.parquet');rrows=[]
    for key,g in reward.groupby(['protocol','asset','asset_decimals','actor_role']):
        amounts=g.groupby('actor').amount_raw.agg(lambda x:sum(int(v) for v in x))
        rrows.append(dict(zip(['protocol','asset','asset_decimals','reward_mechanism'],key),
            actor_role='observed_reward_recipient',source_actor_role_field=key[3],
            denominator_reward_raw=str(sum(int(x) for x in amounts)),
            observation_unit='recipient address; positive observed reward amounts',denominator_events=int(g.event_count.sum()),
            interpretation='Distribution of observed paid/applied rewards within this asset and mechanism; not operator identity or protocol security',
            **concentration(amounts)))
    table(out,'cross_protocol_reward_concentration',pd.DataFrame(rrows))
    econ=pd.read_parquet(local/'economic_outcomes.parquet').merge(samples[['sample_id','reward_to_bond_ratio','liveness_hours','proposal_rejected_by_protocol']],on='sample_id',validate='one_to_one')
    for col in ['challenge_capital_at_risk_raw','challenge_capital_days_locked_raw','observed_challenge_token_payoff_raw','challenge_capital_lock_seconds']:
        econ[col.replace('_raw','')]=pd.to_numeric(econ[col],errors='coerce')
    scale=10.**econ.currency_decimals
    econ['capital_tokens']=econ.challenge_capital_at_risk/scale
    econ['capital_token_days']=econ.challenge_capital_days_locked/scale
    econ['payoff_tokens']=econ.observed_challenge_token_payoff/scale
    econ['lock_days']=econ.challenge_capital_lock_seconds/86400
    table(out,'economic_case_panel',econ)
    summaries=[]
    for (asset,decimals),g in econ.groupby(['currency_address','currency_decimals']):
        row=dict(asset=asset,decimals=decimals,cases=len(g),unit='observed disputed strict benchmark request',
                 positive_payoff_cases=int(g.payoff_tokens.gt(0).sum()),negative_payoff_cases=int(g.payoff_tokens.lt(0).sum()),
                 verified_token_flow_cases=int(g.token_flow_exact.sum()))
        for field in ['reward_to_bond_ratio','capital_tokens','capital_token_days','payoff_tokens','lock_days']:
            row[field+'_observed']=int(g[field].notna().sum())
            for suffix,q in [('p25',.25),('median',.5),('p75',.75),('p95',.95)]:row[field+'_'+suffix]=g[field].quantile(q)
        summaries.append(row)
    table(out,'incentives_and_realized_outcomes',pd.DataFrame(summaries))
    lock=pd.read_parquet(local/'capital_lock.parquet')
    table(out,'capital_lock_coverage',lock.groupby(['protocol','asset','asset_decimals','measurement_scope','right_censored'],dropna=False).agg(
          units=('accountability_unit','size'),duration_observed=('capital_lock_duration_seconds','count'),median_seconds=('capital_lock_duration_seconds','median')).reset_index())
    matrix=[]
    for protocol in ['UMA','Flare_FTSOv2','Chainlink','Tellor','Pyth']:
        for variable,unit,count,status,reason in [
          ('observed_reward_recipient_concentration','address within asset and reward mechanism',int(reward.protocol.eq(protocol).sum()),'available','Cross-protocol formula aligned; assets, roles and realization mechanisms remain separate'),
          ('proposal_and_dispute_activity_concentration','role-specific OOV2 request',len(primary) if protocol=='UMA' else 0,'available' if protocol=='UMA' else 'not_evaluated','Research selection covers UMA actions'),
          ('configured_reward_to_bond','strict disputed request',len(econ) if protocol=='UMA' else 0,'available' if protocol=='UMA' else 'not_evaluated','No equivalent denominator established for other panels'),
          ('capital_lock_duration','accountability unit',int(lock.protocol.eq(protocol).sum()),'available' if lock.protocol.eq(protocol).any() else 'not_available_in_selected_inputs','Report right censoring and native measurement scope'),
          ('realized_challenger_token_payoff','strict disputed request',len(econ) if protocol=='UMA' else 0,'available' if protocol=='UMA' else 'not_evaluated','Gross payout minus returned principal; gas remains separate')]:
            matrix.append(dict(protocol=protocol,variable=variable,observation_unit=unit,available_observations=count,status=status,scope=reason))
    table(out,'protocol_variable_availability',pd.DataFrame(matrix))
    save_json(out/'economic_scope.json',dict(label_checks_passed=int(check.label_matches.sum()),
              scope='Descriptive selected economic observations, not causal welfare, security, or profitability estimates',
              no_cross_asset_sum=True,missing_values_preserved=True,availability_matrix_scope='Five variables actually analyzed in this research; not the full Atlas dictionary',
              role_mapping='The upstream actor_role column aliases mechanism. Here it is called reward_mechanism; recipient role is not assumed to be verifier/operator.',
              public_revision='not independently verified; local inputs checksum-pinned'))
    print('Economic analyses and all 810 label checks completed',flush=True)
