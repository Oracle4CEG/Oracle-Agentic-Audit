"""Export the 18 camera-ready result tables and exact prompts from frozen evidence.

Only table definitions are included here; confidential manuscript prose is excluded.
"""
import argparse
import ast
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import pandas as pd
from oracle_audit.io import ROOT, save_json, sha256


def render(results, output=None):
    REPO=ROOT
    RESULTS=Path(results)
    output=Path(output) if output else RESULTS
    def write(relative, text):
        path=output/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text+'\n',encoding='utf-8')
    def rows(name):
        with (RESULTS / (name + '.csv')).open() as f:
            return list(csv.DictReader(f))

    def esc(x):
        return str(x).replace('&', r'\&').replace('_', r'\_').replace('%', r'\%')

    def fmt(x, digits=4):
        if x in ['', None]: return '--'
        return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))

    def pct(x, digits=2):
        return '--' if x in ['', None] else fmt(Decimal(str(x))*100, digits)

    table_sources = {}
    def table(name, caption, label, headers, data, sources, notes='', long=False):
        # Wide, long result tables belong in the appendix; no font shrinking by resizebox.
        cols = 'l' + 'r' * (len(headers)-1)
        data = [list(map(str, row)) for row in data]
        head = ' & '.join(headers) + r' \\'
        body = '\n'.join(' & '.join(row) + r' \\' for row in data)
        if long:
            tex = (r'\begingroup\small\setlength{\tabcolsep}{3pt}' + '\n'
                   + r'\begin{longtable}{' + cols + '}\n'
                   + r'\caption{' + caption + r'}\label{' + label + r'}\\' + '\n'
                   + '\\toprule\n' + head + '\n\\midrule\n\\endfirsthead\n'
                   + '\\toprule\n' + head + '\n\\midrule\n\\endhead\n'
                   + body + '\n\\bottomrule\n\\end{longtable}\n\\endgroup\n' + notes)
        else:
            tex = (r'\begin{table}[t]\centering\small' + '\n'
                   + r'\setlength{\tabcolsep}{3pt}' + '\n'
                   + r'\caption{' + caption + '}\n' + r'\label{' + label + '}\n'
                   + r'\begin{tabular}{' + cols + '}\n\\toprule\n' + head + '\n\\midrule\n'
                   + body + '\n\\bottomrule\n\\end{tabular}\n'
                   + (r'\par\smallskip\parbox{\linewidth}{\footnotesize ' + notes + '}\n' if notes else '')
                   + r'\end{table}')
        write(f'tables/{name}.tex', tex)
        table_sources[name] = dict(sources=sources, rows=len(data), caption=caption,
                                   headers=headers, data=data, notes=notes)

    table('cohort', 'Chronological benchmark partitions. Dates are UTC; exact timestamps are retained in the case manifest.', 'tab:cr-cohort',
          ['Split', 'Cases', 'First proposal', 'Last proposal', 'Rejected'],
          [[esc(r['split']),r['cases'],r['start_utc'][:10],r['end_utc'][:10],r['rejected']] for r in rows('cohort_summary')],
          ['cohort_summary.csv'])

    metrics = rows('policy_metrics')
    def metric(system, cal='raw', policy='registered', guard='on'):
        return next(r for r in metrics if (r['system'],r['calibration'],r['policy'],r['guard']) == (system,cal,policy,guard))
    names = {'B0plus':'B0+', 'TrainPrior':'Train prior'}
    def sysname(s): return names.get(s,s)
    selected = [('TrainPrior','raw'),('B0','platt'),('B0plus','raw'),('B0plus','platt'),('B1','raw'),('A1','raw')]
    main_rows=[]
    for s,c in selected:
        r=metric(s,c)
        main_rows.append([sysname(s),c]+[fmt(r[k]) for k in ['auroc','brier','decision_loss','coverage','selective_error']])
    main_rows.append(['Always investigate','--','--','--','0.1000','0.0000','--'])
    table('main_results','Verified main results under the original action policy with guard overrides, except the unconstrained always-Investigate reference.', 'tab:cr-main',
          ['System','Calibration','AUROC','Brier','Loss','Coverage','Sel. error'],main_rows,['policy_metrics.csv'],
          'There are 160 unique test cases. B1/A1 metrics average five repeats; feature baselines are deterministic. Coverage is a fraction, not a percentage. A dash denotes an undefined or inapplicable quantity. An action-only constant baseline has no probability score. B2 uses a separate matched-repeat table; calibrated LLM results are in the appendix.')

    matched=[r for r in rows('matched_evidence_comparison_metrics') if r['calibration']=='raw' and r['policy']=='registered' and r['guard']=='on']
    table('matched_evidence','Matched-evidence comparison on exactly the same 160 cases and repeats 3 and 4 (320 case-runs per system).','tab:cr-matched',
          ['System','AUROC','Brier','Loss','Coverage','Sel. error'],
          [[r['system']]+[fmt(r[k]) for k in ['auroc','brier','decision_loss','coverage','selective_error']] for r in matched],
          ['matched_evidence_comparison_metrics.csv'], 'The A1 values in this table differ from its five-repeat means by design.')

    costs={r['system']:r for r in rows('cost_accounting')}
    table('cost_summary','Mean logged workload per active case-run. Local monetary cost is unmeasured.','tab:cr-cost',
          ['System','Runs','Prompt','Completion','Total','Seconds'],
          [[s,costs[s]['case_runs']]+[fmt(costs[s][k],2) for k in ['input_tokens_mean','output_tokens_mean','total_tokens_mean','latency_seconds_mean']] for s in ['B1','A1','B2']],
          ['cost_accounting.csv'], 'Cached prompt tokens are included once in prompt tokens. B2 uses two repeats; B1/A1 use five. Wall time is descriptive across recorded runtime conditions. Archived failed or superseded attempts are outside these active-panel means.')

    scenarios=['clean','missing_required','conflicting_records','future_record','prompt_injection','plausible_wrong']
    scenario_names={'clean':'Clean','missing_required':'Missing required','conflicting_records':'Conflicting','future_record':'Future record','prompt_injection':'Known injection','plausible_wrong':'Plausible wrong'}
    guards={(r['scenario'],r['guard_mode']):r for r in rows('guard_scenario_metrics')}
    table('guard_summary','A1 full-system ablation: 40 cases per condition. Rates are percentages.','tab:cr-guards',
          ['Condition','Auto off','Auto on','Abstain off','Abstain on'],
          [[scenario_names[s]]+[pct(guards[s,g][k],1) for k in ['final_auto_rate','final_abstain_rate'] for g in ['all_off','on']] for s in scenarios],
          ['guard_scenario_metrics.csv'], 'All-off jointly ablates admission, validation, and final override. The 17.5\\% clean-input abstention rate includes output-contract failures and is not an independently established semantic false-positive rate.')

    # Extract the actual prompts without importing code or contacting the model server.
    module=ast.parse((REPO/'models/reference/run_agentic_oracle_auditor.py').read_text())
    constants={}
    def literal(node):
        if isinstance(node,ast.Constant): return node.value
        if isinstance(node,ast.Name): return constants[node.id]
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add): return literal(node.left)+literal(node.right)
        raise ValueError('Unexpected prompt expression')
    for node in module.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            key=node.targets[0].id
            if key in ['COMMON_INSTRUCTION','B1_INSTRUCTION','A1_INSTRUCTION']: constants[key]=literal(node.value)
    prompt_tex=[]
    for key,title in [('COMMON_INSTRUCTION','Common instruction'),('B1_INSTRUCTION','B1 suffix (also used by B2)'),('A1_INSTRUCTION','A1 suffix')]:
        content=constants[key] if key=='COMMON_INSTRUCTION' else constants[key][len(constants['COMMON_INSTRUCTION']):]
        prompt_tex.append(r'\paragraph{'+title+'}\n'+r'\begin{lstlisting}[basicstyle=\ttfamily\footnotesize,breaklines=true,columns=fullflexible]'+'\n'+content.strip()+'\n'+r'\end{lstlisting}')
    write('tables/prompts_exact.tex','\n\n'.join(prompt_tex))
    write('evidence/prompt_strings.json',json.dumps(constants,indent=2))

    table('request_composition','Observed request categories by split; these are not semantic topic annotations.','tab:cr-request-composition',
          ['Split','Adapter','Proposed-value class','Cases'],
          [[esc(r[k]) for k in ['split','adapter_version','proposed_price_class','cases']] for r in rows('request_composition')],
          ['request_composition.csv'],long=True)

    actor_ci=rows('actor_group_confidence_intervals')
    actor_rows=[]
    for r in rows('seen_unseen_actor_metrics'):
        if r['system'] not in ['B1','A1','B0plus'] or r['calibration']!='platt':continue
        ci=next(x for x in actor_ci if x['system']==r['system'] and x['calibration']=='platt' and x['role']==r['role'] and x['seen_in_train']==r['seen_in_train'] and x['metric']=='auroc')
        actor_rows.append([sysname(r['system']),r['role'],'Seen' if r['seen_in_train']=='True' else 'Unseen',r['cases'],fmt(r['auroc']),f"[{fmt(ci['ci_low'])}, {fmt(ci['ci_high'])}]",fmt(r['brier'])])
    table('actor_groups','Actor-group diagnostics for validation-calibrated probabilities under the original policy. AUROC intervals use case bootstrap; raw point estimates are discussed in the text.','tab:cr-actors',
          ['System','Role','Training overlap','Cases','AUROC',r'95\% interval','Brier'],actor_rows,
          ['seen_unseen_actor_metrics.csv','actor_group_confidence_intervals.csv'],long=True)

    policy_rows=[]
    for s in ['TrainPrior','B0','B0plus','B1','A1','B2']:
        for c in ['raw','platt']:
            for policy in ['registered','cost_derived','validation_tuned']:
                a=[r for r in metrics if (r['system'],r['calibration'],r['policy'],r['guard'])==(s,c,policy,'on')]
                if not a: continue
                r=a[0]
                policy_rows.append([sysname(s),c,{'registered':'Original','cost_derived':'Cost-derived','validation_tuned':'Val. selected'}[policy]]+[fmt(r[k]) for k in ['auroc','brier','ece10','decision_loss','coverage','selective_error']])
    table('policy_details','Complete main policy variants with guard overrides. B2 has two repeats; A1/B1 have five. B2 paired contrasts use the separate matched table.','tab:cr-policies',
          ['System','Cal.','Policy','AUROC','Brier','ECE','Loss','Cov.','Sel. err.'],policy_rows,['policy_metrics.csv'],long=True)

    sensitivity=[]
    for r in rows('cost_sensitivity'):
        if r['guard']!='on' or r['system'] not in ['B0plus','B1','A1']:continue
        sensitivity.append([sysname(r['system']),r['calibration'],fmt(r['investigate_cost'],2),fmt(r['abstain_cost'],2),fmt(r['decision_loss']),fmt(r['coverage'])])
    table('sensitivity','Investigation-cost sensitivity with error costs fixed at one and Abstain fixed at 0.2. All reported rows use guard overrides.','tab:cr-sensitivity',
          ['System','Cal.','Investigate cost','Abstain cost','Loss','Coverage'],sensitivity,['cost_sensitivity.csv'],long=True)

    paired=[]
    for r in rows('paired_bootstrap'):
        if (r['left'],r['right']) not in [('B1','A1'),('B2','A1'),('B0','B0plus')]:continue
        if r['metric'] not in ['auroc','decision_loss','coverage']:continue
        paired.append([sysname(r['right'])+'--'+sysname(r['left']),r['calibration'],{'registered':'Original','cost_derived':'Cost-derived'}[r['policy']],{'auroc':'AUROC','decision_loss':'Loss','coverage':'Coverage'}[r['metric']],fmt(r['difference']),f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]"])
    table('paired_intervals','Paired differences (right system minus left system), with 2,000 case-bootstrap resamples and no multiplicity adjustment.','tab:cr-paired',
          ['Contrast','Cal.','Policy','Metric','Difference',r'95\% interval'],paired,['paired_bootstrap.csv'],long=True)

    guard_details=[]
    for s in scenarios:
        for g in ['all_off','on']:
            r=guards[s,g]
            guard_details.append([scenario_names[s],'Off' if g=='all_off' else 'On']+[pct(r[k],1) for k in ['raw_auto_rate','unenforced_policy_auto_rate','final_auto_rate','final_abstain_rate','future_exposure_rate','future_citation_rate']])
    table('guard_detail','A1 model action, probability-policy action, final enforcement, and actual future-content delivery. All rates are percentages (40 cases per row).','tab:cr-guard-detail',
          ['Condition','Guards','Model auto','Policy auto','Final auto','Abstain','Future body','Future ID'],guard_details,
          ['guard_scenario_metrics.csv'],long=True)

    citations={r['system']:r for r in rows('citation_error_rates')}
    audit_ci=rows('citation_cost_confidence_intervals')
    citation_rows=[]
    fields=[('required_content_received_rate','Required content received'),('required_ids_cited_rate','All required IDs cited'),('unknown_id_rate','At least one unknown ID'),('future_id_rate','At least one future ID'),('valid_id_not_received_rate','Valid ID without delivered content'),('missing_required_rate','Missing required citation'),('output_invalid_rate','Invalid final output'),('no_citations_rate','No citations')]
    for key,label in fields:
        citation_rows.append([label]+[pct(citations[s][key],3) for s in ['B1','A1','B2']])
    citation_rows.append(['All citations valid']+[pct(next(r['estimate'] for r in audit_ci if r['system']==s and r['metric']=='citation_valid'),3) for s in ['B1','A1','B2']])
    citation_rows.append(['Semantic irrelevance','Not measured','Not measured','Not measured'])
    table('citation_details','Citation diagnostics: percentage of case-runs. Denominators are 800 for B1/A1 and 320 for B2.','tab:cr-citations',
          ['Diagnostic','B1','A1','B2'],citation_rows,['citation_error_rates.csv','citation_cost_confidence_intervals.csv'],
          'Error categories overlap. Unknown-ID occurrence counts are 17, 17, and 7 for B1, A1, and B2; these differ from affected case-run counts. Semantic relevance is not inferred from identifier validity.')

    cost_detail=[]
    for s in ['B1','A1','B2']:
        for field,label in [('total_tokens','Total tokens'),('latency_seconds','Seconds'),('required_citation_complete','Required citation fraction'),('citation_valid','Citation-valid fraction'),('cap_reached','Cap fraction')]:
            r=next(r for r in audit_ci if r['system']==s and r['metric']==field)
            d=2 if field in ['total_tokens','latency_seconds'] else 4
            cost_detail.append([s,label,fmt(r['estimate'],d),f"[{fmt(r['ci_low'],d)}, {fmt(r['ci_high'],d)}]",r['case_runs']])
    table('cost_details','Case-bootstrap intervals for workload and evidence metrics. Intervals do not control for serving-runtime differences.','tab:cr-accounting',
          ['System','Metric','Estimate',r'95\% interval','Runs'],cost_detail,['citation_cost_confidence_intervals.csv'],long=True)

    table('economic_activity','UMA role-specific activity concentration. Denominators are distinct primary OOV2 requests for the stated role.','tab:cr-economic-activity',
          ['Role','Requests','Addresses',r'Top 1 (\%)',r'Top 10 (\%)','HHI'],
          [[r['role'],r['denominator'],r['actors'],pct(r['top1']),pct(r['top10']),fmt(r['hhi'])] for r in rows('economics/verification_activity_concentration')],
          ['economics/verification_activity_concentration.csv'])
    econ=rows('economics/incentives_and_realized_outcomes')
    table('economic_incentives','Asset-specific medians in the 810-case strict disputed cohort. Payoffs exclude gas and other unmeasured costs.','tab:cr-economic-incentives',
          ['Asset','Cases','Reward/bond','Capital','Days','Token-days','Payoff'],
          [[('USDC.e' if r['asset'].startswith('0x2791') else 'USDC'),r['cases']]+[fmt(r[k],2) for k in ['reward_to_bond_ratio_median','capital_tokens_median','lock_days_median','capital_token_days_median','payoff_tokens_median']] for r in econ],
          ['economics/incentives_and_realized_outcomes.csv'])
    assets={'0x514910771af9ca656af840dff83e8264ecf986ca':'LINK','0x04fa0d235c4abf4bcf4787af4cf447de572ef828':'UMA','0x2791bca1f2de4661ed88a30c99a7a9449aa84174':'USDC.e','0x3c499c542cef5e3811e1192ce70d8cc03d5c3359':'USDC'}
    rewards=[]
    for r in rows('economics/cross_protocol_reward_concentration'):
        mechanism={'Staking v0.2 RewardVault':'RewardVault v0.2','Oracle Integrity Staking':'Integrity Staking','Reporter tips escrow':'Reporter tips','OptimisticOracleV2':'OOV2'}.get(r['reward_mechanism'],r['reward_mechanism'])
        rewards.append([esc(r['protocol']),assets.get(r['asset'],r['asset']),mechanism,r['denominator_events'],r['actors'],pct(r['top1']),pct(r['top10']),fmt(r['hhi'])])
    table('reward_concentration','Observed reward-recipient concentration by protocol, native asset, and mechanism. Event and recipient denominators are retained separately.','tab:cr-reward-concentration',
          ['Protocol','Asset','Mechanism','Events','Recipients',r'Top 1 (\%)',r'Top 10 (\%)','HHI'],rewards,
          ['economics/cross_protocol_reward_concentration.csv'],long=True)
    avail=rows('economics/protocol_variable_availability')
    variables=['observed_reward_recipient_concentration','proposal_and_dispute_activity_concentration','configured_reward_to_bond','capital_lock_duration','realized_challenger_token_payoff']
    states={'available':'A','not_evaluated':'NE','not_available_in_selected_inputs':'U'}
    table('availability','Availability of the five economic measures analyzed in this research.','tab:cr-availability',
          ['Protocol','Reward conc.','Activity conc.','Reward/bond','Lock duration','Challenger payoff'],
          [[esc(s)]+[states[next(r['status'] for r in avail if r['protocol']==s and r['variable']==v)] for v in variables] for s in ['UMA','Chainlink','Flare_FTSOv2','Tellor','Pyth']],
          ['economics/protocol_variable_availability.csv'],
          'A: observed in the selected analysis scope. NE: not evaluated under a common denominator. U: unavailable in selected inputs. Neither NE nor U means zero activity. Reward strata retain their asset, mechanism, and realization-stage distinctions.')
    episode = json.loads((REPO/'data/local/uma_case/economic_episode.json').read_text())
    flows = pd.read_parquet(REPO/'data/local/uma_case/token_flows.parquet')
    scale = Decimal(10) ** episode['asset_decimals']
    oov2 = episode['source_contract']
    proposer, disputer = episode['counterparty'], episode['actor']
    def amount(sender, receiver):
        selected=flows[(flows.sender==sender)&(flows.receiver==receiver)]
        return sum((Decimal(v) for v in selected.amount_raw),Decimal(0))/scale
    def token(field): return Decimal(episode[field])/scale
    fee_recipient='0xe58480ca74f1a819fafd777beded4e2d5629943d'
    # Cross-check episode accounting against independent observed token transfers.
    if amount(oov2,disputer)!=token('gross_payout_raw') or amount(oov2,fee_recipient)!=token('protocol_fee_raw'):
        raise ValueError('Illustrated episode disagrees with its observed transfer records')
    if token('gross_payout_raw')-token('principal_returned_raw')!=token('realized_payoff_raw'):
        raise ValueError('Episode payoff decomposition fails')
    table('uma_cashflow','Verified USDC.e cash flows for the illustrated UMA episode. All values are token amounts.','tab:cr-uma-cashflow',
          ['Flow','Amount','Accounting interpretation'],
          [['Proposer to OOV2',fmt(amount(proposer,oov2),0),'Bond + fee exposure'],
           ['Disputer to OOV2',fmt(amount(disputer,oov2),0),'Bond + fee exposure'],
           ['OOV2 to protocol-fee recipient',fmt(amount(oov2,fee_recipient),0),'Aggregate escrow fee outflow'],
           ['OOV2 to successful disputer',fmt(amount(oov2,disputer),0),'Returned capital + realized gain'],
           ['Capital returned to losing proposer',fmt(amount(oov2,proposer),0),'Losing exposure'],
           ['Configured reward via adapter',fmt(token('reward_configured_raw'),0),'Refunded and re-posted; separate']],
          ['data/local/uma_case/token_flows.parquet','data/local/uma_case/economic_episode.json'],
          'Cash flows are recomputed from the archived transfers. Gas remains in native Polygon units and is not subtracted from USDC.e.')
    write('tables/source_index.json',json.dumps(table_sources,indent=2))
    return table_sources


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=ROOT/'outputs/final')
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    render(a.results,a.out)
