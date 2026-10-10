"""Original publication figures generated only from observed saved results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from oracle_audit.io import save_json,sha256

COLORS={'B0':'#6b7280','B0plus':'#bb6a25','B1':'#3178a7','A1':'#00866c','B2':'#9060a6'}


def save(fig,out,name):
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/(name+'.png'),dpi=240,bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


def calibration(predictions,out):
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharex=True,sharey=True)
    for ax,cal in zip(axes,['raw','platt']):
        ax.plot([0,1],[0,1],ls='--',color='#aaa',lw=1)
        for system,color in COLORS.items():
            g=predictions[(predictions.system==system)&(predictions.calibration==cal)&(predictions.split=='test')].copy()
            if g.empty:continue
            g['bin']=np.minimum((g.p_reject*10).astype(int),9)
            summary=g.groupby('bin').agg(p=('p_reject','mean'),y=('label','mean'),n=('label','size'))
            ax.plot(summary.p,summary.y,'o-',color=color,ms=4,lw=1.2,label=system)
        ax.set(xlabel='Predicted rejection probability',title='Raw probabilities' if cal=='raw' else 'Validation-only Platt calibration',xlim=(0,1),ylim=(0,1))
        ax.grid(alpha=.18);ax.legend(fontsize=8,loc='upper left')
    axes[0].set_ylabel('Observed rejection fraction')
    fig.text(.5,-.025,'10 fixed-width bins; descriptive case-run means. Empty bins omitted; repeats share cases.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,out,'calibration')


def citations(analysis_dir,out):
    frame=pd.read_csv(analysis_dir/'citation_error_rates.csv')
    fields=['unknown_id_rate','future_id_rate','valid_id_not_received_rate','missing_required_rate','output_invalid_rate']
    labels=['Nonexistent ID','Post-cutoff ID','ID without received\nrecord content','Missing required\ncitation','Invalid final\noutput schema']
    fig,ax=plt.subplots(figsize=(9,3.8));x=np.arange(len(fields));width=.75/len(frame)
    for i,row in enumerate(frame.itertuples()):
        ax.bar(x+(i-(len(frame)-1)/2)*width,[getattr(row,f)*100 for f in fields],width,label=row.system,color=COLORS.get(row.system))
    ax.set(xticks=x,xticklabels=labels,ylabel='Case-runs with error (%)',title='Citation and output failures on clean test evidence')
    ax.legend();ax.grid(axis='y',alpha=.2)
    fig.text(.5,-.04,'Run-level errors can overlap. ID validity does not establish semantic support.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,out,'citation_errors')


def costs(analysis_dir,out):
    metrics=pd.read_csv(analysis_dir/'policy_metrics.csv');cost=pd.read_csv(analysis_dir/'cost_accounting.csv')
    chosen=metrics[(metrics.calibration=='raw')&(metrics.policy=='registered')&(metrics.guard=='on')]
    merged=chosen.merge(cost,on='system',validate='one_to_one')
    ci_path=analysis_dir/'performance_confidence_intervals.csv'
    ci=pd.read_csv(ci_path) if ci_path.exists() else pd.DataFrame()
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for row in merged.itertuples():
        for ax,xcol,xlabel in zip(axes,['total_tokens_mean','latency_seconds_mean'],['Mean total tokens per case-run','Mean end-to-end seconds per case-run']):
            value=getattr(row,xcol)
            ax.scatter(value,row.auroc,s=60,color=COLORS.get(row.system),label=row.system)
            if len(ci):
                interval=ci[(ci.system==row.system)&(ci.calibration=='raw')&(ci.metric=='auroc')]
                if len(interval)==1:
                    bounds=interval.iloc[0]
                    ax.errorbar(value,row.auroc,yerr=[[max(0,row.auroc-bounds.ci_low)],[max(0,bounds.ci_high-row.auroc)]],fmt='none',color=COLORS.get(row.system),capsize=3,lw=1)
            ax.annotate(row.system,(value,row.auroc),xytext=(6,5),textcoords='offset points',fontsize=9)
            ax.set(xlabel=xlabel,ylabel='Test AUROC');ax.grid(alpha=.2)
    fig.text(.5,-.035,'Qwen only. Tokens include cache once. Timing includes queues; different run batches are not controlled latency trials.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,out,'cost_performance')


def workflow(out):
    fig,ax=plt.subplots(figsize=(11,4.2));ax.set(xlim=(0,11),ylim=(0,4.2));ax.axis('off')
    boxes=[(.1,2.8,2.0,.9,'810 disputed cases\n489 train / 161 validation\n160 test'),
           (2.55,2.8,2.1,.9,'Frozen evidence\nsource, hash, event time'),
           (5.05,2.8,2.25,.9,'B0 / B0+: features\nB1: supplied packet\nA1: up to 8 tool calls'),
           (7.85,2.8,2.45,.9,'Saved raw predictions\nprobabilities + citations'),
           (5.05,1.05,2.25,.9,'B2: single pass\nexact A1 tool results'),
           (7.85,1.05,2.45,.9,'Validation-only calibration\ncost-derived actions\nexplicit guard override'),
           (2.0,.15,3.8,.6,'Paired case bootstrap; calibration, loss,\ncoverage, selective error, tokens and latency')]
    for x,y,w,h,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',fc='#f0f5f7',ec='#426477',lw=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8.8)
    for start,end in [((2.15,3.25),(2.5,3.25)),((4.7,3.25),(5,3.25)),((7.35,3.25),(7.8,3.25)),
                      ((6.15,2.75),(6.15,2.02)),((7.35,1.5),(7.8,1.5)),((9.07,2.75),(9.07,2.02)),((7.8,1.18),(5.85,.45))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#426477',lw=1.4))
    ax.text(.15,1.3,'Guard experiment:\n40 paired cases, clean + 5 corruptions\nsource admission and validation on/off\nmodel behavior recorded before override',fontsize=9,va='center')
    ax.text(5.05,4.02,'Research evaluation of protocol adjudication; no on-chain interventions',ha='center',fontsize=11)
    save(fig,out,'workflow')


def guards(analysis_dir,out):
    path=analysis_dir/'guard_scenario_metrics.csv'
    if not path.exists():return
    frame=pd.read_csv(path);scenarios=['clean','missing_required','conflicting_records','future_record','prompt_injection','plausible_wrong']
    systems=sorted(frame.system.unique())
    fig,axes=plt.subplots(1,len(systems),figsize=(4.5*len(systems),4.0),sharey=True,squeeze=False)
    for ax,system in zip(axes.ravel(),systems):
        values=frame[frame.system==system].pivot(index='scenario',columns='guard_mode',values='final_auto_rate').reindex(scenarios)[['all_off','on']]
        im=ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        for (i,j),v in np.ndenumerate(values.to_numpy()):ax.text(j,i,f'{v:.0%}',ha='center',va='center',color='white' if v>.55 else 'black')
        ax.set(xticks=[0,1],xticklabels=['Guards off','Guards on'],yticks=range(len(scenarios)),yticklabels=[s.replace('_',' ') for s in scenarios],title=system)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Final automatic-action fraction',shrink=.8)
    case_counts=sorted(frame.cases.unique())
    denominator=str(case_counts[0]) if len(case_counts)==1 else '/'.join(map(str,case_counts))
    fig.suptitle(f'Full paired guard ablation ({denominator} cases per cell)',fontsize=12)
    save(fig,out,'guard_ablation')


def economic(economic_dir,out):
    activity=pd.read_csv(economic_dir/'verification_activity_concentration.csv')
    rewards=pd.read_csv(economic_dir/'cross_protocol_reward_concentration.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for i,(field,label) in enumerate([('top1','Top 1'),('top5','Top 5'),('top10','Top 10')]):
        axes[0].bar(np.arange(len(activity))+(i-1)*.22,activity[field]*100,.22,label=label)
    axes[0].set(xticks=range(len(activity)),xticklabels=activity.role,ylabel='Share of observed requests (%)',title='UMA primary activity by role');axes[0].legend(fontsize=8)
    labels=[f'{r.protocol}: {r.reward_mechanism}\n{str(r.asset)[:12]}; n={r.actors:,} recipients' for r in rewards.itertuples()]
    axes[1].barh(range(len(rewards)),rewards.top1*100,color='#3178a7')
    axes[1].set(yticks=range(len(rewards)),yticklabels=labels,xlabel='Top recipient share within asset (%)',title='Observed reward concentration')
    axes[1].tick_params(axis='y',labelsize=6.5)
    fig.text(.5,-.055,'Recipient addresses are not operator identities. Assets and reward mechanisms remain separate.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,out,'economic_concentration')


def policies_and_budget(analysis_dir,out):
    sensitivity=pd.read_csv(analysis_dir/'cost_sensitivity.csv')
    cap=pd.read_csv(analysis_dir/'capacity_scenarios.csv')
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for system,color in COLORS.items():
        g=sensitivity[(sensitivity.system==system)&(sensitivity.guard=='on')]
        if g.empty:continue
        cal='platt' if 'platt' in set(g.calibration) else 'raw'
        g=g[g.calibration.eq(cal)].sort_values('investigate_cost')
        axes[0].plot(g.investigate_cost,g.decision_loss,'o-',color=color,label=system+' '+cal,ms=4)
        c=cap[(cap.system==system)&(cap.calibration==cal)].sort_values('budget_fraction')
        axes[1].plot(c.budget_fraction,c.decision_loss,'o-',color=color,label=system+' '+cal,ms=4)
    grid=np.array([.05,.1,.15,.2,.3,.4])
    axes[0].plot(grid,np.minimum(grid,.2),'--',color='#999',label='Best constant deferral')
    axes[0].set(xlabel='Assumed terminal Investigate cost',ylabel='Observed decision loss',title='Cost-derived policy, guards on')
    axes[1].set(xlabel='Assumed review fraction of 160-case batch',ylabel='Observed decision loss',title='Capacity scenario; infeasible points omitted')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=6.8)
    fig.tight_layout();save(fig,out,'policy_sensitivity')
    frame=pd.read_csv(analysis_dir/'tool_budget_distribution.csv');g=frame[frame.system.eq('A1')]
    fig,ax=plt.subplots(figsize=(5,3.4))
    ax.bar(g.tool_calls,g.case_runs,color=['#d2853b' if v>=8 else '#00866c' for v in g.tool_calls])
    ax.set(xlabel='Attempted tool calls per A1 case-run',ylabel='Case-runs',xticks=g.tool_calls,title='Eight-call budget saturation')
    for r in g.itertuples():ax.text(r.tool_calls,r.case_runs+3,str(r.case_runs),ha='center',fontsize=8)
    fig.text(.5,-.01,'Attempts beyond eight are denied; successful calls never exceed eight.',ha='center',fontsize=7)
    fig.tight_layout();save(fig,out,'tool_budget')


def render(predictions,analysis_dir,economic_dir):
    analysis_dir=Path(analysis_dir);out=analysis_dir/'figures'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    calibration(predictions,out);citations(analysis_dir,out);costs(analysis_dir,out);workflow(out);guards(analysis_dir,out);economic(Path(economic_dir),out)
    policies_and_budget(analysis_dir,out)
    save_json(out/'figure_manifest.json',[dict(path=p.name,sha256=sha256(p)) for p in sorted(out.glob('*')) if p.suffix in ['.png','.pdf']])
