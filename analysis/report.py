"""Readable result register; missing experiments remain visibly incomplete."""
import json
from pathlib import Path
import pandas as pd


def md(frame):
    def value(x):
        if pd.isna(x):return '—'
        if isinstance(x,float):return f'{x:.4f}'
        return str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(value(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])


def write(out):
    out=Path(out);receipt=json.loads((out/'replay_receipt.json').read_text())
    completion=pd.DataFrame([dict(experiment=k,**v) for k,v in receipt['completion'].items()])
    metrics=pd.read_csv(out/'policy_metrics.csv')
    selected=metrics[(metrics.policy=='registered')&(metrics.guard=='on')&metrics.system.isin(['B0','B0plus','B1','A1','B2'])]
    fields=['system','calibration','cases','case_runs','auroc','brier','decision_loss','coverage','selective_error']
    derived=metrics[(metrics.policy=='cost_derived')&(metrics.guard=='on')&metrics.system.isin(['B0','B0plus','B1','A1','B2'])]
    cost=pd.read_csv(out/'cost_accounting.csv')
    cohorts=pd.read_csv(out/'cohort_summary.csv')
    economics=pd.read_csv(out/'economics/verification_activity_concentration.csv')
    outcomes=pd.read_csv(out/'economics/incentives_and_realized_outcomes.csv')
    lexical=pd.read_csv(out/'reasoning_guard_review_summary.csv')
    lines=['# Camera-ready 必做实验结果',
      '\n当前状态：**'+receipt['status']+'**。只将完整实验面板计入报告；试跑、未完成批次和不同模型的历史日志不混入主结果。',
      '\n## 实验完成核对\n',md(completion),
      '\n原始 Qwen 主实验为 160 个测试案例 × 5 次 × B1/A1，共 1,600 次；另有原始扰动试验 320 次，均复用已有记录。用户于 2026-10-09 要求只做必要实验，补充范围缩减为验证集 483 次、已有 B2 匹配证据 320 次、A1 防护实验 480 次，共 1,283 次。低温度补充实验取消，不能记为已完成。所有失败保留在分母中；无有效概率时沿用原程序的 0.5 回退，并单独标记。',
      '\n**缩减规则：** B2 使用已经完整完成的 repeat 3、4（编号从 0 起），配对 A1 使用相同案例和相同两轮；原 B1/A1 五轮主结果保持独立。选择依据是完成状态和节省计算，不依据性能；这是实验开始后的资源约束修订，并非事前预注册的两轮设计，不能等同于五轮证据强度。安全检查只对 A1 做六场景、开关对照，不据此声称 B1 消融效果。额外完成记录保留在 `*_runs_outside_scope.csv`，见 `manifests/experiment-scope-20261009.json`。',
      '\n## 模型与论文数字冲突\n',
      '本地旧 `Results/AgenticOracleAuditor` 日志的实际模型为 gpt-5.6-sol。服务器上找到了独立的 Qwen3.8-27B 正式实验；已校验 14,345 个实验文件，以及 34 个当前 checkpoint 文件。主文的模型名称、性能、token 和成本必须统一到同一套实际日志，不能只改模型名。',
      '本地模型卡名称为 Qwen3.8-27B，配置使用 Qwen3_5ForConditionalGeneration 架构和 BF16 文本权重。下载元数据只有可变的 master 标记，未核实公共不可变 commit；本次通过逐文件 SHA-256 固定实际权重，而不是将架构名称当作模型版本。',
      '\n下表为原阈值、保留防护的结果。`B0plus` 为固定配置 LightGBM，使用与 B0 相同的 16 个可用特征。raw/platt 分开报告；若验证集补跑未完成，不生成 LLM 校准结果。B1/A1 为五轮，B2 为两轮；严格配对比较另见下方两轮对照表。\n',
      md(selected[fields]),
      '\n## 决策损失修正\n',
      '正确自动决策成本为 0，错误为 1，Investigate 为 0.1，Abstain 为 0.2。在这套终止式损失下，始终 Investigate 的损失严格为 0.1、自动覆盖率为 0；不包含后续人工判断的成本或错误。校准概率下，p<0.1 时 Accept、p>0.9 时 Challenge，中间 Investigate；边界相等时保守推迟决定。防护强制弃权是额外约束，不是该损失矩阵的无约束最优动作。',
      '\n保留同一模型原有 guard 标记的事后延迟基线，损失为 0.1+0.1q；q 包括该模型的输出/引用契约失败。这与只检查干净输入的无模型基线不同。提高 Investigate 成本超过 0.2 时，Abstain 会成为最优中间动作。完整敏感性表见 `cost_sensitivity.csv`。\n',
      md(derived[fields]),
      '\n## 不确定性\n',
      '配对 bootstrap 以案例为单位重抽样 2,000 次，保留该案例的全部重复运行；800 次运行不能当成 800 个独立案例。区间未校正多重比较。零自动覆盖时 selective error 未定义，保留为空。',
      '\n## 队列、证据和污染边界\n',md(cohorts),
      '\n64 个公开演示案例全部属于训练集；与验证集和测试集无重叠。实验目标是在已被争议、具有严格链接的案例中预测 UMA 裁决，不能推广为对所有请求或独立事实真假的验证。',
      '\n输入包含请求自然语言、ancillary 文本、提案、历史角色统计和可用的市场价格特征。全部 16 个结构化特征、810 个标签与时间划分已从固定来源重建；795 个市场记录已从原始价格点重算。历史数据的事后下载时间不等于提案时已发布的证明。模型训练截止日期未核实；时间过滤和提示词均不能排除参数化记忆。',
      '\n防护词法标记的 76 次输出已进行一次 AI 辅助上下文核查。以下不是独立专家盲审，也不估计漏报率；记忆表述不证明所声称事实真实或确在训练集中。原预测未按此核查事后修改。\n',md(lexical),
      '\n## 引用与防护\n',
      '“收到完整证据内容”与“最终答案引用完整”分开统计，见 `citation_error_rates.csv`。B1 原本就收到预装证据；引用完整率不能直接称为检索获得证据的提升。',
      '引用分类区分不存在 ID、晚于截止时间、ID 有效但未收到完整内容、缺少必需引用和无效输出格式。不能把 ID/哈希检查通过写成语义支持成立。原输出协议没有逐主张的证据蕴含标注，故不虚构 valid-but-irrelevant 比例。干净输入的全部必需记录在 160/160 测试案例上通过结构校验；模型生成错误导致的弃权不自动算防护误报。',
      '\n新的 A1 全流程消融包括 clean、missing_required、conflicting_records、future_record、prompt_injection、plausible_wrong，40 个固定案例 × 六场景 × 开关两种状态，各一次。后者在截止时间内改变历史事实并重算内部哈希，代表上游快照本身不可靠的威胁；它不同于篡改独立可信哈希下的数据。分别保存模型原始动作、按概率生成的动作和防护后的动作。',
      '实际暴露以模型请求上下文为准：工具已返回但未进入下一轮请求的记录不算模型已收到。未来 ID 引用包括只从目录获知 ID、未读取内容的情况。提示注入防护按固定测试字符串构造，不支持对未知攻击的检测能力主张；完整照搬恶意目标的比例与任意自动动作变化分开报告。',
      '\n## Token 与延迟\n',
      md(cost[[c for c in ['system','unique_cases','case_runs','input_tokens_mean','cached_input_tokens_mean','output_tokens_mean','total_tokens_mean','latency_seconds_mean','latency_seconds_p95','cap_rate','failed_runs'] if c in cost]]),
      '\n总 token = 输入 + 输出；缓存已包含在输入中，不能重复相加。Qwen 自托管美元成本未测量，不能用 0 或 GPT API 估算金额代替。端到端时间含队列和检索，旧批次与新批次不是受控的延迟比较。主补跑使用 4×RTX 5090 / TP4，防护消融使用独立 2×RTX 5090 / TP2；防护整组使用相同服务配置。恢复阶段启用 CUDA graphs 并限制 CPU 库线程数，具体配置与分组成本见 `manifests/recovery-serving-runtime.json` 和 `serving_runtime_cohorts.csv`。',
      '\n## 经济分析\n',md(economics),
      '\n分资产、分角色的挑战资本和观察到的 token 收益：\n',md(outcomes[[c for c in ['asset','cases','positive_payoff_cases','negative_payoff_cases','reward_to_bond_ratio_median','capital_tokens_median','capital_token_days_median','payoff_tokens_median','lock_days_median'] if c in outcomes]]),
      '\n五协议的奖励接收地址集中度和所分析的五个变量可用性分别见 `economics/cross_protocol_reward_concentration.csv` 与 `economics/protocol_variable_availability.csv`。地址不是实际控制人；不同资产、奖励机制和生命周期阶段不相加，缺失不是零。收益统计不含未测量的 gas、风险和机会成本，也不是因果激励效果。',
      '\n## 可复现产物与交付边界\n',
      '完整离线复现：`python -m analysis.replay --retrain`。进行中检查：`python -m oracle_audit experiment-check`。原始调用、输入消息、响应、工具记录和失败原因保存在 `outputs/qwen_supplement` 和 `outputs/guard_experiment`；清单与权重哈希在 `manifests/`。图表脚本为 `figures/render.py`。',
      '\n公共数据 revision、descriptor arXiv 版本、作者许可、科学审阅及真实 hosted Colab 测试仍需各自的真实回执；本地实验结果不等于这些发布条件通过。']
    if (out/'infrastructure_scope.json').exists():
        scope=json.loads((out/'infrastructure_scope.json').read_text())
        # Make the recovery amendment prominent and remove the superseded ITT claim.
        lines=[line.replace('所有失败保留在分母中；无有效概率时沿用原程序的 0.5 回退，并单独标记。',
              '正常返回的无效模型输出保留在固定分母中；无有效概率时沿用原程序的 0.5 回退。基础设施失败另行归档，恢复规则见下述说明。') for line in lines]
        lines.insert(2,'\n**基础设施恢复说明：** 2026-10-09 的调度中断与超时产生了 '+str(scope['archived_attempts'])+
          ' 个归档尝试目录。主补充实验按超时/中断选择恢复；防护消融整组在同一修复后的运行配置下重跑，原先正常返回的防护结果也归档保留。选择不依据答案质量。单次超时也可能来自模型轨迹过长，不能全部归因于网络故障。当前补充面板不能称为原始首次尝试分析。详见 `infrastructure_attempt_audit.csv`。主成本表只统计当前面板；归档请求的已记录 token 另报，未返回用量的计算成本未知。\n')
    ci=out/'paired_bootstrap.csv'
    if (out/'scope_reduction_accounting.json').exists():
        amendment=json.loads((out/'scope_reduction_accounting.json').read_text())
        lines.insert(3,'\n**排程缩减记录：** 停止旧排程时有 '+str(amendment['interrupted_attempts'])+
            ' 个未完成尝试，原文件已单独归档，见 `scope_interrupted_attempts.csv`；未返回的 token 用量未知。已有有效结果未删除，也未依据答案好坏筛选。\n')
    if (out/'matched_evidence_comparison_metrics.csv').exists():
        frame=pd.read_csv(out/'matched_evidence_comparison_metrics.csv')
        keep=frame.policy.eq('registered')&frame.guard.eq('on')
        lines.extend(['\n## 同案例、同两轮的 B2/A1 对照\n',md(frame.loc[keep,fields+['repeat_ids']])])
    if ci.exists():
        frame=pd.read_csv(ci);keep=frame.metric.isin(['auroc','decision_loss','coverage'])
        lines.extend(['\n## 配对差值与 95% 区间\n',md(frame.loc[keep,['left','right','calibration','policy','metric','difference','ci_low','ci_high','cases']])])
    if (out/'guard_scenario_metrics.csv').exists():
        frame=pd.read_csv(out/'guard_scenario_metrics.csv')
        lines.extend(['\n## 完整防护消融结果\n',md(frame[['scenario','system','guard_mode','cases','raw_auto_rate','final_auto_rate','final_abstain_rate','future_exposure_rate','future_citation_rate']])])
    names=['workflow','calibration','citation_errors','cost_performance','policy_sensitivity','tool_budget','economic_concentration','guard_ablation']
    lines.extend(['\n## 原创结果图\n']+[f'\n![{name}](figures/{name}.png)\n' for name in names if (out/'figures'/f'{name}.png').exists()])
    (out/'RESULTS_zh.md').write_text('\n'.join(lines)+'\n')
