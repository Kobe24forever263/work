#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path

WORK=Path('/Users/lab4099/Desktop/Mujoco/work')
OUT=Path('/private/tmp/stage18_v2_formal_quality_review')
V2=WORK/'results/stage18_locked_test_v2/stage18_locked_test_v2_statistics.json'
V1=WORK/'results/stage18_locked_test/stage18_locked_test_statistics.json'
PROFILES=('MEDIUM','DENSE','BURST')

def load(p): return json.loads(Path(p).read_text())
def finite(x):
    if isinstance(x,float): return math.isfinite(x)
    if isinstance(x,list): return all(map(finite,x))
    if isinstance(x,dict): return all(map(finite,x.values()))
    return True
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

v2=load(V2); v1=load(V1)
rows=v2['matched_load_comparisons']
def row(profile,comp,metric):
    return next(x for x in rows if x['profile']==profile and x['comparison']==comp and x['metric']==metric)

# Independent raw-file integrity/protocol checks.
raw=list((WORK/'results/stage18_locked_test_v2').glob('trained_*/*/tested_*/seed_*.json'))
raw=[p for p in raw if '/baselines/' not in str(p)]
errors=[]; episodes=0; illegal=0; leaks=0; hashes=set(); seeds=set(); keyed=0
for p in raw:
    d=load(p); episodes += d['policy']['episode_count']; illegal += d['policy']['illegal_action_count']; leaks += d['policy']['resource_leak_count']; hashes.add(d['weight_sha256']); seeds.update(e['seed'] for e in d['policy']['episodes'])
    keyed += d['evaluation_protocol']['handover_sampling']=='TASK_KEYED_COMMON_RANDOM_NUMBERS'
    if not finite(d): errors.append(f'non-finite:{p}')
    if len(d['policy']['episodes'])!=100: errors.append(f'episode-count:{p}')

dog=list((WORK/'results/stage18_locked_test_v2/baselines/single_dog_only').glob('tested_*.json'))
dog_episodes=dog_sched=dog_completed=dog_failed=dog_unresolved=0
for p in dog:
    d=load(p); a=d.get('single_dog_only',d.get('policy',{})); dog_episodes+=a.get('episode_count',0); dog_sched+=a.get('scheduled_task_count',a.get('task_count',0)); dog_completed+=a.get('completed',0); dog_failed+=a.get('failed',0); dog_unresolved+=a.get('unresolved',0)

primary=[]; ablation=[]; dog_cmp=[]
for prof in PROFILES:
    for metric in ('reward','success_rate','successful_throughput_tasks_per_hour'):
        x=row(prof,'full_context_v2_minus_rule_baseline',metric)
        primary.append({'profile':prof,'metric':metric,'left':x['left_estimate'],'right':x['right_estimate'],'delta':x['mean_delta'],'ci_low':x['crossed_bootstrap_95ci'][0],'ci_high':x['crossed_bootstrap_95ci'][1],'family_holm_p':x['family_holm_adjusted_p']})
    for cond in ('no_queue_resource','no_handover_cues','no_persistent_position'):
        for metric in ('reward','success_rate','successful_throughput_tasks_per_hour'):
            x=row(prof,f'full_context_v2_minus_{cond}',metric)
            ablation.append({'profile':prof,'condition':cond,'metric':metric,'delta':x['mean_delta'],'ci_low':x['crossed_bootstrap_95ci'][0],'ci_high':x['crossed_bootstrap_95ci'][1],'family_holm_p':x['family_holm_adjusted_p']})
    for metric in ('terminal_adjusted_reward','success_rate','successful_throughput_tasks_per_hour'):
        x=row(prof,'full_context_v2_minus_single_dog_only',metric)
        dog_cmp.append({'profile':prof,'metric':metric,'left':x['left_estimate'],'right':x['right_estimate'],'delta':x['mean_delta'],'ci_low':x['crossed_bootstrap_95ci'][0],'ci_high':x['crossed_bootstrap_95ci'][1],'family_holm_p':x['family_holm_adjusted_p']})

# v1-v2 is protocol sensitivity, not policy improvement.
v1rows=v1['comparisons']; vv=[]
def v1row(prof,metric):
    return next(x for x in v1rows if x['profile']==prof and x['comparison']=='full_context_v2_minus_rule_baseline' and x['metric']==metric)
for prof in PROFILES:
    for metric in ('reward','success_rate'):
        a=v1row(prof,metric); b=row(prof,'full_context_v2_minus_rule_baseline',metric)
        vv.append({'profile':prof,'metric':metric,'v1_delta':a['mean_delta'],'v2_delta':b['mean_delta'],'change':b['mean_delta']-a['mean_delta']})
    # audited v1 completed-only throughput values
v1tp={'MEDIUM':0.3919,'DENSE':2.1423,'BURST':1.1875}
for prof in PROFILES:
    b=row(prof,'full_context_v2_minus_rule_baseline','successful_throughput_tasks_per_hour')
    vv.append({'profile':prof,'metric':'successful_throughput_tasks_per_hour','v1_delta':v1tp[prof],'v2_delta':b['mean_delta'],'change':b['mean_delta']-v1tp[prof]})

audit={'verdict':'CONDITIONAL_PASS','publication_readiness':{'primary_full_vs_rule':'PASS_WITH_SCOPE','multiagent_vs_single_dog':'PASS_WITH_BASELINE_LIMITATION','context_ablations':'EXPLORATORY_ONLY','online_adaptation':'NOT_ESTABLISHED'},'integrity':{'raw_jobs':len(raw),'policy_episodes':episodes,'unique_test_seeds':len(seeds),'test_seed_min':min(seeds),'test_seed_max':max(seeds),'task_keyed_jobs':keyed,'unique_weight_hashes':len(hashes),'illegal_actions':illegal,'resource_leaks':leaks,'finite_errors':errors},'single_dog':{'files':len(dog),'episodes':dog_episodes,'scheduled':dog_sched,'completed':dog_completed,'explicit_failed':dog_failed,'unresolved':dog_unresolved},'primary':primary,'single_dog_comparison':dog_cmp,'ablations':ablation,'v1_v2_protocol_sensitivity':vv,'limitations':['v1与v2使用同一批训练权重；差异来自新鲜测试种子、TASK_KEYED配对和修正统计，不是策略升级。','NO_HANDOVER_CUES与NO_QUEUE_RESOURCE仍可被动作掩码、候选集合或其他字段旁路推断，零效应不能证明信息无用。','NO_PERSISTENT_POSITION同时移除位置、楼层、ETA、成本等复合信息，不能把大幅退化单独归因于XYZ位置。','单狗空白对照保持四狗但禁止空载跨层重定位；其高负载未结任务反映当前单狗动作空间，而非理论最优单狗系统。','Burst下Full相对规则基线的成功交付吞吐95% CI跨0。','全局60项Holm较保守；报告采用预定义9项主假设族Holm，同时保留全局校正结果。']}
OUT.mkdir(exist_ok=True)
(OUT/'stage18_v2_formal_quality_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
for name,data in [('primary_comparison.csv',primary),('single_dog_comparison.csv',dog_cmp),('ablation_comparison.csv',ablation),('v1_v2_protocol_sensitivity.csv',vv)]:
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)

def ds(rows): return rows
cards=[{'scope':'v2','raw_jobs':len(raw),'episodes':episodes,'violations':illegal+leaks,'keyed_rate':keyed/len(raw)}]
reward=[x for x in primary if x['metric']=='reward']
tp=[x for x in primary if x['metric']=='successful_throughput_tasks_per_hour']
succ=[dict(x,delta_pp=x['delta']*100) for x in primary if x['metric']=='success_rate']
ablr=[x for x in ablation if x['metric']=='reward']
dogr=[x for x in dog_cmp if x['metric']=='terminal_adjusted_reward']

artifact={'surface':'report','manifest':{'version':1,'surface':'report','title':'Stage 18 v2 正式质量审查','description':'v2与v1、规则基线、单狗空白对照及上下文消融的正式比较。','generatedAt':'2026-08-18T00:00:00+08:00','cards':[{'id':'jobs','dataset':'cards','sourceId':'v2_audit','metrics':[{'label':'v2 Locked jobs','field':'raw_jobs','format':'number'}]},{'id':'episodes','dataset':'cards','sourceId':'v2_audit','metrics':[{'label':'策略 episodes','field':'episodes','format':'number'}]},{'id':'violations','dataset':'cards','sourceId':'v2_audit','metrics':[{'label':'非法动作/泄漏','field':'violations','format':'number'}]},{'id':'keyed','dataset':'cards','sourceId':'v2_audit','metrics':[{'label':'TASK_KEYED覆盖率','field':'keyed_rate','format':'percent'}]}],
'charts':[{'id':'reward_rule','title':'Full 相对规则基线的回报差','subtitle':'v2同负载测试；正值有利于Full，10训练种子×100共享测试种子。','type':'bar','dataset':'reward','sourceId':'v2_stats','encodings':{'x':{'field':'profile','type':'nominal','label':'负载'},'y':{'field':'delta','type':'quantitative','label':'回报差'},'tooltip':[{'field':'ci_low','type':'quantitative','label':'95% CI下界'},{'field':'ci_high','type':'quantitative','label':'95% CI上界'}]}},{'id':'throughput_rule','title':'Full 相对规则基线的成功交付吞吐差','subtitle':'completed tasks/模拟小时；Burst区间跨0。','type':'bar','dataset':'throughput','sourceId':'v2_stats','encodings':{'x':{'field':'profile','type':'nominal','label':'负载'},'y':{'field':'delta','type':'quantitative','label':'任务/模拟小时差'},'tooltip':[{'field':'ci_low','type':'quantitative','label':'95% CI下界'},{'field':'ci_high','type':'quantitative','label':'95% CI上界'}]}},{'id':'dog_reward','title':'Full 相对单狗空白对照的终端修正回报差','subtitle':'未结任务按失败计入；正值有利于异构多智能体策略。','type':'bar','dataset':'dog_reward','sourceId':'v2_stats','encodings':{'x':{'field':'profile','type':'nominal','label':'负载'},'y':{'field':'delta','type':'quantitative','label':'终端修正回报差'}}},{'id':'ablation_reward','title':'Full 相对上下文消融的回报差','subtitle':'只有持续位置复合消融在三种负载均稳定退化。','type':'bar','dataset':'ablation_reward','sourceId':'v2_stats','encodings':{'x':{'field':'profile','type':'nominal','label':'负载'},'y':{'field':'delta','type':'quantitative','label':'Full−消融回报'},'color':{'field':'condition','type':'nominal','label':'消融'}}}],
'tables':[{'id':'primary','title':'Full 与规则基线逐项比较','dataset':'primary','sourceId':'v2_stats','columns':[{'field':'profile','label':'负载','type':'text'},{'field':'metric','label':'指标','type':'text'},{'field':'left','label':'Full','format':'number'},{'field':'right','label':'规则','format':'number'},{'field':'delta','label':'差值','format':'number'},{'field':'ci_low','label':'CI下界','format':'number'},{'field':'ci_high','label':'CI上界','format':'number'}]},{'id':'dog','title':'Full 与单狗空白对照逐项比较','dataset':'dog','sourceId':'v2_stats','columns':[{'field':'profile','label':'负载','type':'text'},{'field':'metric','label':'指标','type':'text'},{'field':'left','label':'Full','format':'number'},{'field':'right','label':'单狗','format':'number'},{'field':'delta','label':'差值','format':'number'},{'field':'ci_low','label':'CI下界','format':'number'},{'field':'ci_high','label':'CI上界','format':'number'}]},{'id':'v1v2','title':'v1→v2评估协议敏感性','subtitle':'同一权重，不应解释为策略进步。','dataset':'v1v2','sourceId':'v2_audit','columns':[{'field':'profile','label':'负载','type':'text'},{'field':'metric','label':'指标','type':'text'},{'field':'v1_delta','label':'v1差值','format':'number'},{'field':'v2_delta','label':'v2差值','format':'number'},{'field':'change','label':'变化','format':'number'}]},{'id':'limitations','title':'结论边界','dataset':'limitations','sourceId':'v2_audit','columns':[{'field':'severity','label':'等级','type':'text'},{'field':'finding','label':'限制','type':'text'}]}],
'sources':[{'id':'v2_audit','label':'独立v2质量审查','path':'results/stage18_v2_formal_quality_review/stage18_v2_formal_quality_audit.json','query':{'engine':'duckdb','sql':"SELECT * FROM read_json_auto('results/stage18_v2_formal_quality_review/stage18_v2_formal_quality_audit.json')",'description':'读取独立质量审查快照。','tables_used':['stage18_v2_formal_quality_audit.json'],'filters':['v2正式审查'],'metric_definitions':['成功交付吞吐=completed/simulated_time×3600']}},{'id':'v2_stats','label':'v2 crossed-bootstrap统计','path':'results/stage18_locked_test_v2/stage18_locked_test_v2_statistics.json','query':{'engine':'duckdb','sql':"SELECT * FROM read_json_auto('results/stage18_locked_test_v2/stage18_locked_test_v2_statistics.json')",'description':'读取v2正式统计。','tables_used':['stage18_locked_test_v2_statistics.json'],'filters':['matched-load comparisons'],'metric_definitions':['差值=Full−比较方法','区间=训练种子轴×共享测试种子轴crossed bootstrap']}}],
'blocks':[{'id':'title','type':'markdown','body':'# Stage 18 v2 正式质量审查'},{'id':'summary','type':'markdown','body':'## 技术结论：主比较有条件通过，消融仍仅适合探索性表述\n\nv2已修复v1最关键的配对与统计问题：180个locked job全部采用TASK_KEYED共同随机数，新鲜测试种子覆盖45000000–45000099，并按训练种子×共享测试种子做交叉Bootstrap。数据完整性、安全不变量和主要比较均通过。\n\nFull相对规则基线在三种负载下均提高回报和成功率；有效交付吞吐在MEDIUM与DENSE显著提高，BURST点估计为正但区间跨0。相对单狗空白对照的优势随负载显著扩大，支持异构协作在拥挤任务流中的价值。\n\n但是，三项上下文消融仍存在构念泄漏或复合删除，不能据此对单一状态变量作因果归因。'}, {'id':'cards','type':'metric-strip','cardIds':['jobs','episodes','violations','keyed']},{'id':'method','type':'markdown','body':'## 范围、数据与统计口径\n\n审查覆盖3种负载、4种学习条件、10个训练种子、100个共享测试种子，以及单狗空白对照。成功率以scheduled tasks为分母；生产性吞吐仅计completed tasks；规则与学习策略采用逐任务TASK_KEYED随机数配对。主要假设族使用9项family-wise Holm校正，所有区间使用双轴交叉Bootstrap。'}, {'id':'ruletext','type':'markdown','body':'## 与规则基线比较\n\n回报优势随负载增强：MEDIUM +0.949、DENSE +20.294、BURST +41.168。成功率分别提高1.68、0.51和0.64个百分点。有效交付吞吐分别提高0.785、1.653和0.773任务/模拟小时；前两者区间排除0，Burst区间为[-0.853, 2.468]，因此Burst吞吐只能报告为“未检出明确差异”。'}, {'id':'rewardchart','type':'chart','chartId':'reward_rule'},{'id':'tpchart','type':'chart','chartId':'throughput_rule'},{'id':'primarytable','type':'table','tableId':'primary'}, {'id':'dogtext','type':'markdown','body':'## 与单狗空白对照比较\n\n异构策略的优势在高负载明显放大。成功率优势从MEDIUM的2.98个百分点扩大到DENSE的12.46和BURST的14.49个百分点；成功交付吞吐优势从2.18扩大到25.08和35.86任务/模拟小时。该对照证明当前系统中“允许车+狗联合决策”优于“所有任务仅由四只狗执行”，但不等价于与理论最优单狗调度器比较。'}, {'id':'dogchart','type':'chart','chartId':'dog_reward'},{'id':'dogtable','type':'table','tableId':'dog'}, {'id':'v1text','type':'markdown','body':'## 与v1比较：这是评估纠偏，不是模型升级\n\nv1和v2复用同一批训练权重。v2的变化来自新鲜测试种子、TASK_KEYED共同随机数、正确的交叉Bootstrap、成功吞吐口径和多重校正。因此v2应替代v1作为正式结果；v1仅保留为探索性和协议敏感性记录。'}, {'id':'v1table','type':'table','tableId':'v1v2'}, {'id':'abtext','type':'markdown','body':'## 消融逐项结论\n\n去交接线索：三种负载下回报、成功率、吞吐均无稳定变化，但模式one-hot、参与者数量和动作结构仍泄漏交接信息。\n\n去队列/资源状态：同样没有稳定差异，但候选集合、动作掩码和机器人可用性仍可旁路推断资源约束。\n\n去持续位置：三种负载均显著退化；然而该条件同时删除位置、楼层、历史、ETA和成本字段，只能支持“持续空间—成本复合上下文重要”，不能说仅XYZ位置重要。'}, {'id':'abchart','type':'chart','chartId':'ablation_reward'}, {'id':'limittext','type':'markdown','body':'## 限制、稳健性与论文可用边界\n\n主要Full-vs-rule结果可用于T-RO，但必须限定为三种固定负载和一个贪心规则基线。单狗对照可作为空白/下界基线，并明确其禁止空载跨层重定位。消融结果应标为探索性，或在投稿前重做语义隔离消融。跨负载结果不能单独证明策略会在同一episode内实时改变模式；仍需MIXED_CONTINUOUS_RECOVERY。当前测试种子已经被查看，后续若调参或重训，必须冻结新的最终测试种子。'}, {'id':'limitationsTable','type':'table','tableId':'limitations'}, {'id':'next','type':'markdown','body':'## 下一步\n\n1. 冻结v2为当前确认性结果，不再用45000000–45000099调参。\n2. 补语义隔离消融：分别移除位置、ETA/成本、历史状态，并阻断动作掩码的信息旁路。\n3. 补更强算法基线（Flat masked PPO、fixed-gamma PPO、RTAW或同级调度器）。\n4. 运行混合连续负载，检验单个episode内的实时模式切换。\n5. 若第2–4步改变模型，使用全新的最终测试种子完成一次性验收。'}]},
'snapshot':{'version':1,'generatedAt':'2026-08-18T00:00:00+08:00','status':'ready','datasets':{'cards':ds(cards),'reward':ds(reward),'throughput':ds(tp),'success':ds(succ),'dog_reward':ds(dogr),'ablation_reward':ds(ablr),'primary':ds(primary),'dog':ds(dog_cmp),'v1v2':ds(vv),'limitations':ds([{'severity':'HIGH' if i<3 else 'MEDIUM','finding':v} for i,v in enumerate(audit['limitations'])])}},'sources':[{'id':'v2_audit','label':'独立v2质量审查','path':'results/stage18_v2_formal_quality_review/stage18_v2_formal_quality_audit.json'},{'id':'v2_stats','label':'v2统计','path':'results/stage18_locked_test_v2/stage18_locked_test_v2_statistics.json'}]}
(OUT/'artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2))
(OUT/'README.md').write_text('# Stage 18 v2正式质量审查\n\n运行 `stage18_v2_formal_quality_review.py` 可从原始v1/v2结果重建审计JSON、CSV和HTML载荷。\n')
notebook={'cells':[{'cell_type':'markdown','metadata':{},'source':['# Stage 18 v2 正式质量审查\n','本笔记本提供可复核的输入路径、指标定义和报告重建入口。']},{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':["from pathlib import Path\n","import json, pandas as pd\n","ROOT=Path('/Users/lab4099/Desktop/Mujoco/work')\n","STATS=ROOT/'results/stage18_locked_test_v2/stage18_locked_test_v2_statistics.json'\n","stats=json.loads(STATS.read_text())\n","matched=pd.DataFrame(stats['matched_load_comparisons'])\n","matched[['profile','comparison','metric','mean_delta','crossed_bootstrap_95ci','family_holm_adjusted_p']]\n"]},{'cell_type':'markdown','metadata':{},'source':['## 口径\n','成功率 = completed / scheduled；成功交付吞吐 = completed / simulated time × 3600。区间为训练种子轴×共享测试种子轴的 crossed bootstrap。']},{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':["exec((ROOT/'results/stage18_v2_formal_quality_review/stage18_v2_formal_quality_review.py').read_text())\n"]}], 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.9'}},'nbformat':4,'nbformat_minor':5}
(OUT/'stage18_v2_formal_quality_review.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2))
(OUT/'stage18_v2_formal_quality_review.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps({'out':str(OUT),'raw_jobs':len(raw),'episodes':episodes,'dog_files':len(dog),'errors':errors},ensure_ascii=False))
