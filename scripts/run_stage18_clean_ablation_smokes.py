#!/usr/bin/env python3
"""Run short safety/stability gates for five clean ablations per load."""
from __future__ import annotations
import argparse
import json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'scripts/run_stage14_smdp_ppo.py'
BASE_OUT=ROOT/'results/stage18_clean_ablation/smoke'
VARIANTS={
 'no_position_only':'NO_POSITION_ONLY',
 'no_eta_cost_only':'NO_ETA_COST_ONLY',
 'no_history_only':'NO_HISTORY_ONLY',
 'no_explicit_queue_resource':'NO_EXPLICIT_QUEUE_RESOURCE',
 'no_explicit_handover_risk':'NO_EXPLICIT_HANDOVER_RISK'}
REQUIRED=(
 'ppo_parameters_updated','training_metrics_are_finite','discount_contract_is_valid',
 'evaluation_resolved_all_tasks','evaluation_actions_all_legal',
 'evaluation_has_no_resource_leak','one_policy_exposes_all_three_transport_modes',
 'context_conditioned_metrics_cover_every_assignment','rule_baseline_has_no_resource_leak',
 'concurrent_mode_overlaps_when_requested')
SEED_BASE={'MEDIUM':47000000,'DENSE':52000000,'BURST':53000000}

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--profiles',nargs='+',choices=('MEDIUM','DENSE','BURST'),default=['MEDIUM'])
 args=parser.parse_args(); rows=[]
 for profile in args.profiles:
  out=BASE_OUT/profile.lower();out.mkdir(parents=True,exist_ok=True)
  for i,(name,variant) in enumerate(VARIANTS.items()):
   weight=out/f'{name}.pt'
   cmd=[sys.executable,str(TRAIN),'--execution-mode','CONCURRENT','--arrival-profile',profile,'--observation-variant',variant,'--updates','5','--episodes-per-update','2','--ppo-epochs','2','--minibatch-size','16','--eval-episodes','5','--eval-every','5','--checkpoint-every','5','--log-every','5','--seed',str(SEED_BASE[profile]+i*100),'--validation-seed-start',str(SEED_BASE[profile]+10000),'--output',str(weight)]
   completed=subprocess.run(cmd,cwd=ROOT,check=False)
   summary=json.loads(weight.with_suffix('.summary.json').read_text()); assertions=summary.get('assertions',{})
   rows.append({'profile':profile,'condition':name,'variant':variant,'summary':str(weight.with_suffix('.summary.json')),'process_exit_code':completed.returncode,'hard_passed':all(assertions.get(k) is True for k in REQUIRED),'context_mode_reference_gate':assertions.get('formal_policy_switches_modes_by_context'),'parameter_delta_l2':summary['parameter_delta_l2'],'illegal_action_count':summary['evaluation']['illegal_action_count'],'resource_leak_count':summary['evaluation']['resource_leak_count']})
 checks={'all_requested_conditions_completed':len(rows)==len(args.profiles)*5,'all_hard_safety_and_stability_gates_passed':all(x['hard_passed'] for x in rows),'all_parameters_updated':all(x['parameter_delta_l2']>0 for x in rows),'no_illegal_actions':all(x['illegal_action_count']==0 for x in rows),'no_resource_leaks':all(x['resource_leak_count']==0 for x in rows)}
 report={'stage':18,'gate':'CLEAN_ABLATION_HIGH_LOAD_SHORT_TRAINING','claim_boundary':'Five-update safety/stability smoke; no performance claim.','profiles':args.profiles,'conditions':rows,'assertions':checks,'semantic_note':'Cost-reference agreement is recorded but not a safety/stability gate; it can fail by design for NO_ETA_COST_ONLY.','passed':all(checks.values())}
 path=BASE_OUT/'high_load_short_training_gate.summary.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'output':str(path),'passed':report['passed'],'condition_count':len(rows)},ensure_ascii=False,indent=2)); return 0 if report['passed'] else 1
if __name__=='__main__': raise SystemExit(main())
