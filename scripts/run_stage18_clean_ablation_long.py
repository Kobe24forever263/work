#!/usr/bin/env python3
"""Plan or manually launch the final clean-ablation training campaign."""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'scripts/run_stage18_train_one.py'
ALL=('no_position_only','no_eta_cost_only','no_history_only','no_explicit_queue_resource','no_explicit_handover_risk')

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--profiles',nargs='+',choices=('MEDIUM','DENSE','BURST'),default=['MEDIUM','DENSE','BURST'])
 p.add_argument('--conditions',nargs='+',choices=ALL,default=list(ALL))
 p.add_argument('--seed-start',type=int,default=1);p.add_argument('--seed-end',type=int,default=10)
 p.add_argument('--updates',type=int,default=2000);p.add_argument('--execute',action='store_true')
 a=p.parse_args(); jobs=[]
 for profile in a.profiles:
  for condition in a.conditions:
   for seed in range(a.seed_start,a.seed_end+1):
    jobs.append([sys.executable,str(TRAIN),condition,str(seed),'--arrival-profile',profile,'--phase','all','--updates',str(a.updates),'--execute'])
 plan={'stage':18,'campaign':'CLEAN_SEMANTIC_ABLATIONS','profiles':a.profiles,'conditions':a.conditions,'seed_range':[a.seed_start,a.seed_end],'updates_per_run':a.updates,'episodes_per_run':a.updates*4,'job_count':len(jobs),'total_ppo_episodes':len(jobs)*a.updates*4,'execution':'sequential_manual_launcher','claim_boundary':'Training only; fresh locked-test seeds required after model freeze.'}
 print(json.dumps(plan,ensure_ascii=False,indent=2))
 if not a.execute:return 0
 profile_key='_'.join(profile.lower() for profile in a.profiles)
 status=ROOT/'results/stage18_clean_ablation'/f'long_campaign_{profile_key}.status.json';status.parent.mkdir(parents=True,exist_ok=True)
 for index,cmd in enumerate(jobs,1):
  status.write_text(json.dumps({**plan,'state':'RUNNING','job_index':index,'command':cmd,'updated_at':time.time()},ensure_ascii=False,indent=2)+'\n')
  try:
   subprocess.run(cmd,cwd=ROOT,check=True)
  except BaseException as exc:
   status.write_text(json.dumps({**plan,'state':'FAILED','job_index':index,
    'command':cmd,'error':repr(exc),'updated_at':time.time()},
    ensure_ascii=False,indent=2)+'\n')
   raise
 status.write_text(json.dumps({**plan,'state':'COMPLETE','job_index':len(jobs),'updated_at':time.time()},ensure_ascii=False,indent=2)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
