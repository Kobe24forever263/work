#!/usr/bin/env python3
"""Fair serial/concurrent rule-baseline comparison for Stage 14."""
import json, statistics, sys
from collections import Counter
from pathlib import Path

ROOT = Path("/Users/lab4099/Desktop/Mujoco/work")
sys.path.insert(0, str(ROOT / "src" / "warehouse_core"))
from warehouse_core.handover_timing import nearest_rank_percentile
from warehouse_core.persistent_dispatch import Assignment
from warehouse_core.stage14_training import DENSE_INTERVAL, MEDIUM_INTERVAL, WarehouseDispatchGymEnv

def run_episode(seed, mode, interval):
    env = WarehouseDispatchGymEnv(seed=seed, execution_mode=mode, arrival_interval=interval)
    env.reset(seed=seed); dispatch = env.dispatch
    arrivals = {x.task.task_id: x.arrival_time for x in [*dispatch.queue.waiting, *dispatch.queue.pending_arrivals]}
    rows, reward, peak, blocked, decisions = {}, 0.0, 0, 0, 0
    done = False
    while not done:
        index = env.rule_action(); decoded = env.decision.action_ids[index]
        started, had_waiting = dispatch.now, bool(dispatch.queue.waiting)
        if isinstance(decoded, Assignment):
            rows[decoded.task_id] = {
                "seed": seed, "execution_mode": mode, "task_id": decoded.task_id,
                "task_type": env.task_types[decoded.task_id], "arrival_time": arrivals[decoded.task_id],
                "dispatch_time": started, "waiting_time": started - arrivals[decoded.task_id],
                "transport_mode": decoded.transport_mode, "task_result": "", "failure_reason": "",
                "finish_time": None, "flow_time": None}
        _, value, terminated, truncated, info = env.step(index)
        reward += value; decisions += 1; peak = max(peak, info["active_task_count"])
        blocked += int(info["selected_action"].get("type") == "WAIT" and had_waiting)
        if mode == "SERIAL" and isinstance(decoded, Assignment):
            finish = started + info["delta_time"] - dispatch.decision_gap; row = rows[decoded.task_id]
            row.update(task_result=info["selected_action"]["task_result"],
                       failure_reason=info["selected_action"]["failure_reason"],
                       finish_time=finish, flow_time=finish - row["arrival_time"])
        done = terminated or truncated
    if mode == "CONCURRENT":
        for event in dispatch.completion_events:
            row = rows[event["task_id"]]
            row.update(task_result=event["task_result"], failure_reason=event["failure_reason"],
                       finish_time=event["finished_at"], flow_time=event["finished_at"] - row["arrival_time"])
    return {"seed": seed, "mode": mode, "rows": [rows[k] for k in sorted(rows)],
            "completed": dispatch.completed, "failed": dispatch.failed, "time": dispatch.now,
            "reward": reward, "peak": peak, "blocked_waits": blocked, "decisions": decisions,
            "distance": sum(x.distance_total for x in dispatch.robots.values()),
            "handoffs": len(dispatch.handover_events),
            "timeouts": sum(x["timed_out"] for x in dispatch.handover_events),
            "leak": bool(dispatch.resource_claims or getattr(dispatch, "active_standby_claims", {}) or
                         getattr(dispatch, "active_tasks", {}) or
                         any(x.robot.task_id or x.robot.cargo_id for x in dispatch.robots.values()))}

def aggregate(episodes):
    rows = [r for e in episodes for r in e["rows"]]; waits = [r["waiting_time"] for r in rows]
    flows = [r["flow_time"] for r in rows]; handoffs = sum(e["handoffs"] for e in episodes)
    success = sum(r["task_result"] == "COMPLETED" for r in rows); total_time = sum(e["time"] for e in episodes)
    return {"episode_count": len(episodes), "task_count": len(rows), "completed_task_count": success,
            "failed_task_count": len(rows)-success, "success_rate": success/len(rows),
            "mean_episode_time": statistics.mean(e["time"] for e in episodes),
            "throughput_tasks_per_hour": len(rows)/total_time*3600,
            "mean_waiting_time": statistics.mean(waits), "p95_waiting_time": nearest_rank_percentile(waits,.95),
            "mean_flow_time": statistics.mean(flows), "p95_flow_time": nearest_rank_percentile(flows,.95),
            "mean_distance_per_task": sum(e["distance"] for e in episodes)/len(rows),
            "mean_episode_reward": statistics.mean(e["reward"] for e in episodes),
            "blocked_wait_count": sum(e["blocked_waits"] for e in episodes),
            "maximum_active_tasks": max(e["peak"] for e in episodes),
            "transport_modes": dict(Counter(r["transport_mode"] for r in rows)),
            "handover_attempt_count": handoffs, "handover_timeout_count": sum(e["timeouts"] for e in episodes),
            "handover_timeout_rate": sum(e["timeouts"] for e in episodes)/handoffs if handoffs else 0.0}

def compare(s,c):
    change=lambda old,new:(old-new)/max(old,1e-9)
    return {"episode_time_reduction":change(s["mean_episode_time"],c["mean_episode_time"]),
            "throughput_gain":c["throughput_tasks_per_hour"]/s["throughput_tasks_per_hour"]-1,
            "mean_waiting_reduction":change(s["mean_waiting_time"],c["mean_waiting_time"]),
            "p95_waiting_reduction":change(s["p95_waiting_time"],c["p95_waiting_time"]),
            "mean_flow_time_reduction":change(s["mean_flow_time"],c["mean_flow_time"]),
            "success_rate_delta":c["success_rate"]-s["success_rate"],
            "distance_per_task_delta":c["mean_distance_per_task"]-s["mean_distance_per_task"]}

def signature(e):
    return [(r["task_id"],r["transport_mode"],r["task_result"],round(r["dispatch_time"],9),round(r["finish_time"],9)) for r in e["rows"]]

def make_report(summary,path):
    lines=["# Stage 14 串行/并发任务模式规则基线对比","","SERIAL保留单主任务；CONCURRENT允许最多4个无冲突任务在途。当前是规则基线，不是RL结果。",""]
    for name,m in summary["profiles"].items():
        s,c,d=m["SERIAL"],m["CONCURRENT"],summary["comparisons"][name]
        lines += [f"## {name}","","| 指标 | SERIAL | CONCURRENT |","|---|---:|---:|",
          f"| 成功率 | {s['success_rate']:.2%} | {c['success_rate']:.2%} |",
          f"| 吞吐(任务/小时) | {s['throughput_tasks_per_hour']:.3f} | {c['throughput_tasks_per_hour']:.3f} |",
          f"| 平均等待(s) | {s['mean_waiting_time']:.3f} | {c['mean_waiting_time']:.3f} |",
          f"| P95等待(s) | {s['p95_waiting_time']:.3f} | {c['p95_waiting_time']:.3f} |",
          f"| 平均流转(s) | {s['mean_flow_time']:.3f} | {c['mean_flow_time']:.3f} |",
          f"| P95流转(s) | {s['p95_flow_time']:.3f} | {c['p95_flow_time']:.3f} |",
          f"| 峰值在途任务 | {s['maximum_active_tasks']} | {c['maximum_active_tasks']} |","",
          f"并发吞吐提升{d['throughput_gain']:.2%}，平均等待降低{d['mean_waiting_reduction']:.2%}。",""]
    lines += ["## 强化学习计划","","分别训练SERIAL-only、CONCURRENT-only，并用MIXED按episode交替两种模式。观测显式包含模式位；三组使用相同训练/验证/测试种子，分别报告成功率、吞吐、等待、流转时间、里程、交接超时及资源冲突。","","```json",json.dumps(summary["assertions"],ensure_ascii=False,indent=2),"```",""]
    path.write_text("\n".join(lines),encoding="utf-8")

def main():
    seeds=range(20261601,20261701); intervals={"MEDIUM":MEDIUM_INTERVAL,"DENSE":DENSE_INTERVAL}
    raw={}; profiles={}; comparisons={}; replay_equal=paired_equal=True
    for profile,interval in intervals.items():
        raw[profile]={}; profiles[profile]={}
        for mode in ("SERIAL","CONCURRENT"):
            episodes=[run_episode(seed,mode,interval) for seed in seeds]; print(profile,mode,"episodes=100")
            replay_equal &= signature(run_episode(20261601,mode,interval))==signature(episodes[0])
            raw[profile][mode]=episodes; profiles[profile][mode]=aggregate(episodes)
        for s,c in zip(raw[profile]["SERIAL"],raw[profile]["CONCURRENT"]):
            paired_equal &= [(r["task_id"],r["task_type"],r["arrival_time"]) for r in s["rows"]]==[(r["task_id"],r["task_type"],r["arrival_time"]) for r in c["rows"]]
        comparisons[profile]=compare(profiles[profile]["SERIAL"],profiles[profile]["CONCURRENT"])
    episodes=[e for p in raw.values() for m in p.values() for e in m]
    assertions={"400_episodes_resolve_8000_tasks":len(episodes)==400 and all(e["completed"]+e["failed"]==20 for e in episodes),
      "paired_task_streams_equal":paired_equal,"fixed_seed_replay_equal":replay_equal,
      "only_controlled_handover_failures":all(all(r["task_result"]!="FAILED" or r["failure_reason"]=="HANDOVER_TIMEOUT" for r in e["rows"]) for e in episodes),
      "concurrent_reaches_overlap":all(profiles[p]["CONCURRENT"]["maximum_active_tasks"]>=2 for p in profiles),
      "concurrent_throughput_not_worse":all(comparisons[p]["throughput_gain"]>=0 for p in profiles),
      "concurrent_waiting_not_worse":all(comparisons[p]["mean_waiting_reduction"]>=0 for p in profiles),
      "success_delta_within_2pct":all(abs(comparisons[p]["success_rate_delta"])<=.02 for p in profiles),
      "no_state_or_resource_leaks":not any(e["leak"] for e in episodes)}
    summary={"stage":14,"gate":"SERIAL_CONCURRENT_EXECUTION_MODE_COMPARISON","seed_start":20261601,"seed_end":20261700,
      "observation_size":284,"mode_conditioning":{"state_index_3":"active_task_ratio","state_index_4":"is_concurrent","mixed_curriculum":"even_seed=SERIAL, odd_seed=CONCURRENT"},
      "profiles":profiles,"comparisons":comparisons,
      "training_plan":{"experiments":["SERIAL_ONLY","CONCURRENT_ONLY","MIXED"],"fairness":"Identical seeds, arrivals, initial state, P95 samples and rewards.","metrics":["success_rate","throughput","mean_and_p95_waiting","mean_and_p95_flow_time","distance","handover_timeout","illegal_action","resource_conflict"]},
      "assertions":assertions,"passed":all(assertions.values())}
    output=ROOT/"results"/"stage14"; output.mkdir(parents=True,exist_ok=True)
    rows=[r for p in raw.values() for m in p.values() for e in m for r in e["rows"]]
    (output/"stage14_execution_mode_comparison.tasks.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    (output/"stage14_execution_mode_comparison.summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    make_report(summary,output/"stage14_execution_mode_comparison_report.md"); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0 if summary["passed"] else 1

if __name__=="__main__": raise SystemExit(main())
