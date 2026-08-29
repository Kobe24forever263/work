# GitHub 归档快照（2026-08-29）

本次快照目标是保存可复现实验代码、文档、论文源文件、图表数据和已完成的质量审计结果。

## 已纳入

- Stage 23 的 `MARKOV_CONTEXT_V3` 状态编码、连续时间奖励、duration-scaled GAE、归一化器修复及测试；
- Stage 23 接口门禁、状态别名审计、事件切分审计、三种子 pilot 和精确断点续训门禁；
- Stage 22 已冻结的算法消融协议与接口/冻结摘要；
- 已完成的 Stage 23 seed 1、seed 6 训练摘要和历史记录（不含权重）；
- Stage 1–18 的可复现实验快照：阶段代码与配置（仓库原有内容）、summary/report/gate/status/manifest/schema、CSV 统计和质量审计文档；
- Stage 20/21 论文数据、图表源数据、矢量图、绘图脚本和 T-RO 论文源文件；
- 0803、0806、0809、0823、0826、0827 交接文档及阶段说明。

## Stage 1–18 快照边界

- Stage 1–4 在当前工作区没有独立的 `results/stage1`–`results/stage4` 目录；对应的代码、配置和阶段文档仍按仓库现有目录归档。
- Stage 5–18 的非 history JSON（含 locked-test seed 输出、冻结/质量审查结果）、摘要、报告、门禁/冻结结果、状态清单、CSV，以及质量审查 HTML/脚本/Notebook/小型样本已纳入本次 Git 提交；Stage 18 的 Medium/Dense/Burst 及 clean-ablation/locked-test 结果均按此规则保留。
- 为控制仓库体量和避免把训练中间态误当成论文证据，本次不纳入 `*.pt`/`*.pth` 权重、`*.history.json` 逐回合历史、`*.jsonl` 逐步日志、检查点和缓存。它们仍保留在本机；需要复现实验时按对应脚本和 manifest 重新生成。
- 本次候选快照约 1,686 个文件、136.67 MB；本地 Stage 5–18 仍有约 0.58 GB history、28 MB JSONL 和 11.8 GB 权重未上传。

## 明确未纳入

- `results/stage23_markov_continuous/long/` 中正在变化的 seed 2、seed 7 及后续种子的训练文件不能上传；仅保留已完成 seed 1、seed 6 的不可变 summary/history/status/run-plan 摘要，作为当前进度快照；
- 所有 `*.pt` 模型权重：权重较大且当前 cohort 尚未全部冻结；正式十种子完成后单独发布权重清单和哈希；
- `*.history.json`、`*.jsonl` 逐步日志、缓存、LaTeX 编译临时文件和构建目录；
- Stage 1–18 的大体积权重/逐步日志不会通过本次普通 Git 提交上传；如论文复核需要，再单独制作带校验和的发布包；
- 任何未通过正式 locked test 的中间结果。

## 结果解释边界

Stage 23 pilot 和接口审计用于验证状态/回报契约、训练安全性和断点可复现性，不构成正式性能优越性结论。正式结论必须等待十种子训练完成、策略冻结、共享 locked test 和预先声明的统计分析。

## 恢复当前长训练

训练目录仍保留在本机。重新运行对应的 Stage 23 campaign 命令即可按每 50 update checkpoint 自动续训；不要从 GitHub 快照中的摘要文件推断训练已经完成。
