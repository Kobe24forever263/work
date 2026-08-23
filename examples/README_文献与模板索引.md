# T-RO 参考文献与 Overleaf 模板

本目录面向“双层仓储中的 Carter–Go2 异构多机器人、事件驱动 SMDP-PPO 高层调度、资源约束和交接风险”整理。

## 当前内容

- `papers/`：45 篇可公开获取的参考论文 PDF，以 `01_` 至 `45_` 的自然数开头命名。
- `PAPER_INDEX.md`：45 篇论文的编号、BibTeX key、主题分组及引用边界。
- `references.bib`：正文当前使用的 45 条 BibTeX 记录。
- `manuscript/`：IEEE T-RO 双盲 LaTeX 草稿、参考文献和可直接导入 Overleaf 的 `TRO_Overleaf_Draft.zip`。
- `template/IEEE_Transactions_LaTeX_Overleaf_Template.zip`：IEEE 官方 Transactions LaTeX 模板。
- `papers_backup_20260813_01_26/`：扩展到 45 篇前的 26 篇版本，仅作可恢复备份。

## 文献分布

45 篇文献覆盖 MRTA 分类与市场机制、异构能力与动态联盟、仓储调度与多机器人路径、MARL 与异步宏动作、SCAN-Planner 及低层导航、SMDP/PPO/GAE/动作掩码。具体优先级和可支撑的表述以 `PAPER_INDEX.md` 为准。

## 使用边界

- SCAN-Planner 的发表验证对象是 Unitree Go2；Carter 是本项目的工程适配。
- Stage 14/18 调度训练使用抽象事件模型，并未与 SCAN-Planner 构成在线传感器闭环。
- 这些文献用于建立相关工作、选择基线和解释方法；是否需要新引用，后续再根据实际新增的实验方法、基线或设备决定。

## 官方入口

- T-RO Information for Authors: <https://www.ieee-ras.org/publications/t-ro/t-ro-information-for-authors/>
- IEEE Template Selector: <https://template-selector.ieee.org/>
