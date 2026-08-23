# 论文 PDF 与 IEEE 引用编号索引

编号按 `main.tex` 中首次引用的先后顺序生成，并与 IEEE 数字引用顺序一致。后续如果调整正文第一次引用的位置，需要同步重排本表和 `papers/` 文件名。

| 编号 | BibTeX key | PDF |
|---:|---|---|
| 1 | `gerkey2004taxonomy` | `01_MRTA_Taxonomy_IJRR2004.pdf` |
| 2 | `korsah2013taxonomy` | `02_Comprehensive_MRTA_Taxonomy_IJRR2013.pdf` |
| 3 | `dias2006market` | `03_Market_Based_Multirobot_Coordination_ProceedingsIEEE2006.pdf` |
| 4 | `zlot2006market` | `04_Market_Based_Complex_Tasks_IJRR2006.pdf` |
| 5 | `parker1994alliance` | `05_ALLIANCE_IROS1994.pdf` |
| 6 | `notomista2022resilient` | `06_Resilient_Energy_Aware_MRTA_TRO2022.pdf` |
| 7 | `fu2023robust` | `07_Robust_Heterogeneous_Task_Scheduling_TRO2023.pdf` |
| 8 | `aswale2023coalition` | `08_Heterogeneous_Coalition_Formation_IROS2023.pdf` |
| 9 | `dai2025heterogeneous` | `09_HetMRTA_RAL2025.pdf` |
| 10 | `parker2006asymtre` | `10_ASyMTRe_ProceedingsIEEE2006.pdf` |
| 11 | `zhang2013iqasymtre` | `11_IQ_ASyMTRe_TRO2013.pdf` |
| 12 | `choudhury2022dynamic` | `12_SCoBA_AutonomousRobots2022.pdf` |
| 13 | `neville2023ditags` | `13_D_ITAGS_RAL2023.pdf` |
| 14 | `calvo2025longendurance` | `14_Heterogeneous_MRTA_Long_Endurance_TRO2025.pdf` |
| 15 | `bays2017satp` | `15_Service_Transport_Agent_Scheduling_RAS2017.pdf` |
| 16 | `yi2025tidying` | `16_Cooperative_Tidying_Transport_Agents_Sensors2025.pdf` |
| 17 | `zhou2022reactive` | `17_Quadrupedal_Wheeled_Teaming_CASE2022.pdf` |
| 18 | `oliveira2022warehouse` | `18_Heterogeneous_Robots_Smart_Warehouse_FUSION2022.pdf` |
| 19 | `wurman2007warehouse` | `19_Coordinating_Hundreds_Warehouse_Robots_IAAI2007.pdf` |
| 20 | `ma2017lifelong` | `20_Lifelong_MAPD_AAMAS2017.pdf` |
| 21 | `damani2021primal2` | `21_PRIMAL2_RAL2021.pdf` |
| 22 | `agrawal2023rtaw` | `22_RTAW_ICRA2023.pdf` |
| 23 | `pal2024heuristic` | `23_HeuRAL_MATE_ECAI2024.pdf` |
| 24 | `pal2025together` | `24_MRTAgent_AAMAS2025.pdf` |
| 25 | `krnjaic2024scalable` | `25_Scalable_MARL_Warehouse_IROS2024.pdf` |
| 26 | `zhang2024asynchronous` | `26_Async_Heterogeneous_MARL_RAL2024.pdf` |
| 27 | `jose2024learning` | `27_LVWS_ICRA2024.pdf` |
| 28 | `omidshafiei2017multitask` | `28_Deep_Decentralized_Multitask_MARL_ICML2017.pdf` |
| 29 | `lowe2017maddpg` | `29_MADDPG_NeurIPS2017.pdf` |
| 30 | `foerster2018coma` | `30_COMA_AAAI2018.pdf` |
| 31 | `rashid2018qmix` | `31_QMIX_ICML2018.pdf` |
| 32 | `yu2022mappo` | `32_MAPPO_NeurIPS2022.pdf` |
| 33 | `kuba2022happo` | `33_HAPPO_ICLR2022.pdf` |
| 34 | `bezerra2025dynamic` | `34_Dynamic_Coalition_MAPPO_RAL2025.pdf` |
| 35 | `xiao2025asynchronous` | `35_Asynchronous_Multi_Agent_Deep_RL_IJRR2025.pdf` |
| 36 | `farjadnasab2025catmip` | `36_CATMiP_RAS2025.pdf` |
| 37 | `zheng2026scan` | `37_SCAN_Planner_arXiv2026.pdf` |
| 38 | `zhou2021egoplanner` | `38_EGO_Planner_RAL2021.pdf` |
| 39 | `ren2024rogmap` | `39_ROG_Map_IROS2024.pdf` |
| 40 | `wellhausen2023artplanner` | `40_ArtPlanner_FieldRobotics2023.pdf` |
| 41 | `chen2023smug` | `41_SMUG_Planner_RAL2023.pdf` |
| 42 | `messias2013gsmdp` | `42_GSMDP_Multi_Robot_AAAI2013.pdf` |
| 43 | `schulman2017ppo` | `43_PPO_arXiv2017.pdf` |
| 44 | `schulman2016gae` | `44_GAE_ICLR2016.pdf` |
| 45 | `huang2022masking` | `45_Invalid_Action_Masking_FLAIRS2022.pdf` |

## 使用边界

- `[3]`--`[5]` 用于市场机制、复杂任务分解和故障容错的历史脉络；本文没有实现拍卖或 ALLIANCE。
- `[15]`--`[16]` 支撑异构角色、运输代理与载荷转移的相关工作；它们不是本文两层仓库和两次交接协议的直接基线。
- `[19]`--`[25]` 区分仓储系统、在线取送、路径规划和学习型任务分配；当前逻辑 SMDP 没有与 MAPF/SCAN 闭环联合训练。
- `[28]`--`[36]` 用于 MARL、CTDE、信用分配、异步宏动作和异构协作脉络；当前实现仍是集中式 masked SMDP-PPO，而不是 MAPPO、HAPPO、QMIX、COMA 或 CATMiP。
- `[37]`--`[41]` 属于导航执行层。SCAN-Planner 当前只有 2026 年 arXiv 原论文；项目使用社区 ROS 2 移植和静态 PCD/预设航点/开环回放，不能据此声称传感器闭环、接触动力学或实机验证。
