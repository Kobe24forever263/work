# 双层仓储异构多智能体协作仿真

本工作区严格按《文字文稿.docx》的技术路线组织：先环境、后机器人；先单体、后协作；先规则、后强化学习；先固定分工、后动态优化。高层策略只选择任务、楼梯、交接、等待、重规划和协助，不直接输出关节控制或 `cmd_vel`。

自 2026-08-04 起，阶段 11 之后采用异步连续任务主线：一次只执行一个主运输
任务，任务之间保留调度间隔，机器人位置、楼层和累计工作状态不复位。规则或
强化学习在每次任务到达时组合选择取货 Carter、Go2、楼梯、目标楼层接货
Carter、交接点和任务后待命策略。原 2–4 个任务并发压力测试降为后续扩展。
详细决策见 `docs/architecture/decision_20260804_async_persistent_dispatch.md`。

## 当前交付边界

当前包含第一阶段完整设计基线、第二阶段静态环境，以及 SCAN-Planner
上楼规划资产：统一接口、全部场景参数、10 机器人配置、两层任务点/区域/楼梯/
交接位、规则运输链、资源预约、事务性交接状态机、动作掩码、四条连续实体楼梯、
同源货架/二维地图、仓库 PCD 点云和四条 Go2 上楼 YAML 路线。自 2026-08-02
方案变更后，Go2 与全部 Carter 统一采用 SCAN-Planner，阶段开发和中间验收统一
在 RViz 中完成；Gazebo 接触动力学、传感器闭环和性能优化集中到最终集成阶段。
SCAN 的 RViz 运动学验证不等于 Gazebo 实体运动验收，最终交付前仍必须补齐。

## 构建

```bash
cd /Users/lab4099/Desktop/Mujoco/task2_handover
pixi run colcon build --base-paths /Users/lab4099/Desktop/Mujoco/work/src \
  --build-base /Users/lab4099/Desktop/Mujoco/work/build_context_v2 \
  --install-base /Users/lab4099/Desktop/Mujoco/work/install
```

`build_context_v2`用于避开旧`build/warehouse_core`中由早期setuptools/colcon组合留下的
develop元数据；继续使用旧目录会出现`--uninstall`或`--editable`参数错误。

纯逻辑检查（不要求 ROS）：

```bash
python3 scripts/validate_config.py
python3 scripts/run_logic_tests.py
python3 scripts/generate_world.py
python3 scripts/validate_world.py
```

运行模式：`single_dog`、`single_car`、`rule`、`mappo_train`、`mappo_eval`。当前可执行基线是 `rule`；其余模式已保留稳定配置入口，必须在对应阶段达到验收门槛后启用。

## 打开 Gazebo 并控制 Carter

### 1. 打开仿真场景

在 Finder 中进入：

`/Users/lab4099/Desktop/Mujoco/work/scripts`

双击：

`open_gazebo.command`

macOS 会打开一个终端窗口并启动 Gazebo。首次加载机器人网格可能需要数秒，
看到双层仓库、4 台 Go2 和 6 台 Carter 后再启动控制器。仿真运行期间不要关闭
这个终端窗口；关闭 Gazebo 窗口或按终端中的 `Control+C` 可以停止仿真。

如果 macOS 阻止首次打开，可右键该文件，选择“打开”，再在提示框中确认。

### 2. 打开 Carter 键盘控制器

保持 Gazebo 运行，再次进入同一目录并双击：

`control_carter.command`

控制键：

- 数字键 `1–6`：切换 `car_f1_1`、`car_f1_2`、`car_f1_3`、
  `car_f1_4`、`car_f2_1`、`car_f2_2`
- `W/S`：前进、后退
- `A/D`：左转、右转
- 空格或 `X`：立即停车
- `Q`：全部车辆停车并退出控制器

移动时需要持续按键；停止操作约 `0.7 s` 后脚本会自动停车。切换车辆、退出程序
或程序异常结束时，也会向车辆发送停车命令。

### 3. 命令行启动方式

也可以在终端执行：

```bash
open /Users/lab4099/Desktop/Mujoco/work/scripts/open_gazebo.command
open /Users/lab4099/Desktop/Mujoco/work/scripts/control_carter.command
```

## SCAN-Planner：四只 Go2 上楼

先完成一次上面的构建。然后在 Finder 中进入：

`/Users/lab4099/Desktop/Mujoco/work/scripts`

双击：

`run_four_go2_scan.command`

脚本会为 `dog_1` 至 `dog_4` 各启动一套隔离话题的 SCAN-Planner，读取
`warehouse_full.pcd` 和各自的 YAML 路线。终端中出现每只狗的
`Planning to waypoint 13/13`，随后状态进入 `WAIT_TARGET`，表示规划器的
开环可达性验证完成。按 `Control+C` 关闭。

相关文件：

- `src/warehouse_bringup/maps/warehouse_full.pcd`：两层仓库及四条楼梯点云
- `src/warehouse_bringup/config/scan_planner/dog_*_up.yaml`：四条上楼路径
- `src/warehouse_bringup/config/scan_planner/warehouse_tuning.yaml`：Go2
  机身包络、局部地图和速度参数
- `src/warehouse_bringup/launch/four_go2_scan_validation.launch.py`：四狗启动器

重新生成或检查资产：

```bash
cd /Users/lab4099/Desktop/Mujoco/task2_handover
.pixi/envs/default/bin/python \
  /Users/lab4099/Desktop/Mujoco/work/scripts/generate_scan_assets.py
.pixi/envs/default/bin/python \
  /Users/lab4099/Desktop/Mujoco/work/scripts/validate_scan_assets.py
```

四狗仿真运行时，可在另一个已加载工作区环境的终端执行接口隔离检查：

```bash
python3 /Users/lab4099/Desktop/Mujoco/work/scripts/validate_go2_interfaces.py
```

检查器会在 8 秒内核对四个 `robot_state` 与 `diagnostics` 命名空间中的
robot_id，输出 `Go2 interface isolation: PASS` 表示未发生串台。

阶段 4 的运行时验收使用：

```bash
python3 /Users/lab4099/Desktop/Mujoco/work/scripts/validate_stage4_runtime.py
```

应在四狗仿真运行期间执行。它会等待可靠楼层切换完成，检查四条楼梯均已释放、
区域识别正确、没有提前切层，并确认 Carter 的楼梯预约被拒绝。

## SCAN-Planner：车—狗—车跨楼层协作演示

不启动 Gazebo，仅使用 RViz 查看一楼 Carter、Go2 和二楼 Carter 的完整协作链：

```bash
cd /Users/lab4099/Desktop/Mujoco/work/scripts
./run_cross_floor_scan_rviz.command ne
./run_cross_floor_scan_rviz.command nw
./run_cross_floor_scan_rviz.command sw
./run_cross_floor_scan_rviz.command se
```

参数分别代表东北、西北、西南、东南楼梯；不提供参数时默认 `ne`。每条任务均按
“一楼车沿外圈到发送位 → 对应 Go2 上楼 → 所属侧二楼车到接收位”执行。三个
智能体均使用 SCAN-Planner；RViz 会自动切换到真实机器人 namespace，同时显示
完整 URDF、仓库三维 PCD、局部点云、规划轨迹和已走路径。

机器人对应关系：

- `ne`：`car_f1_1 → dog_1 → car_f2_2`
- `nw`：`car_f1_2 → dog_2 → car_f2_1`
- `sw`：`car_f1_3 → dog_3 → car_f2_1`
- `se`：`car_f1_4 → dog_4 → car_f2_2`

2026-08-02 东北路线人工验收通过，西北/西南/东南路线完成实际无人值守运行
验收。二楼接收位以唯一场景配置
`STAIR_NE_F2_RECEIVE = (9.5, 6.5, 4.0)` 为准；规划节点使用车体中心高度
`z = 4.70`。运行终端出现 `CROSS-FLOOR SCAN TASK PASS`，且 RViz 中三段轨迹
依次完成，才算本演示通过。详细记录见
`docs/architecture/cross_floor_scan_demo.md`。

该演示证明异构规划话题、TF、三维显示和顺序交接链可运行；它不替代阶段 5、
阶段 6 和阶段 8 要求的批量可靠性与事务性交接硬门禁，也不替代最终阶段的
Gazebo 动力学验收。

## 单只 Go2 双向楼梯验证

带 Gazebo 和 RViz 启动单次验证：

```bash
/Users/lab4099/Desktop/Mujoco/work/scripts/run_single_go2_stair.command dog_1 up
/Users/lab4099/Desktop/Mujoco/work/scripts/run_single_go2_stair.command dog_1 down
```

机器人可选 `dog_1` 至 `dog_4`，方向可选 `up` 或 `down`。

阶段 5 正式批测为 160 次，即四条楼梯、两个方向、每组合 20 次：

```bash
python3 /Users/lab4099/Desktop/Mujoco/work/scripts/run_stage5_batch.py \
  --seed 20260730 --repeats 20 \
  --output /Users/lab4099/Desktop/Mujoco/work/results/stage5/stage5_full.jsonl
```

批测会保存逐次结果、初始随机扰动、耗时、失败分类和汇总文件。未真正完成
160 次前，不得将阶段 5 标为通过。

## 目录

- `src/warehouse_interfaces`：统一消息、服务和动作接口
- `src/warehouse_core`：领域状态、运输链、预约、交接事务、动作掩码
- `src/warehouse_bringup`：总启动、场景/机器人/任务/导航/RL 参数和 Gazebo 世界
- `docs/architecture`：架构、通信、TF、状态机与阶段门禁
- `scripts/validate_config.py`：参数一致性与唯一性检查
- `scripts/generate_world.py`：由 YAML 确定性生成 SDF 与双层地图
- `scripts/validate_world.py`：楼梯连续性、净宽、护栏、标记与地图检查

唯一配置源是 `src/warehouse_bringup/config/*.yaml`。增加机器人、任务点、区域或调整坐标时不得修改核心代码。
