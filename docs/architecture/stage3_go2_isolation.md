# 阶段 3：Go2 接口与隔离实测

日期：2026-07-30  
环境：macOS Apple Silicon、Pixi ROS 2 Jazzy、Gazebo Sim 8.10.0

## 已实现

- 4 台 Go2 分别使用 `/dog_1` 至 `/dog_4` 命名空间。
- 每台机器人独立发布 `odom`、`gait_joint_states`、`robot_state` 和
  `diagnostics`。
- TF 根节点分别为 `world → dog_<n>/base`，机器人模型关节 frame 使用相同
  `dog_<n>` 前缀。
- Gazebo 关节命令分别写入
  `/model/dog_<n>/joint/<joint>/0/cmd_pos`，不存在共享控制话题。
- `RobotState` 统一维护楼层、区域、位姿、速度、导航状态、运动代价和最后进展
  时间；诊断维护里程计新鲜度、楼层和终点位姿锁定状态。

## 自动化实测

在四套 SCAN-Planner 和 Gazebo 同时运行时执行
`scripts/validate_go2_interfaces.py`，8 秒采样窗口结果：

- `dog_1`：625 条独立状态消息
- `dog_2`：625 条独立状态消息
- `dog_3`：624 条独立状态消息
- `dog_4`：621 条独立状态消息

所有 `robot_state.robot_id`、`diagnostics.hardware_id` 和诊断名称均与所在
namespace 一致，未发现状态或诊断串台，结果为 `PASS`。

## 结论

四台 Go2 的 namespace、TF 根、状态、诊断和关节控制目标均按 robot_id
隔离。结合 Carter 已完成的六机隔离实测，阶段 3 的十机器人接口门禁完成。
