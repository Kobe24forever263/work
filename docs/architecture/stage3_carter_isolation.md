# 阶段 3：Carter 控制隔离实测

日期：2026-07-29  
环境：macOS Apple Silicon、Pixi ROS 2 Jazzy、Gazebo Sim 8.10.0

## 已实现

- 6 台 Carter 均加载独立 `DiffDrive` 系统。
- Gazebo 控制/里程计话题按模型名隔离：
  `/model/<robot_id>/cmd_vel`、`/model/<robot_id>/odometry`。
- ROS 2 侧映射为：
  `/<robot_id>/cmd_vel`、`/<robot_id>/odom`。
- Go2 仍保持固定，等待 SCAN-Planner 与步态控制接入。

## 实测结果

仅向 `car_f1_1` 发送 2 秒、`0.35 m/s` 前进命令：

- `car_f1_1` 里程计位移：`0.98568 m`
- 其余 5 台 Carter：约 `6e-15 m`，属于浮点噪声

随后通过 ROS 2 namespace ` /car_f1_2/cmd_vel` 发送 2 秒、
`0.30 m/s` 前进命令：

- ROS 2 ` /car_f1_2/odom` 报告位移：`0.90524 m`
- ROS 2 图中只出现 ` /car_f1_2/cmd_vel` 与 ` /car_f1_2/odom`

结论：Gazebo 原生控制与 ROS 2 桥接均未发生车辆控制串台。

## 传感器与 TF

6 台 Carter 已各自生成独立 Gazebo `scan`、`IMU` 和 `joint_state`。
ROS 2 实测收到：

- `/car_f1_1/scan`：约 `10 Hz`，frame 为
  `car_f1_1/chassis_link/lidar`
- `/car_f1_1/imu`：frame 为 `car_f1_1/chassis_link/imu`
- `/car_f1_1/joint_states`：包含左右驱动轮、后转向架和后轮轴

Gazebo 原生 `/model/<robot_id>/tf` 未产生有效消息，因此增加
`carter_state_bridge`，直接订阅 6 条真实 Gazebo odometry，并发布：

- `/<robot_id>/odom`
- `<robot_id>/odom → <robot_id>/base_link`

移动 `car_f1_3` 后实测 ROS odom 位移为 `0.382 m`，TF 消息持续发布；
6 条 odom 话题和 12 个父子 frame 名均带 robot_id，不存在重复 TF。

## 统一状态与诊断

`carter_state_bridge` 从唯一配置源 `robots.yaml` 读取 home_floor、
home_region 和初始世界位姿，把 Gazebo 相对里程计转换为世界坐标，并为每台车发布：

- `/<robot_id>/robot_state`
- `/<robot_id>/diagnostics`

抽查结果：

- `car_f1_1`：world `(2.5, 2.5)`、floor `1`、region `F1_NE`
- `car_f2_1`：world `(-5.5, 0.0)`、floor `2`、region `F2_WEST`
- 静止导航状态为 `IDLE`
- 健康诊断为 `OK`，实测 odom age 为 `0.019 s`

状态同时维护速度、累计运动距离、可用性、failure_code 和最后有效进展时间。
至此 Carter 的阶段 3 接口已完成；剩余工作为 4 台 Go2 接口与隔离测试。
