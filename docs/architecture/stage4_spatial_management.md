# 阶段 4：地图权限、区域与楼梯管理验收

日期：2026-07-30  
环境：macOS Apple Silicon、Pixi ROS 2 Jazzy、Gazebo Sim 8.10.0

## 实现

- `RegionManager` 从唯一 `scene.yaml` 区域边界计算机器人当前区域、占用者和
  拥堵状态。
- `StairManager` 提供四条楼梯的独占租约、FIFO 排队、持有者校验、占用状态、
  超时回收和自动晋级。
- Carter 的楼梯申请在资源占用判断之前即因机器人类型不符被拒绝。
- `FloorTransitionTracker` 只在依次满足预约、入口、实际占用、通过楼梯、出口、
  低速稳定 1.5 秒和定位新鲜后切换楼层。瞬时高度不能单独触发切换。
- 运行时统一发布：
  `/warehouse/state/robots`、`/warehouse/state/stairs`、
  `/warehouse/state/regions` 和 `/warehouse/events`。
- 预约服务为 `/warehouse/reservations/reserve` 与
  `/warehouse/reservations/release`。

## 自动测试

纯逻辑测试覆盖：

- 车辆楼梯权限拒绝；
- 双狗争用时 FIFO 排队；
- 非持有者不得释放；
- 租约超时后自动晋级；
- 仅高度变化不得切层；
- 定位失效时不得切层；
- 出口稳定时间门禁；
- 配置驱动的跨楼层区域识别。

结果：`8 passed`。配置、世界和双地图静态检查同时通过。

## 四狗运行时验收

四套 SCAN-Planner、Gazebo 和空间管理节点同时运行，最终结果：

- `dog_1`：`floor=2`，`region=F2_EAST`
- `dog_2`：`floor=2`，`region=F2_WEST`
- `dog_3`：`floor=2`，`region=F2_WEST`
- `dog_4`：`floor=2`，`region=F2_EAST`
- `STAIR_NE/NW/SW/SE`：全部恢复 `FREE`
- Carter 权限探针：`ROBOT_NOT_STAIR_CAPABLE`

验收器未检测到二楼出口高度以下的提前切层，结果为
`Stage 4 runtime validation: PASS`。

## 结论

阶段 4 的双层空间状态、区域识别、楼梯权限、互斥租约、超时恢复和可靠楼层
切换门禁完成。单车批量导航属于阶段 6，不在本阶段冒充验收。该句原指
Nav2+MPPI；依据 2026-08-02 方案变更，阶段 6 已改为 Carter SCAN-Planner
与 RViz 批量验收。
