# 机器人资产来源

## Unitree Go2

模型来自同机既有项目：

`task2_handover/SCAN-Planner-Ros2/src/simulator/Utils/go2_description`

复制 xacro、配置和一套 `meshes/*.dae`；未复制内容相同的 `dae/` 副本。`model.sdf` 由该 xacro 在 `use_gazebo:=false` 下确定性转换，保留完整四足关节、惯量、碰撞体和外观网格。

## NVIDIA Carter

模型来自：

`task2_handover/third_party/nvidia_carter/carter`

保留官方 URDF、OBJ、质量、惯量、轮径、轮距和后万向轮层级。原始来源说明一并保存在 `src/carter_description/SOURCE.md`。上游压缩包没有独立许可证文件，使用和再分发仍受 NVIDIA 原始来源条款约束。
