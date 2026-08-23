"""Launch one Go2 on one configured stair in either direction."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


ASSIGNMENTS = {
    "dog_1": "stair_ne",
    "dog_2": "stair_nw",
    "dog_3": "stair_sw",
    "dog_4": "stair_se",
}


def _setup(context):
    robot = LaunchConfiguration("robot").perform(context)
    direction = LaunchConfiguration("direction").perform(context)
    show_rviz = LaunchConfiguration("show_rviz").perform(context)
    publish_robot_state = LaunchConfiguration(
        "publish_robot_state").perform(context)
    gazebo_bridge = LaunchConfiguration("gazebo_bridge").perform(
        context).lower() in ("1", "true", "yes")
    offset_x = float(LaunchConfiguration("initial_offset_x").perform(context))
    offset_y = float(LaunchConfiguration("initial_offset_y").perform(context))
    if robot not in ASSIGNMENTS:
        raise RuntimeError("robot must be dog_1, dog_2, dog_3, or dog_4")
    if direction not in ("up", "down"):
        raise RuntimeError("direction must be up or down")

    bringup = Path(get_package_share_directory("warehouse_bringup"))
    scan = Path(get_package_share_directory("scan_planner"))
    route = (
        bringup / "config" / "scan_planner"
        / f"{robot}_{ASSIGNMENTS[robot]}_{direction}.yaml"
    )
    manifest = __import__("yaml").safe_load(
        (bringup / "config" / "scan_planner"
         / "four_go2_manifest.yaml").read_text(encoding="utf-8")
    )["robots"][robot]
    initial = (
        manifest["up_initial_xyz"]
        if direction == "up" else manifest["down_initial_xyz"]
    )
    actions = []
    if gazebo_bridge:
        actions.append(Node(
            package="warehouse_core",
            executable="go2_gazebo_pose_bridge",
            name="go2_gazebo_pose_bridge",
            output="screen",
            parameters=[{
                "robots_config": str(
                    bringup / "config" / "robots.yaml"),
                "lock_floor2_arrival": direction == "up",
            }],
        ))
    actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(scan / "launch" / "run.launch.py")),
            launch_arguments={
                "is_real_world": "false",
                "use_sim_time": "false",
                "agent_namespace": robot,
                "navi_mode": "2",
                "controller_mode": "open_loop",
                "sensor_type": "lidar",
                "use_gpu": "false",
                "use_pcd_map": "true",
                "pcd_map_file": str(
                    bringup / "maps" / "warehouse_full.pcd"),
                "keypoints_file": str(route),
                "planner_overrides_file": str(
                    bringup / "config" / "scan_planner"
                    / "warehouse_tuning.yaml"),
                "map_size_x": "52.0",
                "map_size_y": "44.0",
                "map_size_z": "7.0",
                "init_x": str(initial[0] + offset_x),
                "init_y": str(initial[1] + offset_y),
                "init_z": str(initial[2]),
                "publish_global_map": LaunchConfiguration(
                    "publish_global_map").perform(context),
                "publish_robot_state": publish_robot_state,
                "publish_gait_state": "true",
                "show_rviz": show_rviz,
            }.items(),
        ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="dog_1"),
        DeclareLaunchArgument("direction", default_value="up"),
        DeclareLaunchArgument("show_rviz", default_value="true"),
        DeclareLaunchArgument("publish_robot_state", default_value="true"),
        DeclareLaunchArgument("publish_global_map", default_value="true"),
        DeclareLaunchArgument("gazebo_bridge", default_value="true"),
        DeclareLaunchArgument("initial_offset_x", default_value="0.0"),
        DeclareLaunchArgument("initial_offset_y", default_value="0.0"),
        OpaqueFunction(function=_setup),
    ])
