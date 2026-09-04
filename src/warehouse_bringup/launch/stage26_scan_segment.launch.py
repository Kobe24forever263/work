"""Run one generated Stage 26 route through a real SCAN-Planner instance."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _truth(value):
    return value.lower() in ("1", "true", "yes", "on")


def _setup(context):
    robot = LaunchConfiguration("robot").perform(context).strip()
    robot_type = LaunchConfiguration("robot_type").perform(context).lower()
    route_file = Path(
        LaunchConfiguration("route_file").perform(context)).expanduser()
    if not robot:
        raise RuntimeError("robot must not be empty")
    if robot_type not in ("car", "dog"):
        raise RuntimeError("robot_type must be car or dog")
    if not route_file.is_file():
        raise RuntimeError(f"SCAN route file does not exist: {route_file}")
    bringup = Path(get_package_share_directory("warehouse_bringup"))
    scan = Path(get_package_share_directory("scan_planner"))
    tuning = ("carter_tuning.yaml" if robot_type == "car"
              else "warehouse_tuning.yaml")
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(scan / "launch" / "run.launch.py")),
        launch_arguments={
            "is_real_world": "false",
            "use_sim_time": "false",
            "agent_namespace": robot,
            "navi_mode": "2",
            "controller_mode": "open_loop",
            "sensor_type": "lidar",
            "use_gpu": "false",
            "use_pcd_map": "true",
            "pcd_map_file": str(bringup / "maps" / "warehouse_full.pcd"),
            "keypoints_file": str(route_file.resolve()),
            "planner_overrides_file": str(
                bringup / "config" / "scan_planner" / tuning),
            "map_size_x": "52.0",
            "map_size_y": "44.0",
            "map_size_z": "7.0",
            "init_x": LaunchConfiguration("init_x").perform(context),
            "init_y": LaunchConfiguration("init_y").perform(context),
            "init_z": LaunchConfiguration("init_z").perform(context),
            "publish_global_map": LaunchConfiguration(
                "publish_global_map").perform(context),
            "publish_robot_state": "false",
            "publish_gait_state": "false",
            "show_rviz": "false",
        }.items(),
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot"),
        DeclareLaunchArgument("robot_type"),
        DeclareLaunchArgument("route_file"),
        DeclareLaunchArgument("init_x"),
        DeclareLaunchArgument("init_y"),
        DeclareLaunchArgument("init_z"),
        DeclareLaunchArgument("publish_global_map", default_value="false"),
        OpaqueFunction(function=_setup),
    ])
