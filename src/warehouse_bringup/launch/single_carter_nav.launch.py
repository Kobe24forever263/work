"""Start one namespaced Carter Nav2+AMCL+MPPI stack on its fixed floor."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def _setup(context):
    robot = LaunchConfiguration("robot").perform(context)
    bringup = Path(get_package_share_directory("warehouse_bringup"))
    robots = yaml.safe_load(
        (bringup / "config" / "robots.yaml").read_text(encoding="utf-8")
    )["robots"]
    if robot not in robots or robots[robot]["type"] != "car":
        raise RuntimeError("robot must name one of the six configured Carter cars")
    floor = int(robots[robot]["home_floor"])
    initial = robots[robot]["initial_xyz_yaw"]
    nav2 = Path(get_package_share_directory("nav2_bringup"))
    return [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            namespace=robot,
            name=f"{robot}_map_odom_tf",
            arguments=[
                "--x", str(initial[0]), "--y", str(initial[1]), "--z", "0.0",
                "--yaw", str(initial[3]), "--pitch", "0.0", "--roll", "0.0",
                "--frame-id", "map",
                "--child-frame-id", f"{robot}/odom",
            ],
            parameters=[{"use_sim_time": True}],
            remappings=[("/tf_static", "tf_static")],
            output="screen",
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            namespace=robot,
            name=f"{robot}_lidar_static_tf",
            arguments=[
                "--x", "0.12", "--y", "0.0", "--z", "0.48",
                "--yaw", "0.0", "--pitch", "0.0", "--roll", "0.0",
                "--frame-id", f"{robot}/base_link",
                "--child-frame-id", f"{robot}/chassis_link/lidar",
            ],
            parameters=[{"use_sim_time": True}],
            remappings=[("/tf_static", "tf_static")],
            output="screen",
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(nav2 / "launch" / "bringup_launch.py")),
            launch_arguments={
                "namespace": robot,
                "use_namespace": "True",
                "slam": "False",
                "map": str(bringup / "maps" / f"floor{floor}.yaml"),
                "use_sim_time": "True",
                "params_file": str(
                    bringup / "config" / f"nav2_{robot}.yaml"),
                "autostart": "True",
                "use_composition": "False",
                "use_respawn": "False",
            }.items(),
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="car_f1_1"),
        OpaqueFunction(function=_setup),
    ])
