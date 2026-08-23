"""Run four isolated SCAN-Planner upstairs routes on the warehouse PCD."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


ROBOTS = (
    ("dog_1", "stair_ne", (20.8, 17.8, 0.45)),
    ("dog_2", "stair_nw", (-20.8, 17.8, 0.45)),
    ("dog_3", "stair_sw", (-20.8, -17.8, 0.45)),
    ("dog_4", "stair_se", (20.8, -17.8, 0.45)),
)


def generate_launch_description():
    bringup = Path(get_package_share_directory("warehouse_bringup"))
    scan = Path(get_package_share_directory("scan_planner"))
    pcd = bringup / "maps" / "warehouse_full.pcd"
    tuning = bringup / "config" / "scan_planner" / "warehouse_tuning.yaml"
    actions = [
        Node(
            package="warehouse_core",
            executable="spatial_manager",
            name="spatial_manager",
            output="screen",
            parameters=[{
                "robots_config": str(bringup / "config" / "robots.yaml"),
                "stairs_config": str(bringup / "config" / "stairs.yaml"),
                "scene_config": str(bringup / "config" / "scene.yaml"),
            }],
        ),
        Node(
            package="warehouse_core",
            executable="go2_gazebo_pose_bridge",
            name="go2_gazebo_pose_bridge",
            output="screen",
            parameters=[{
                "robots_config": str(
                    bringup / "config" / "robots.yaml"),
            }],
        ),
    ]
    for index, (robot, stair, initial) in enumerate(ROBOTS):
        route = (
            bringup / "config" / "scan_planner"
            / f"{robot}_{stair}_up.yaml"
        )
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
                "pcd_map_file": str(pcd),
                "keypoints_file": str(route),
                "planner_overrides_file": str(tuning),
                "map_size_x": "52.0",
                "map_size_y": "44.0",
                "map_size_z": "7.0",
                "init_x": str(initial[0]),
                "init_y": str(initial[1]),
                "init_z": str(initial[2]),
                "publish_global_map": "true" if index == 0 else "false",
                "publish_robot_state": "true",
                "publish_gait_state": "true",
                "show_rviz": "true" if index == 0 else "false",
            }.items(),
        ))
    return LaunchDescription(actions)
