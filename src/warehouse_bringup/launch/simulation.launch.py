"""Launch Gazebo Sim and six independently namespaced Carter bridges."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


CARTERS = (
    "car_f1_1", "car_f1_2", "car_f1_3", "car_f1_4", "car_f2_1", "car_f2_2"
)


def carter_bridge(robot_id):
    gz_cmd = f"/model/{robot_id}/cmd_vel"
    sensor_root = (
        f"/world/two_floor_warehouse/model/{robot_id}/link/chassis_link/sensor"
    )
    gz_scan = f"{sensor_root}/lidar/scan"
    gz_imu = f"{sensor_root}/imu/imu"
    gz_joints = f"/world/two_floor_warehouse/model/{robot_id}/joint_state"
    return Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        namespace=robot_id,
        name="gazebo_bridge",
        arguments=[
            f"{gz_cmd}@geometry_msgs/msg/Twist@gz.msgs.Twist",
            f"{gz_scan}@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
            f"{gz_imu}@sensor_msgs/msg/Imu@gz.msgs.IMU",
            f"{gz_joints}@sensor_msgs/msg/JointState@gz.msgs.Model",
        ],
        remappings=[
            (gz_cmd, "cmd_vel"),
            (gz_scan, "scan"),
            (gz_imu, "imu"),
            (gz_joints, "joint_states"),
        ],
        output="screen",
    )


def generate_launch_description():
    gz_share = Path(get_package_share_directory("ros_gz_sim"))
    warehouse_share = Path(get_package_share_directory("warehouse_bringup"))
    world = warehouse_share / "worlds" / "two_floor_warehouse.sdf"
    resource_path = "/Users/lab4099/Desktop/Mujoco/work/src"
    actions = [
        DeclareLaunchArgument("gui", default_value="true"),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource_path),
        SetEnvironmentVariable("GZ_IP", "127.0.0.1"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(gz_share / "launch" / "gz_sim.launch.py")),
            launch_arguments={
                # Gazebo Sim on macOS requires server and GUI in separate
                # processes.
                "gz_args": ["-s -r -v 3 ", world],
                "on_exit_shutdown": "true",
            }.items(),
        ),
        ExecuteProcess(
            cmd=[
                "gz", "sim", "-g", "-v", "3",
                "--render-engine-gui-api-backend", "opengl",
            ],
            output="screen",
        ),
    ]
    actions.extend(carter_bridge(robot_id) for robot_id in CARTERS)
    actions.append(Node(
        package="warehouse_core",
        executable="carter_state_bridge",
        name="carter_state_bridge",
        parameters=[{
            "robots_config": str(warehouse_share / "config" / "robots.yaml"),
        }],
        output="screen",
    ))
    return LaunchDescription(actions)
