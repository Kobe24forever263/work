"""Launch six Carter ROS-Gazebo bridges without starting Gazebo itself."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


CARTERS = (
    "car_f1_1", "car_f1_2", "car_f1_3",
    "car_f1_4", "car_f2_1", "car_f2_2",
)


def bridge(robot):
    command = f"/model/{robot}/cmd_vel"
    root = (
        f"/world/two_floor_warehouse/model/{robot}/"
        "link/chassis_link/sensor")
    scan = f"{root}/lidar/scan"
    imu = f"{root}/imu/imu"
    joints = f"/world/two_floor_warehouse/model/{robot}/joint_state"
    return Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        namespace=robot,
        name="gazebo_bridge",
        arguments=[
            f"{command}@geometry_msgs/msg/Twist@gz.msgs.Twist",
            f"{scan}@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
            f"{imu}@sensor_msgs/msg/Imu@gz.msgs.IMU",
            f"{joints}@sensor_msgs/msg/JointState@gz.msgs.Model",
        ],
        remappings=[
            (command, "cmd_vel"),
            (scan, "scan_raw"),
            (imu, "imu"),
            (joints, "joint_states"),
        ],
        output="screen",
    )


def _setup(context):
    selected = LaunchConfiguration("robot").perform(context)
    if selected != "all" and selected not in CARTERS:
        raise RuntimeError("robot must be all or one configured Carter name")
    bringup = Path(get_package_share_directory("warehouse_bringup"))
    actions = [Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="clock_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )]
    robots = CARTERS if selected == "all" else (selected,)
    actions.extend(bridge(robot) for robot in robots)
    actions.append(Node(
        package="warehouse_core",
        executable="carter_state_bridge",
        name="carter_state_bridge",
        parameters=[{
            "robots_config": str(bringup / "config" / "robots.yaml"),
            "use_sim_time": True,
        }],
        output="screen",
    ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="all"),
        OpaqueFunction(function=_setup),
    ])
