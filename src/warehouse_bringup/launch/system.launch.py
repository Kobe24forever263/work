from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    mode = LaunchConfiguration("mode")
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="rule"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        LogInfo(msg=["Warehouse mode: ", mode]),
        Node(
            package="warehouse_core",
            executable="rule_scheduler",
            name="rule_scheduler",
            output="screen",
            parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
            condition=IfCondition(PythonExpression(["'", mode, "' == 'rule'"])),
        ),
    ])
