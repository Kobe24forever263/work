"""Show the selected Carter-Go2-Carter mission in one RViz frame."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


MISSIONS = {
    "ne": ("car_f1_1", "dog_1", "car_f2_2"),
    "nw": ("car_f1_2", "dog_2", "car_f2_1"),
    "sw": ("car_f1_3", "dog_3", "car_f2_1"),
    "se": ("car_f1_4", "dog_4", "car_f2_2"),
}


def _setup(context):
    mission = LaunchConfiguration("mission").perform(context).lower()
    if mission not in MISSIONS:
        raise RuntimeError("mission must be ne, nw, sw, or se")
    floor1_car, dog, floor2_car = MISSIONS[mission]
    bringup = Path(get_package_share_directory("warehouse_bringup"))

    carter_source = Path(
        "/Users/lab4099/Desktop/Mujoco/work/src/carter_description/urdf/carter.urdf"
    )
    carter_description = carter_source.read_text(encoding="utf-8").replace(
        "package://carter/meshes",
        "file:///Users/lab4099/Desktop/Mujoco/work/src/carter_description/meshes",
    )
    carter_description = carter_description.replace(
        '<material name="gray"/>',
        '<material name="carter_blue"><color rgba="0.04 0.32 0.92 1"/></material>',
    ).replace(
        '<material name="gray" />',
        '<material name="carter_blue"><color rgba="0.04 0.32 0.92 1"/></material>',
    ).replace(
        '<material name="black"/>',
        '<material name="carter_black"><color rgba="0.03 0.04 0.07 1"/></material>',
    ).replace(
        '<material name="black" />',
        '<material name="carter_black"><color rgba="0.03 0.04 0.07 1"/></material>',
    )
    go2_xacro = Path(
        "/Users/lab4099/Desktop/Mujoco/work/src/go2_description/xacro/robot.xacro")

    # The checked-in RViz file is the accepted NE layout. Generate a mission-
    # specific runtime copy so every display follows the real robot namespace.
    rviz_text = (bringup / "config" / "heterogeneous.rviz").read_text(
        encoding="utf-8")
    rviz_text = rviz_text.replace("car_f1_1", floor1_car)
    rviz_text = rviz_text.replace("dog_1", dog)
    rviz_text = rviz_text.replace("car_f2_2", floor2_car)
    rviz_config = Path(f"/private/tmp/warehouse_heterogeneous_{mission}.rviz")
    rviz_config.write_text(rviz_text, encoding="utf-8")

    actions = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="map_to_world",
            arguments=["0", "0", "0", "0", "0", "0", "map", "world"],
            parameters=[{"use_sim_time": False}],
        )
    ]
    for car in (floor1_car, floor2_car):
        actions.append(Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=car,
            name=f"{car}_robot_state_publisher",
            parameters=[{
                "robot_description": carter_description,
                "frame_prefix": f"{car}/",
                "use_sim_time": False,
            }],
            remappings=[("joint_states", f"/task2/{car}/joint_states")],
            output="screen",
        ))
    actions.extend([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=dog,
            name=f"{dog}_robot_state_publisher",
            parameters=[{
                "robot_description": Command(
                    ["xacro ", str(go2_xacro), " use_gazebo:=false"]),
                "frame_prefix": f"{dog}/",
                "use_sim_time": False,
            }],
            remappings=[("joint_states", f"/task2/{dog}/joint_states")],
            output="screen",
        ),
        Node(
            package="warehouse_core",
            executable="multifloor_rviz_state_bridge",
            name="multifloor_rviz_state_bridge",
            parameters=[{
                "floor1_car": floor1_car,
                "dog": dog,
                "floor2_car": floor2_car,
            }],
            output="screen",
        ),
        Node(
            package="warehouse_core",
            executable="rviz_robot_markers",
            name="rviz_robot_markers",
            parameters=[{
                "floor1_car": floor1_car,
                "dog": dog,
                "floor2_car": floor2_car,
            }],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", str(rviz_config)],
            output="screen",
        ),
    ])
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mission", default_value="ne"),
        OpaqueFunction(function=_setup),
    ])
