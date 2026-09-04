"""Display one Stage 26 random-task policy decision and the full fleet."""

from copy import deepcopy
import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, OpaqueFunction,
    RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
import yaml


EXTRA_DISPLAYS = """
    - Class: rviz_default_plugins/MarkerArray
      Enabled: true
      Name: Stage 26 random task and status
      Namespaces: {}
      Queue Size: 100
      Topic:
        Depth: 10
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /task2/stage26_acceptance_markers
      Value: true
    - Class: rviz_default_plugins/Path
      Color: 40; 180; 255
      Enabled: true
      Line Style: Lines
      Line Width: 0.09
      Name: Stage 26 floor-1 Carter route
      Pose Style: None
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /task2/stage26/path_floor1_car
      Value: true
    - Class: rviz_default_plugins/Path
      Color: 255; 220; 30
      Enabled: true
      Line Style: Lines
      Line Width: 0.09
      Name: Stage 26 Go2 route
      Pose Style: None
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /task2/stage26/path_dog
      Value: true
    - Class: rviz_default_plugins/Path
      Color: 20; 235; 210
      Enabled: true
      Line Style: Lines
      Line Width: 0.09
      Name: Stage 26 floor-2 Carter route
      Pose Style: None
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /task2/stage26/path_floor2_car
      Value: true
"""


def _carter_description():
    source = Path(
        "/Users/lab4099/Desktop/Mujoco/work/src/carter_description/urdf/"
        "carter.urdf")
    text = source.read_text(encoding="utf-8").replace(
        "package://carter/meshes",
        "file:///Users/lab4099/Desktop/Mujoco/work/src/"
        "carter_description/meshes")
    return text.replace(
        '<material name="gray"/>',
        '<material name="carter_blue"><color rgba="0.04 0.32 0.92 1"/>'
        '</material>').replace(
        '<material name="gray" />',
        '<material name="carter_blue"><color rgba="0.04 0.32 0.92 1"/>'
        '</material>').replace(
        '<material name="black"/>',
        '<material name="carter_black"><color rgba="0.03 0.04 0.07 1"/>'
        '</material>').replace(
        '<material name="black" />',
        '<material name="carter_black"><color rgba="0.03 0.04 0.07 1"/>'
        '</material>')


def _runtime_rviz(bringup: Path, robots):
    config = yaml.safe_load(
        (bringup / "config" / "heterogeneous.rviz").read_text(
            encoding="utf-8"))
    displays = config["Visualization Manager"]["Displays"]
    car_template = next(
        item for item in displays if item.get("Name", "").startswith("Carter"))
    dog_template = next(
        item for item in displays if item.get("Name", "").startswith("Go2 ")
        and item.get("Class") == "rviz_default_plugins/RobotModel")
    grid = next(item for item in displays if item.get("Name") == "Grid")
    global_cloud = next(
        item for item in displays if item.get("Name") == "Warehouse 3D PCD")
    labels = next(
        item for item in displays
        if item.get("Name") == "Robot labels and attached cargo")
    tf_display = next(
        item for item in displays
        if item.get("Class") == "rviz_default_plugins/TF")

    output = [grid]
    for robot in robots:
        template = dog_template if robot.startswith("dog_") else car_template
        model = deepcopy(template)
        model["Name"] = (
            f"Go2 {robot} URDF" if robot.startswith("dog_")
            else f"Carter {robot} URDF")
        model["Description Topic"]["Value"] = f"/{robot}/robot_description"
        model["TF Prefix"] = robot
        output.append(model)
    output.extend([global_cloud, labels])
    extras = yaml.safe_load(EXTRA_DISPLAYS)
    if not isinstance(extras, list):
        raise RuntimeError("Stage 26 RViz extra displays are invalid")
    output.extend(extras)
    output.append(tf_display)
    config["Visualization Manager"]["Displays"] = output
    destination = Path("/private/tmp/stage26_random_single_task.rviz")
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def _setup(context):
    plan_file = Path(
        LaunchConfiguration("plan_file").perform(context)).expanduser()
    if not plan_file.exists():
        raise RuntimeError(f"Stage 26 plan does not exist: {plan_file}")
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if plan.get("scope") != "ONE_RANDOM_TASK_ONLY":
        raise RuntimeError("only ONE_RANDOM_TASK_ONLY plans are accepted")
    display = plan["display_robots"]
    robots = tuple(plan.get("robot_ids", plan["robot_initial_poses"].keys()))
    expected = {
        "car_f1_1", "car_f1_2", "car_f1_3", "car_f1_4",
        "dog_1", "dog_2", "dog_3", "dog_4", "car_f2_1", "car_f2_2",
    }
    if len(robots) != 10 or set(robots) != expected:
        raise RuntimeError("single-task RViz plan must contain all 10 robots")
    robot_ids_csv = ",".join(robots)
    bringup = Path(get_package_share_directory("warehouse_bringup"))
    carter_description = _carter_description()
    go2_xacro = Path(
        "/Users/lab4099/Desktop/Mujoco/work/src/go2_description/xacro/"
        "robot.xacro")

    rviz_config = _runtime_rviz(bringup, robots)

    actions = [Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="map_to_world", arguments=[
            "0", "0", "0", "0", "0", "0", "map", "world"],
        parameters=[{"use_sim_time": False}], output="screen")]
    for robot in robots:
        description = (Command(
            ["xacro ", str(go2_xacro), " use_gazebo:=false"])
            if robot.startswith("dog_") else carter_description)
        actions.append(Node(
            package="robot_state_publisher",
            executable="robot_state_publisher", namespace=robot,
            name=f"{robot}_robot_state_publisher",
            parameters=[{
                "robot_description": description,
                "frame_prefix": f"{robot}/", "use_sim_time": False}],
            remappings=[(
                "joint_states", f"/task2/{robot}/joint_states")],
            output="screen"))
    actions.extend([
        Node(
            package="warehouse_core",
            executable="multifloor_rviz_state_bridge",
            name="multifloor_rviz_state_bridge",
            parameters=[{"robot_ids_csv": robot_ids_csv}], output="screen"),
        Node(
            package="warehouse_core", executable="rviz_robot_markers",
            name="rviz_robot_markers",
            parameters=[{"robot_ids_csv": robot_ids_csv}],
            output="screen"),
        Node(
            package="warehouse_core",
            executable="stage26_rviz_mission_player",
            name="stage26_rviz_mission_player",
            parameters=[{
                "plan_file": str(plan_file.resolve()),
                "time_scale": float(
                    LaunchConfiguration("time_scale").perform(context))}],
            output="screen"),
    ])
    rviz = Node(
        package="rviz2", executable="rviz2",
        arguments=["-d", str(rviz_config)], output="screen")
    actions.extend([
        rviz,
        RegisterEventHandler(OnProcessExit(
            target_action=rviz,
            on_exit=[EmitEvent(event=Shutdown(
                reason="Stage 26 RViz window closed"))])),
    ])
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "plan_file", default_value=(
                "/Users/lab4099/Desktop/Mujoco/work/results/"
                "stage26_recurrent_causal/rviz_single_task/latest_plan.json")),
        DeclareLaunchArgument("time_scale", default_value="1.0"),
        OpaqueFunction(function=_setup),
    ])
