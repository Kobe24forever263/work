"""Persistent full-fleet RViz for ten sequential Stage 26 SCAN tasks."""

from copy import deepcopy
import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
import yaml


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


def _topic(display, topic):
    display["Topic"]["Value"] = topic
    return display


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
    global_cloud = next(
        item for item in displays if item.get("Name") == "Warehouse 3D PCD")
    grid = next(item for item in displays if item.get("Name") == "Grid")
    labels = next(
        item for item in displays
        if item.get("Name") == "Robot labels and attached cargo")
    tf_display = next(
        item for item in displays if item.get("Class") == "rviz_default_plugins/TF")
    local_template = next(
        item for item in displays if item.get("Name") == "Go2 Local Sensor Cloud")
    occupancy_template = next(
        item for item in displays if item.get("Name") == "Go2 SCAN Occupancy")
    inflated_template = next(
        item for item in displays if item.get("Name") == "Go2 Inflated Occupancy")
    bbox_template = next(
        item for item in displays if item.get("Name") == "Go2 Sliding Map Bounds")
    path_template = next(
        item for item in displays if item.get("Class") == "rviz_default_plugins/Path")

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
    output.append(global_cloud)
    local = _topic(deepcopy(local_template),
                   "/task2/stage26_scan/local_cloud")
    local["Name"] = "Active SCAN local lidar cloud"
    output.append(local)
    occupancy = _topic(deepcopy(occupancy_template),
                       "/task2/stage26_scan/occupancy")
    occupancy["Name"] = "Active SCAN local occupancy"
    output.append(occupancy)
    inflated = _topic(deepcopy(inflated_template),
                      "/task2/stage26_scan/occupancy_inflate")
    inflated["Name"] = "Active SCAN inflated occupancy"
    output.append(inflated)
    bbox = _topic(deepcopy(bbox_template),
                  "/task2/stage26_scan/sliding_map_bbox")
    bbox["Name"] = "Active SCAN sliding-map bounds"
    output.append(bbox)
    path = _topic(deepcopy(path_template),
                  "/task2/stage26_scan/active_path")
    path["Name"] = "Active SCAN travelled path"
    path["Color"] = "255; 210; 20"
    output.append(path)
    output.append(labels)
    output.append({
        "Class": "rviz_default_plugins/MarkerArray",
        "Enabled": True,
        "Name": "Stage 26 ten-task status",
        "Namespaces": {},
        "Queue Size": 100,
        "Topic": {
            "Depth": 10,
            "Durability Policy": "Volatile",
            "History Policy": "Keep Last",
            "Reliability Policy": "Reliable",
            "Value": "/task2/stage26_scan/campaign_markers",
        },
        "Value": True,
    })
    output.append(tf_display)
    config["Visualization Manager"]["Displays"] = output
    destination = Path("/private/tmp/stage26_scan_campaign.rviz")
    destination.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def _setup(context):
    campaign_file = Path(
        LaunchConfiguration("campaign_file").perform(context)).expanduser()
    if not campaign_file.is_file():
        raise RuntimeError(f"Stage 26 campaign does not exist: {campaign_file}")
    campaign = json.loads(campaign_file.read_text(encoding="utf-8"))
    if campaign.get("scope") != "TEN_SEQUENTIAL_TASKS":
        raise RuntimeError("campaign must contain TEN_SEQUENTIAL_TASKS")
    robots = campaign["robot_ids"]
    robot_ids_csv = ",".join(robots)
    bringup = Path(get_package_share_directory("warehouse_bringup"))
    rviz_config = _runtime_rviz(bringup, robots)
    carter_description = _carter_description()
    go2_xacro = Path(
        "/Users/lab4099/Desktop/Mujoco/work/src/go2_description/xacro/"
        "robot.xacro")
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
            remappings=[("joint_states", f"/task2/{robot}/joint_states")],
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
            parameters=[{"robot_ids_csv": robot_ids_csv}], output="screen"),
        Node(
            package="warehouse_core",
            executable="stage26_scan_campaign_monitor",
            name="stage26_scan_campaign_monitor",
            parameters=[{"campaign_file": str(campaign_file.resolve())}],
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
                reason="Stage 26 SCAN campaign RViz closed"))])),
    ])
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("campaign_file"),
        OpaqueFunction(function=_setup),
    ])
