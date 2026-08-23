"""Run one Carter route with SCAN-Planner and the warehouse PCD map."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


ROUTES = {
    ("car_f1_1", "ne"): ("car_f1_1_stair_ne_send.yaml", (2.5, 2.5, 0.45)),
    ("car_f1_2", "nw"): ("car_f1_2_stair_nw_send.yaml", (-2.5, 2.5, 0.45)),
    ("car_f1_3", "sw"): ("car_f1_3_stair_sw_send.yaml", (-2.5, -2.5, 0.45)),
    ("car_f1_4", "se"): ("car_f1_4_stair_se_send.yaml", (2.5, -2.5, 0.45)),
    ("car_f2_1", "nw"): ("car_f2_1_stair_nw_receive.yaml", (-5.5, 0.0, 4.70)),
    ("car_f2_1", "sw"): ("car_f2_1_stair_sw_receive.yaml", (-5.5, 0.0, 4.70)),
    ("car_f2_2", "ne"): ("car_f2_2_stair_ne_receive.yaml", (5.5, 0.0, 4.70)),
    ("car_f2_2", "se"): ("car_f2_2_stair_se_receive.yaml", (5.5, 0.0, 4.70)),
}

PHASE_ROUTES = {
    ("car_f1_1", "ne", "pickup"): ("car_f1_1_ne_pickup.yaml", (2.5, 2.5, 0.45)),
    ("car_f1_1", "ne", "send_loaded"): ("car_f1_1_ne_pickup_to_stair.yaml", (12.0, 8.0, 0.45)),
    ("car_f2_1", "nw", "receive"): ("car_f2_1_stair_nw_handover.yaml", (-5.5, 0.0, 4.70)),
    ("car_f2_1", "sw", "receive"): ("car_f2_1_stair_sw_handover.yaml", (-5.5, 0.0, 4.70)),
    ("car_f2_2", "ne", "receive"): ("car_f2_2_stair_ne_handover.yaml", (5.5, 0.0, 4.70)),
    ("car_f2_2", "se", "receive"): ("car_f2_2_stair_se_handover.yaml", (5.5, 0.0, 4.70)),
    ("car_f2_1", "nw", "deliver"): ("car_f2_1_stair_nw_dropoff.yaml", (-9.5, 6.5, 4.70)),
    ("car_f2_1", "sw", "deliver"): ("car_f2_1_stair_sw_dropoff.yaml", (-9.5, -6.5, 4.70)),
    ("car_f2_2", "ne", "deliver"): ("car_f2_2_stair_ne_dropoff.yaml", (12.6, 9.6, 4.70)),
    ("car_f2_2", "se", "deliver"): ("car_f2_2_stair_se_dropoff.yaml", (9.5, -6.5, 4.70)),
    ("car_f2_2", "ne", "standby"): ("car_f2_2_ne_to_standby.yaml", (-7.0, -6.5, 4.70)),
}


def _setup(context):
    robot = LaunchConfiguration("robot").perform(context)
    stair = LaunchConfiguration("stair").perform(context).lower()
    phase = LaunchConfiguration("route_phase").perform(context).lower()
    key = (robot, stair)
    phase_key = (robot, stair, phase)
    if phase == "full" and key in ROUTES:
        route_name, initial = ROUTES[key]
    elif phase_key in PHASE_ROUTES:
        route_name, initial = PHASE_ROUTES[phase_key]
    else:
        valid = ", ".join(f"{r}:{s}" for r, s in ROUTES)
        raise RuntimeError(
            f"unsupported robot/stair/phase; choose full for {valid}, or "
            "pickup/send_loaded/receive/deliver/standby for a supported route")
    offset_x = float(LaunchConfiguration("initial_offset_x").perform(context))
    offset_y = float(LaunchConfiguration("initial_offset_y").perform(context))
    bringup = Path(get_package_share_directory("warehouse_bringup"))
    scan = Path(get_package_share_directory("scan_planner"))
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
            "keypoints_file": str(
                bringup / "config" / "scan_planner" / route_name),
            "planner_overrides_file": str(
                bringup / "config" / "scan_planner" / "carter_tuning.yaml"),
            "map_size_x": "52.0",
            "map_size_y": "44.0",
            "map_size_z": "7.0",
            "init_x": str(initial[0] + offset_x),
            "init_y": str(initial[1] + offset_y),
            "init_z": str(initial[2]),
            "publish_global_map": LaunchConfiguration(
                "publish_global_map").perform(context),
            "publish_robot_state": "false",
            "publish_gait_state": "false",
            "show_rviz": "false",
        }.items(),
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="car_f1_1"),
        DeclareLaunchArgument("stair", default_value="ne"),
        DeclareLaunchArgument("route_phase", default_value="full"),
        DeclareLaunchArgument("publish_global_map", default_value="false"),
        DeclareLaunchArgument("initial_offset_x", default_value="0.0"),
        DeclareLaunchArgument("initial_offset_y", default_value="0.0"),
        OpaqueFunction(function=_setup),
    ])
