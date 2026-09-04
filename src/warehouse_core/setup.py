from setuptools import find_packages, setup

package_name = "warehouse_core"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "PyYAML", "numpy", "gymnasium"],
    zip_safe=True,
    maintainer="Warehouse Team",
    maintainer_email="maintainer@example.invalid",
    description="Warehouse domain and rule baseline",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "rule_scheduler = warehouse_core.node:main",
        "carter_state_bridge = warehouse_core.carter_state_bridge:main",
        "go2_gazebo_pose_bridge = warehouse_core.go2_gazebo_pose_bridge:main",
        "multifloor_rviz_state_bridge = warehouse_core.multifloor_rviz_state_bridge:main",
        "rviz_robot_markers = warehouse_core.rviz_robot_markers:main",
        "stage26_rviz_mission_player = warehouse_core.stage26_rviz_mission_player:main",
        "stage26_scan_campaign_monitor = warehouse_core.stage26_scan_campaign_monitor:main",
        "spatial_manager = warehouse_core.spatial_manager_node:main",
    ]},
)
