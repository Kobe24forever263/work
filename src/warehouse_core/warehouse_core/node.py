def main():
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as exc:
        raise SystemExit("ROS 2 Humble environment is required") from exc

    rclpy.init()
    node = Node("rule_scheduler")
    node.get_logger().info("Rule scheduler ready; awaiting configured adapters")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
