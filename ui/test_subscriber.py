import rclpy
from std_msgs.msg import Float32MultiArray

JOINT_ORDER = [
    "thumb_cmc", "thumb_mcp", "thumb_ip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
]

def listener_callback(msg):
    """
    Parse message format:
    [wrist_x, wrist_y, wrist_z,           # Position (3 values)
     wrist_z_euler, wrist_y_euler, wrist_x_euler,  # Orientation ZYX Euler (3 values)
     joint1_yaw, joint1_pitch, joint1_roll,         # Joint 1 (3 values)
     joint2_yaw, joint2_pitch, joint2_roll,         # Joint 2 (3 values)
     ...
    ]
    """
    data = msg.data
    
    # Expected message length
    expected_len = 6 + len(JOINT_ORDER) * 3  # 6 for wrist, 3 per joint
    
    if len(data) != expected_len:
        print(f"⚠️  Message length mismatch!")
        print(f"   Expected: {expected_len} (6 wrist + {len(JOINT_ORDER)*3} joints)")
        print(f"   Received: {len(data)}")
        return
    
    # Clear screen (optional)
    print("\033[H\033[J", end="")
    
    # Parse wrist pose
    wrist_pos = data[0:3]
    wrist_orient = data[3:6]  # ZYX Euler angles in degrees
    
    # Parse joint angles
    joint_data_start = 6
    
    # Display Header
    print("=" * 70)
    print("  HAND POSE & JOINT ANGLES")
    print("=" * 70)
    
    # Display Wrist Pose
    print("\n🤚 WRIST POSE (Camera Frame)")
    print("-" * 70)
    print(f"  Position:    X={wrist_pos[0]:7.3f}, Y={wrist_pos[1]:7.3f}, Z={wrist_pos[2]:7.3f}")
    print(f"  Orientation: Z={wrist_orient[0]:7.1f}°, Y={wrist_orient[1]:7.1f}°, X={wrist_orient[2]:7.1f}°")
    print(f"               (ZYX Euler)")
    
    # Display Joint Angles
    print("\n🦾 JOINT ANGLES")
    print("-" * 70)
    print(f"{'JOINT NAME':<25} | {'YAW':>7} | {'PITCH':>7} | {'ROLL':>7}")
    print("-" * 70)
    
    for i, joint_name in enumerate(JOINT_ORDER):
        base_idx = joint_data_start + (i * 3)
        yaw = data[base_idx]
        pitch = data[base_idx + 1]
        roll = data[base_idx + 2]
        
        print(f"{joint_name:<25} | {yaw:7.1f}° | {pitch:7.1f}° | {roll:7.1f}°")
    
    print("-" * 70)
    print("Press Ctrl+C to exit\n")


def main(args=None):
    rclpy.init(args=args)

    # Create a standard node without a class definition
    node = rclpy.create_node('simple_hand_listener')

    subscription = node.create_subscription(
        Float32MultiArray,
        '/hand/joint_angles',
        listener_callback,
        10
    )
    
    print("Listening on /hand/joint_angles...")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()