# Source this (not execute) to prepare a shell for dexpilot teleop:
#   source setup.sh
#
# Fixes the "eno2: does not match an available interface" CycloneDDS error by
# pinning DDS to loopback — the MediaPipe publisher and the MuJoCo subscriber
# run on the SAME machine, so local loopback is the right (and most robust)
# transport: no dependence on eno1/eno2/wifi link state, lowest latency.

# --- ROS 2 base (Humble). The teleop path uses only std_msgs/geometry_msgs,
#     so the repo overlay is not required; source it too if you later use
#     object_approach_controller. ---
source /opt/ros/humble/setup.bash

# Repo overlay is NOT sourced: the dexpilot teleop path uses only
# std_msgs/geometry_msgs (no custom messages), and ros2_ws/install/setup.bash
# currently has CRLF line endings that break `source`. If you later need the
# object_approach_controller package, fix the line endings
# (sed -i 's/\r$//' ros2_ws/install/setup.bash) and source it manually.

# --- DDS: force CycloneDDS onto loopback for a single-machine session ---
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces>
    <NetworkInterface name="lo" priority="default" multicast="default" />
  </Interfaces></General></Domain></CycloneDDS>'

# Keep publisher and subscriber on the same DDS domain (default 0). Set here so
# an inherited ROS_DOMAIN_ID from another project can't split them apart.
export ROS_DOMAIN_ID=0

echo "[setup] ROS $ROS_DISTRO | RMW=$RMW_IMPLEMENTATION | DDS iface=lo | domain=$ROS_DOMAIN_ID"
echo "[setup] ready. Run: uv run python kinova_leap_pick_place.py --mode dexpilot --camera 0"
