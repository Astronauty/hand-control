#!/usr/bin/env bash
#
# start_teleop.sh — one command per terminal instead of five.
#
#   ./start_teleop.sh link      terminal 1: USB tunnels + launch the headset app
#   ./start_teleop.sh logs      terminal 2: headset app output
#   ./start_teleop.sh hands     terminal 3: hand publisher  (uplink,  9870)
#   ./start_teleop.sh sim       terminal 4: MuJoCo teleop
#   ./start_teleop.sh viz       optional:   skeleton view
#   ./start_teleop.sh ego       optional:   what the tracker sees
#   ./start_teleop.sh mock      no headset: synthetic hand into the real socket
#   ./start_teleop.sh check     one-shot health check, then exits
#
# Run `link` first and leave it; everything else can start in any order.
#
set +u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADB="$HOME/Android/Sdk/platform-tools/adb"
PKG="edu.aipex.handtracker"
HAND_PORT=9870      # headset -> workstation, hand joints
SCENE_PORT=9871     # workstation -> headset, MuJoCo geom poses
ROS_SETUP=/opt/ros/humble/setup.bash

# CYCLONEDDS_URI is pinned to eno1 somewhere in the shell startup, and that
# interface does not exist on this machine — rmw_create_node then fails with
# "does not match an available interface". Unsetting is per-shell, so it has to
# happen in every terminal, which is most of why this script exists.
ros_env() {
	# shellcheck disable=SC1090
	[ -f "$ROS_SETUP" ] && source "$ROS_SETUP"
	unset CYCLONEDDS_URI
	cd "$HERE" || exit 1
}

need_device() {
	if ! "$ADB" get-state >/dev/null 2>&1; then
		echo "no device. Check in order:"
		echo "  1. headset awake and plugged into a REAR usb port"
		echo "  2. lsusb | grep -i htc            (nothing -> cable/power)"
		echo "  3. sudo dmesg -w, then replug     (over-current -> full shutdown)"
		echo "  4. USB mode 'File Transfer' in the headset, debugging on"
		exit 1
	fi
}

case "${1:-}" in

link)
	need_device
	"$ADB" reverse "tcp:$HAND_PORT" "tcp:$HAND_PORT"
	"$ADB" reverse "tcp:$SCENE_PORT" "tcp:$SCENE_PORT"
	echo "tunnels:"; "$ADB" reverse --list
	"$ADB" shell monkey -p "$PKG" 1 >/dev/null 2>&1
	echo "launched $PKG — put the headset ON (OpenXR only feeds a focused session)"
	echo "leave this terminal open; re-run after any replug"
	;;

logs)
	need_device
	"$ADB" logcat -c && "$ADB" logcat -s godot
	;;

hands)
	ros_env
	exec python3 ui/vive_hand_publisher.py --hand "${2:-right}" --frame mujoco
	;;

sim)
	ros_env
	exec python3 kinova_leap_pick_place.py --mode dexpilot --no-mediapipe
	;;

viz)
	ros_env
	exec python3 ui/hand_viz.py
	;;

ego)
	ros_env
	exec python3 ui/hand_ego_view.py
	;;

mock)
	cd "$HERE" || exit 1
	exec python3 ui/mock_headset.py --motion "${2:-open_close}"
	;;

scene)
	# Re-export the MuJoCo geometry after changing the scene XML. The geom count
	# it prints is the number that decides whether the headset can hold 72 Hz.
	cd "$HERE" || exit 1
	python3 ui/scene_export.py --xml models/scene_pick_place.xml \
		--out godot_scene/ --group-max "${2:-2}"
	echo
	echo "copy godot_scene/* into the Godot project's scene/ folder, then"
	echo "re-export the APK (scene.json needs *.json in the resource filter)"
	;;

check)
	echo "== device =="
	"$ADB" devices | sed 1d | grep -v '^$' || echo "  none"
	echo "== tunnels =="
	"$ADB" reverse --list 2>/dev/null || echo "  none"
	echo "== app =="
	if [ -n "$("$ADB" shell pidof "$PKG" 2>/dev/null | tr -d '\r')" ]; then
		echo "  running"
	else
		echo "  NOT running — ./start_teleop.sh link"
	fi
	echo "== ros =="
	# shellcheck disable=SC1090
	[ -f "$ROS_SETUP" ] && source "$ROS_SETUP"
	unset CYCLONEDDS_URI
	timeout 5 ros2 topic hz /hand/joint_angles 2>/dev/null | head -2 \
		|| echo "  no data on /hand/joint_angles"
	;;

*)
	sed -n '3,13p' "$0" | sed 's/^# \{0,1\}//'
	exit 1
	;;
esac
