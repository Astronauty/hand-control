#!/usr/bin/env bash
#
# start_teleop.sh — one command per terminal instead of five.
#
#   ./start_teleop.sh link      terminal 1: USB tunnels + launch the headset app
#   ./start_teleop.sh logs      terminal 2: headset app output
#   ./start_teleop.sh hands     terminal 3: hand publisher  (uplink,  9870)
#   ./start_teleop.sh sim [MODE] [flags...]   terminal 4: MuJoCo teleop
#        MODE (default dexpilot): dexpilot | anyteleop | vwj | vwj_upstream |
#             contact_aware_w_dexpilot | contact_aware_w_anyteleop | contact_aware_w_vwj
#             (the baseline conditions; vwj = clean-room whole-arm-hand optimizer,
#              arXiv:2506.09384; vwj_upstream = the authors' optimizer verbatim, A/B partner)
#        MODE is optional — anything starting with '-' is treated as a flag, so
#        `sim --order 3` and `sim anyteleop --order 3` both work.
#        flags... are forwarded (e.g. --trial-log --object obj_red_box)
#        anyteleop modes need:  uv sync --extra anyteleop
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
	# --yaw aligns the operator's hand motion to the robot table frame. The base is yawed +90
	# (arm reaches +y across the table WIDTH); robot-table LENGTH = world X, WIDTH = world Y.
	# yaw 0 maps hand-right -> robot +X (so operator LEFT<->RIGHT matches the robot's
	# left<->right along the LENGTH) and hand-forward -> robot +Y (reach across the width) —
	# verified via `--check-yaw`. If left/right or forward/back comes out reversed, run
	# `python3 teleop/vive_hand_publisher.py --check-yaw` and pick the matching row (0/90/180/270).
	exec python3 teleop/vive_hand_publisher.py --hand "${2:-right}" --frame mujoco --yaw 0
	;;

sim)
	ros_env
	# $2 = MODE (optional, default dexpilot); the rest = extra flags (e.g. --trial-log).
	# MODE is POSITIONAL and OPTIONAL, so it is only consumed when it is actually a mode
	# name — i.e. when it does NOT start with '-'. Without that test a leading flag was
	# swallowed as the mode: `sim --order 3` became `--mode --order ... 3`, which argparse
	# rejects (or worse, misreads the bare value). Both forms now work:
	#     ./start_teleop.sh sim --order 3                 (default mode + flags)
	#     ./start_teleop.sh sim anyteleop --order 3       (explicit mode + flags)
	# --no-mediapipe is always on: the VR headset (./start_teleop.sh hands) is the sole
	# /hand/joint_angles publisher, so the app must never spawn a camera publisher.
	# anyteleop / contact_aware_w_anyteleop need:  uv sync --extra anyteleop
	shift                                  # drop 'sim'; $1 is now MODE-or-first-flag
	case "${1:-}" in
		-*|"") MODE="dexpilot" ;;          # a flag (or nothing): keep the default mode
		*)     MODE="$1"; shift ;;         # a real mode name: consume it
	esac
	# DP_PROFILE=1 prints the per-iteration wall-time breakdown (retarget / step / draw /
	# record) so a "sim gets stuck" stall shows which bucket spiked. An env assignment must
	# precede the command (or the shell tries to exec a program literally named DP_PROFILE=1);
	# `exec env VAR=val cmd` is the exec-safe form. Set to 0 (or drop it) to silence profiling.
	#
	# BLAS threads: OpenBLAS defaults to one thread per core (48 here), which OVER-
	# SUBSCRIBES the grasp NLP's MUMPS/BLAS calls (~18x slower at 48 vs the ~4-thread knee).
	# NOTE: there is NO in-code cap any more. A threadpool_limits pin was tried and REMOVED
	# (ca1102a) because mutating this pthreads OpenBLAS pool at runtime stalled the RRT —
	# lock-in produced no [RRT] output at all. _blas_pin no longer exists; GRASP_BLAS_THREADS
	# is NOT read as a cap, only OPENBLAS_NUM_THREADS is used as a fallback HINT for the
	# thread count recorded in the grasp-solver log. Capping remains an open question: the
	# solver is still slow when contended, and the fix must not mutate the shared pool at
	# runtime (setting OPENBLAS_NUM_THREADS in the ENV before launch is the safe lever, but
	# it is process-wide and previously throttled RRT/numpy in the main loop).
	# CONSOLE CAPTURE. This used to `exec` straight into python with no redirect, so
	# everything the sim prints to the terminal was lost the moment the window scrolled:
	# the [sdf] per-object bake verdicts, [rec] collision-geom counts, [fingers] slots,
	# and — the one that actually cost debugging time — traceback.print_exc() from the
	# BACKGROUND recommender/IK threads, whose failures are otherwise invisible (a thread
	# that dies mid-solve just stops producing recommendations, silently). Tee to a
	# timestamped file under logs/console/ AND to the terminal, so a session can be
	# diagnosed after the fact without changing how it looks while running.
	# PIPESTATUS preserves python's exit code through the pipe (a bare pipeline reports
	# tee's). No `exec` into python: the pipeline needs this shell to stay and wait.
	mkdir -p logs/console
	_CONSOLE_LOG="logs/console/${MODE}_$(date +%Y%m%d_%H%M%S).log"
	echo "[start_teleop] console -> $_CONSOLE_LOG"
	env DP_PROFILE="${DP_PROFILE:-1}" python3 -u kinova_leap_pick_place.py \
		--mode "$MODE" --no-mediapipe "$@" 2>&1 | tee "$_CONSOLE_LOG"
	exit "${PIPESTATUS[0]}"
	;;

viz)
	ros_env
	exec python3 teleop/hand_viz.py
	;;

ego)
	ros_env
	exec python3 teleop/hand_ego_view.py
	;;

tune)
	# Live pinch/retarget tuner: subscribes to /hand/joint_angles and shows the hand +
	# per-finger pinch detection with sliders (EPS, enter/exit, median) that save to
	# teleop/calibration/retarget_config.json. NEEDS the hand publisher running in another
	# terminal (./start_teleop.sh hands) or it just shows "waiting for hand data...".
	# ros_env sources ROS + unsets CYCLONEDDS_URI so the subscription actually receives.
	ros_env
	shift   # drop 'tune'; forward any remaining args (e.g. --topic, --span) to the tuner
	exec python3 teleop/hand_tune.py "$@"
	;;

mock)
	cd "$HERE" || exit 1
	exec python3 teleop/mock_headset.py --motion "${2:-open_close}"
	;;

scene)
	# Re-export the MuJoCo geometry after changing the scene XML. The geom count
	# it prints is the number that decides whether the headset can hold 72 Hz.
	cd "$HERE" || exit 1
	python3 teleop/scene_export.py --xml models/scene_pick_place.xml \
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
	sed -n '3,16p' "$0" | sed 's/^# \{0,1\}//'
	exit 1
	;;
esac
