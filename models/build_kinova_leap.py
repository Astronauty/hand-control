#!/usr/bin/env python3
"""Build a composite MJCF model combining Kinova Gen3 arm + LEAP right hand.

This script uses MuJoCo's MjSpec API to merge two independent robot specs
(from mujoco_menagerie) into a single MJCF, avoiding XML class/name collisions
by namespacing the LEAP hand's entities with a prefix.
"""

import os
import mujoco


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINOVA_DIR = os.path.join(REPO_ROOT, "mujoco_menagerie/kinova_gen3")
LEAP_DIR = os.path.join(REPO_ROOT, "mujoco_menagerie/leap_hand")

# Mount pose of the LEAP hand relative to bracelet_link.
# TODO(tune): adjust after visual inspection — no official Kinova+LEAP spec exists.
MOUNT_POS = [-.1, -0.05, -0.170]
# MOUNT_QUAT = [1, 0, 0, 0]
MOUNT_QUAT = [0.707, 0, 0.707, 0]  # 90° rotation about y-axis to align LEAP palm with Kinova wrist.

# Fingertip distal bodies, with contact-site positions in each body's local frame.
# Positions taken from the compiled model's geom_pos for if_tip/mf_tip/rf_tip/th_tip
# (the "tip" mesh geom's position already folds in the mesh's own centroid offset,
# i.e. this is where the fingerpad actually sits, not the body origin).
FINGER_TIPS = {
    "index":  ("if_ds", [-0.0013, -0.0336, 0.0145]),
    "middle": ("mf_ds", [-0.0013, -0.0336, 0.0145]),
    "ring":   ("rf_ds", [-0.0013, -0.0336, 0.0145]),
    "thumb":  ("th_ds", [-0.0013, -0.0456, -0.0145]),
}


def _absolutize_meshes(spec, assets_dir):
    """Rewrite mesh file paths to absolute so merging two specs with
    different meshdirs doesn't collide."""
    for mesh in spec.meshes:
        mesh.file = os.path.join(assets_dir, mesh.file)


def _add_fingertip_sites(hand_spec):
    """Add a contact-reference <site> to each LEAP fingertip body (none exist in
    right_hand.xml — only collision geoms). Mirrors the index_touch/thumb_touch
    site convention from models/planar_two_finger_manipulator.xml."""
    for finger, (body_name, pos) in FINGER_TIPS.items():
        body = hand_spec.body(body_name)
        body.add_site(name=f"{body_name}_tip", pos=pos)


def _convert_actuators_to_motor(spec):
    """Replace position-servo actuators with plain torque (motor) actuators, so the
    qfrc_applied-based force controller (internal_force_control.py-style) isn't fought
    by a built-in position spring. Mirrors planar_two_finger_manipulator.xml, which
    deliberately uses <motor> (not <position>) actuators for the same reason.

    set_to_motor() alone leaves a stale ctrllimited/ctrlrange inherited from the old
    position actuator (e.g. the old joint-angle range), which would clamp torque
    commands to near-zero — so ctrllimited must be explicitly cleared. Real torque
    limits come from forcerange (gen3 already defines sane ones; LEAP's are unset).
    """
    for act in spec.actuators:
        act.set_to_motor()
        act.ctrllimited = False


def _build(use_motors: bool, out_name: str):
    arm_spec = mujoco.MjSpec.from_file(os.path.join(KINOVA_DIR, "gen3.xml"))
    hand_spec = mujoco.MjSpec.from_file(os.path.join(LEAP_DIR, "right_hand.xml"))

    _absolutize_meshes(arm_spec, os.path.join(KINOVA_DIR, "assets"))
    _absolutize_meshes(hand_spec, os.path.join(LEAP_DIR, "assets"))

    _add_fingertip_sites(hand_spec)

    if use_motors:
        _convert_actuators_to_motor(arm_spec)
        _convert_actuators_to_motor(hand_spec)

    bracelet = arm_spec.body("bracelet_link")
    mount_site = bracelet.add_site(name="hand_mount", pos=MOUNT_POS, quat=MOUNT_QUAT)
    arm_spec.attach(hand_spec, site=mount_site, prefix="leap_")

    for key in list(arm_spec.keys):
        arm_spec.delete(key)

    out_path = os.path.join(REPO_ROOT, "models", out_name)
    with open(out_path, "w") as f:
        f.write(arm_spec.to_xml())
    print(f"Successfully wrote {out_path}")


def main():
    # Torque-control variant — used by kinova_leap_pick_place.py (qfrc_applied force control)
    _build(use_motors=True,  out_name="kinova_leap.xml")
    # Position-servo variant — used by kinova_leap_rrt_pos_test.py (data.ctrl path following)
    _build(use_motors=False, out_name="kinova_leap_pos.xml")


if __name__ == "__main__":
    main()
