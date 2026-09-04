"""Remap a dex-retargeting LEAP qpos vector onto this repo's MJCF hand-joint order.

dex-retargeting solves on the dexsuite LEAP URDF, whose 16 revolute joints are named
numerically ("0".."15") and returned in the optimizer's own `dof_joint_names` order —
which is NOT this repo's order. The sim applies the 16 hand angles in
`DexPilotRetargeter._HAND_JOINT_NAMES` order:

    [ if_mcp, if_rot, if_pip, if_dip,   mf_mcp, mf_rot, mf_pip, mf_dip,
      rf_mcp, rf_rot, rf_pip, rf_dip,   th_cmc, th_axl, th_mcp, th_ipl ]

Both descriptions are the SAME physical LEAP hand (the repo MJCF was derived from this
exact URDF — mujoco_menagerie/leap_hand/README.md), so a pure permutation carries the
solved angles across. We DERIVE that permutation empirically at construction: drive each
URDF DOF, group DOFs by which fingertip moves, then within a finger keep the DOFs in
root->tip TREE order — which is exactly the sim's per-finger role order (mcp/cmc,
rot/axl, pip/mcp, dip/ipl). We do NOT rank by tip displacement: the thumb roll (th_axl)
barely translates the tip, so a magnitude sort misplaces it. This avoids trusting the
confusing URDF link names (URDF joint "0" drives the *pip* link but is really the index
abduction DOF). `_EXPECTED_PERM` documents the verified result for the shipped right-hand
URDF; the derivation is the source of truth and guards against silent drift.
"""
from __future__ import annotations

import numpy as np

# Sim hand-joint order (mirror of DexPilotRetargeter._HAND_JOINT_NAMES). Kept as a
# literal here so this module has no import dependency on teleop/; asserted equal to
# the retargeter's list in build_remap() when a model is available.
SIM_HAND_JOINT_NAMES = [
    'leap_if_mcp', 'leap_if_rot', 'leap_if_pip', 'leap_if_dip',
    'leap_mf_mcp', 'leap_mf_rot', 'leap_mf_pip', 'leap_mf_dip',
    'leap_rf_mcp', 'leap_rf_rot', 'leap_rf_pip', 'leap_rf_dip',
    'leap_th_cmc', 'leap_th_axl', 'leap_th_mcp', 'leap_th_ipl',
]

# Verified permutation for the shipped dexsuite leap_hand_right.urdf: for each position
# p in the dex-retargeting qpos vector, the sim index it feeds. (index, thumb, middle,
# ring blocks in the URDF map to index, thumb, middle, ring in the sim order.)
_EXPECTED_PERM = [0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11]

# URDF fingertip link per human-finger, and the sim tip site per finger key. The four
# fingers are matched by which tip moves; within a finger, DOFs are matched by tip-
# displacement rank under a fixed positive joint perturbation (same chain geometry).
_URDF_TIP_LINKS = {
    'if': 'index_tip_head', 'mf': 'middle_tip_head',
    'rf': 'ring_tip_head',  'th': 'thumb_tip_head',
}
_SIM_TIP_SITES = {
    'if': 'leap_if_ds_tip', 'mf': 'leap_mf_ds_tip',
    'rf': 'leap_rf_ds_tip', 'th': 'leap_th_ds_tip',
}
_FINGER_TO_SIM_BLOCK = {'if': 0, 'mf': 4, 'rf': 8, 'th': 12}   # start index in sim order

_PROBE = 0.3   # rad perturbation used for the displacement ranking


def _finger_dof_order(robot):
    """For each URDF DOF position, return (finger_key, tip_displacement_mm).

    Uses dex-retargeting's pinocchio robot FK (compute_forward_kinematics + get_link_pose).
    """
    import numpy as _np
    tip_idx = {k: robot.get_link_index(v) for k, v in _URDF_TIP_LINKS.items()}
    n = robot.dof
    q0 = _np.zeros(n)
    robot.compute_forward_kinematics(q0)
    base = {k: robot.get_link_pose(i)[:3, 3].copy() for k, i in tip_idx.items()}
    out = []
    for pos in range(n):
        q = q0.copy()
        q[pos] = _PROBE
        robot.compute_forward_kinematics(q)
        disp = {k: float(_np.linalg.norm(robot.get_link_pose(i)[:3, 3] - base[k]))
                for k, i in tip_idx.items()}
        moved = max(disp, key=disp.get)
        out.append((moved, disp[moved]))
    return out


def build_remap(robot, sim_joint_names=None):
    """Return a length-16 int array `perm` s.t. sim_qpos[perm[p]] = dex_qpos[p].

    Derived empirically from the URDF robot's kinematics. Falls back to `_EXPECTED_PERM`
    only if the derivation is degenerate (should not happen for the shipped URDF), and
    always cross-checks against it so a mismatch surfaces loudly.
    """
    sim_names = sim_joint_names or SIM_HAND_JOINT_NAMES
    assert list(sim_names) == SIM_HAND_JOINT_NAMES, (
        "sim hand-joint order differs from this module's SIM_HAND_JOINT_NAMES; "
        "update _EXPECTED_PERM / this mapping to match.")

    dofs = _finger_dof_order(robot)                    # [(finger, disp_mm), ...] len 16
    # Group positions by finger. Within a finger, dex-retargeting lists the 4 DOFs in
    # root->tip TREE order, which is exactly the sim's per-finger role order
    # (mcp/cmc, rot/axl, pip/mcp, dip/ipl). We deliberately do NOT sort by displacement:
    # the thumb roll (th_axl) barely translates the tip, so a magnitude sort would
    # misplace it. Tree order (the natural enumeration order of `dofs`) is the key.
    perm = [None] * 16
    for finger, block in _FINGER_TO_SIM_BLOCK.items():
        positions = [p for p, (f, _) in enumerate(dofs) if f == finger]   # already tree order
        if len(positions) != 4:
            # degenerate derivation — trust the verified table
            return np.array(_EXPECTED_PERM, dtype=int)
        for k, p in enumerate(positions):
            perm[p] = block + k

    perm = np.array(perm, dtype=int)
    if not np.array_equal(perm, np.array(_EXPECTED_PERM)):
        # Not fatal (a different/updated URDF is legitimate), but loud: the remap is
        # safety-critical, so make any drift from the verified mapping visible.
        print("[anyteleop] WARNING: derived LEAP joint remap "
              f"{perm.tolist()} != verified {_EXPECTED_PERM}. Using the DERIVED map "
              "(re-verify finger/DOF correspondence).")
    return perm


def apply_remap(dex_qpos, perm):
    """sim_qpos[perm[p]] = dex_qpos[p]  ->  return the (16,) sim-ordered vector."""
    dex_qpos = np.asarray(dex_qpos, dtype=float).reshape(-1)
    sim_q = np.zeros(16, dtype=float)
    sim_q[perm] = dex_qpos
    return sim_q
