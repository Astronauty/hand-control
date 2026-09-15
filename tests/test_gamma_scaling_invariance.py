"""Scaling gamma must change only the INTERNAL force, never the equilibrium wrench.

GraspController.internal_force_torques used to scale the allocator's whole output
by gamma/cone_f_min. That vector is

    f_c = pinv(G) w_des  +  N gamma_LP

and only the second term is wrench-neutral (G N = 0). Scaling the first term too
commands a net object wrench of gamma/f_min times w_des -- with support_weight
that is gravity, so the hand pushes the object UP with a multiple of its own
weight. On 036_wood_block the disturbance-budget change took solved gamma ~0.5 ->
12, i.e. a 24x amplifier: the commanded support force reached 171.6 N on a 7.15 N
block, the squeeze ejected it, and the measured contact force was 0.0 N because
nothing was left to touch. Raising finger_kp 0.8 -> 80 did not help, because the
excess is a wrench on the OBJECT rather than a torque the fingers can fight.

These tests pin the invariant directly on the allocator output, independent of
MuJoCo: G @ f_c must equal w_des at every gamma, while the internal part scales.
"""
import sys, os
import numpy as np
try:
    import pytest  # noqa: F401  -- optional; the file also runs standalone
except ImportError:
    pytest = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from grasp_control.force_control import GraspForceAllocator


def _contact_frame(n_in):
    n = np.asarray(n_in, float); n = n / np.linalg.norm(n)
    ref = np.array([0., 0., 1.]) if abs(n[0]) > 0.9 else np.array([1., 0., 0.])
    t1 = np.cross(n, ref); t1 /= np.linalg.norm(t1)
    return np.column_stack([n, t1, np.cross(n, t1)])


def _grasp_map(points, normals):
    cols = []
    for p, n in zip(points, normals):
        R = _contact_frame(n)
        for j in range(3):
            f = R[:, j]
            cols.append(np.r_[f, np.cross(np.asarray(p, float), f)])
    return np.array(cols).T


def _scaled(G, w_des, f_c_lp, gamma, f_min):
    """The fixed rule from internal_force_torques: scale ONLY the null-space part."""
    f_eq = np.linalg.pinv(G) @ w_des
    return f_eq + (gamma / f_min) * (np.asarray(f_c_lp, float) - f_eq)


def _antipodal(d=0.05):
    pts = [np.array([0., 0., -d]), np.array([0., 0., +d])]
    nrm = [np.array([0., 0., +1.]), np.array([0., 0., -1.])]   # inward
    return _grasp_map(pts, nrm)


# 7.15 N = 036_wood_block's weight, the case that regressed.
W_BLOCK = np.array([0., 0., 7.15, 0., 0., 0.])


def test_equilibrium_wrench_is_invariant_to_gamma():
    """G @ f_c == w_des for every gamma -- the property that was violated."""
    G, f_min = _antipodal(), 0.5
    a = GraspForceAllocator(1.0)
    _, info = a.solve_gamma_cone(G, W_BLOCK, 3, [np.array([1., 0., 0.])] * 2,
                                 mu=0.7, f_min=f_min, margin=0.2)
    assert info.get('f_c') is not None
    for gamma in (0.5, 1.0, 2.0, 4.0, 12.0, 27.0):
        w = G @ _scaled(G, W_BLOCK, info['f_c'], gamma, f_min)
        assert np.allclose(w, W_BLOCK, atol=1e-6), (
            f"gamma={gamma} changed the commanded wrench: {np.round(w, 3)} "
            f"!= {W_BLOCK} -- the object is being shoved, not squeezed")


def test_unscaled_rule_amplifies_gravity():
    """The OLD rule really did multiply the support force -- guards the premise."""
    G, f_min = _antipodal(), 0.5
    a = GraspForceAllocator(1.0)
    _, info = a.solve_gamma_cone(G, W_BLOCK, 3, [np.array([1., 0., 0.])] * 2,
                                 mu=0.7, f_min=f_min, margin=0.2)
    w_old = G @ ((12.0 / f_min) * np.asarray(info['f_c'], float))
    assert w_old[2] > 20 * W_BLOCK[2], "expected the old rule to blow gravity up"


def test_internal_force_still_scales_with_gamma():
    """The squeeze must remain controllable: internal part grows with gamma."""
    G, f_min = _antipodal(), 0.5
    a = GraspForceAllocator(1.0)
    _, info = a.solve_gamma_cone(G, W_BLOCK, 3, [np.array([1., 0., 0.])] * 2,
                                 mu=0.7, f_min=f_min, margin=0.2)
    f_eq = np.linalg.pinv(G) @ W_BLOCK
    peaks = [np.abs(_scaled(G, W_BLOCK, info['f_c'], g, f_min) - f_eq).max()
             for g in (1.0, 2.0, 4.0)]
    assert peaks[1] > 1.5 * peaks[0] and peaks[2] > 1.5 * peaks[1], peaks


def test_invariance_holds_for_three_contacts():
    """Same invariant with a 3-D null space, where the LP has real freedom."""
    pts = [np.array([0.05, 0., 0.]), np.array([-0.025, 0.043, 0.]),
           np.array([-0.025, -0.043, 0.])]
    nrm = [-p / np.linalg.norm(p) for p in pts]          # inward
    G, f_min = _grasp_map(pts, nrm), 0.5
    a = GraspForceAllocator(1.0)
    _, info = a.solve_gamma_cone(G, W_BLOCK, 3, [np.array([1., 0., 0.])] * 3,
                                 mu=0.7, f_min=f_min, margin=0.2)
    assert info.get('f_c') is not None
    for gamma in (1.0, 5.0, 12.0):
        w = G @ _scaled(G, W_BLOCK, info['f_c'], gamma, f_min)
        assert np.allclose(w, W_BLOCK, atol=1e-6), (gamma, np.round(w, 3))


def test_zero_w_des_is_unaffected():
    """With w_des = 0 the equilibrium part vanishes, so old and new agree."""
    G, f_min = _antipodal(), 0.5
    w0 = np.zeros(6)
    a = GraspForceAllocator(1.0)
    _, info = a.solve_gamma_cone(G, w0, 3, [np.array([1., 0., 0.])] * 2,
                                 mu=0.7, f_min=f_min, margin=0.2)
    old = (12.0 / f_min) * np.asarray(info['f_c'], float)
    assert np.allclose(_scaled(G, w0, info['f_c'], 12.0, f_min), old, atol=1e-9)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    n_p = n_f = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}"); n_p += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}"); n_f += 1
    print(f"\n{n_p} passed, {n_f} failed")
    sys.exit(1 if n_f else 0)
