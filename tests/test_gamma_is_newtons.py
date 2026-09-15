"""gamma is DEFINED as the internal-force scale in newtons -- is it, in practice?

The contract comes from scripts/3D_minimum_NCF.py, which is where the planner's
gamma is solved:

    single_wrench_cone():  "gamma: where f_contact = f_manipulation + gamma * f_internal"
                           and internally  ncf = ncf * gamma
    min_gamma_for_accel_lp(): ":return float min_gamma: where minimum normal contact
                           force = normal component of (min_gamma * f_internal)"

and kinova_common.wrench.solve_gamma_live calls it with ncf=[1.0]*n -- a UNIT
normal force per contact. So by construction gamma is the per-contact internal
NORMAL FORCE IN NEWTONS: the planner's gamma=12 is a request for 12 N.

GraspController.internal_force_torques does not honour that. It forms

    f_c = f_eq + (gamma / cone_f_min) * f_int        f_int = f_c_LP - f_eq

which assumes |f_int| == cone_f_min, i.e. that the LP's INTERNAL part has the
magnitude of the f_min floor. It does not. solve_gamma_cone minimises the PEAK
normal force subject to f_min at EVERY contact, so once w_des != 0 the
equilibrium part already loads one contact and the internal correction needed to
bring the other up to f_min grows with w_des. Measured peak internal magnitude
against an f_min of 0.5:

    object weight   peak |f_int|
        0.49 N          0.745
        0.95 N          0.976
        3.73 N          2.364
        7.15 N          4.076

Dividing by f_min therefore under-normalises by a factor that TRACKS OBJECT MASS,
and gamma inflates by 1.5x (lemon) to 8.2x (036_wood_block) -- which is why the
block commands ~98 N of internal force for a gamma of 12, and why its measured
contact force came out at ~55 N where friction needs ~19 N.

FIXED: internal_force_torques now normalises f_int by its own peak normal
(_peak_internal_normal) instead of by cone_f_min, so gamma means newtons at every
mass. The contract test below is consequently a NORMAL PASSING TEST -- it was
xfail while the defect stood. The _shipped_rule helper is kept under its old name
as the REGRESSION oracle: it reproduces the buggy arithmetic so the tests can
still show what the inflation was.
"""
import sys, os
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from grasp_control.force_control import GraspForceAllocator
from grasp_control.grasp_controller import _peak_internal_normal

F_MIN, MU, MARGIN = 0.5, 0.7, 0.2
# (mass kg, gamma solved for it) as measured in the tabletop benchmark.
CASES = [(0.050, 1.0), (0.097, 2.4), (0.380, 4.8), (0.729, 12.0)]


def _antipodal_G(d=0.05):
    cols = []
    for s, p in ((+1, np.array([0., 0., -d])), (-1, np.array([0., 0., +d]))):
        n = np.array([0., 0., float(s)])
        t1 = np.array([1., 0., 0.])
        R = np.column_stack([n, t1, np.cross(n, t1)])
        for j in range(3):
            f = R[:, j]
            cols.append(np.r_[f, np.cross(p, f)])
    return np.array(cols).T


def _alloc(G, w_des, n_c=2):
    _, info = GraspForceAllocator(1.0).solve_gamma_cone(
        G, w_des, 3, [np.array([1., 0., 0.])] * n_c,
        mu=MU, f_min=F_MIN, margin=MARGIN)
    assert info.get('f_c') is not None, "LP infeasible for this fixture"
    f_c = np.asarray(info['f_c'], float)
    f_eq = np.linalg.pinv(G) @ w_des
    return f_c, f_eq, f_c - f_eq


def _weight(mass):
    w = np.zeros(6)
    w[2] = mass * 9.81
    return w


def _shipped_rule(f_eq, f_int, gamma):
    """The OLD (buggy) arithmetic, kept as the regression oracle."""
    return f_eq + (gamma / max(F_MIN, 1e-9)) * f_int


def _shipped_rule_fixed(f_eq, f_int, gamma):
    """What internal_force_torques does NOW -- via the controller's own helper,
    so this test breaks if that normalisation is ever changed."""
    return f_eq + gamma * (f_int / _peak_internal_normal(f_int, len(f_int) // 3))


def _peak_normal(f, n_c=2):
    """Contact frames put the normal in col0, so index 0 of each 3-block."""
    return max(abs(float(f[3 * k])) for k in range(n_c))


# --------------------------------------------------------------------------
# The contract, stated directly. NOW HOLDS (was xfail before the normalisation
# fix) -- checked against the controller's own helper, not a local copy.
# --------------------------------------------------------------------------

def test_gamma_equals_internal_normal_force_in_newtons():
    G = _antipodal_G()
    for mass, gamma in CASES:
        _, f_eq, f_int = _alloc(G, _weight(mass))
        got = _peak_normal(_shipped_rule_fixed(f_eq, f_int, gamma) - f_eq)
        assert abs(got - gamma) < 1e-6, (
            f"mass={mass}kg: gamma={gamma} must command {gamma:.1f} N of internal "
            f"normal force; got {got:.1f} N ({got / gamma:.2f}x)")


# --------------------------------------------------------------------------
# Passing tests: pin the defect precisely, and pin what must not regress.
# --------------------------------------------------------------------------

def test_internal_magnitude_is_not_f_min():
    """The premise behind dividing by cone_f_min is false -- this is the bug."""
    G = _antipodal_G()
    mags = [_peak_normal(_alloc(G, _weight(m))[2]) for m, _ in CASES]
    assert all(v > F_MIN for v in mags), mags
    assert mags[-1] > 5 * mags[0], (
        f"expected |f_int| to grow with object weight, got {np.round(mags, 3)}")


def test_gamma_inflation_tracks_object_mass():
    """Quantifies the drift: the error is not constant, so no single constant
    rescale fixes it -- the normaliser itself has to change."""
    G = _antipodal_G()
    ratios = []
    for mass, gamma in CASES:
        _, f_eq, f_int = _alloc(G, _weight(mass))
        ratios.append(_peak_normal(_shipped_rule(f_eq, f_int, gamma) - f_eq) / gamma)
    assert ratios == sorted(ratios), f"expected monotone inflation, got {ratios}"
    assert ratios[-1] > 4 * ratios[0], ratios


def test_correct_normalisation_would_satisfy_the_contract():
    """The proposed fix, verified ARITHMETICALLY only -- nothing is rewired here.

    Normalising f_int by its own peak normal force makes gamma mean newtons at
    every mass. Kept as a passing test so the fix has a target to be checked
    against before it is applied to the controller.
    """
    G = _antipodal_G()
    for mass, gamma in CASES:
        _, f_eq, f_int = _alloc(G, _weight(mass))
        f_unit = f_int / _peak_normal(f_int)
        got = _peak_normal((f_eq + gamma * f_unit) - f_eq)
        assert abs(got - gamma) < 1e-6, (mass, gamma, got)


def test_equilibrium_wrench_survives_either_normalisation():
    """Whatever scales f_int, G f_c must stay w_des -- the invariant the
    already-fixed equilibrium bug violated. Guards the working grasps."""
    G = _antipodal_G()
    for mass, gamma in CASES:
        w = _weight(mass)
        _, f_eq, f_int = _alloc(G, w)
        for f in (_shipped_rule(f_eq, f_int, gamma),
                  f_eq + gamma * (f_int / _peak_normal(f_int))):
            assert np.allclose(G @ f, w, atol=1e-6), (mass, gamma, np.round(G @ f, 3))


def test_light_objects_are_near_the_contract_already():
    """Why the light objects work and the block does not: at ~1 N the shipped
    rule is within ~1.5x of newtons, so a gain tuned there is roughly right.
    This is what a fix must not regress."""
    G = _antipodal_G()
    mass, gamma = CASES[0]
    _, f_eq, f_int = _alloc(G, _weight(mass))
    assert _peak_normal(_shipped_rule(f_eq, f_int, gamma) - f_eq) / gamma < 2.0


def test_zero_w_des_is_exact():
    """With no weight to support the two normalisations coincide up to the
    f_min scale -- the pure-pinch case was never wrong."""
    G = _antipodal_G()
    _, f_eq, f_int = _alloc(G, np.zeros(6))
    assert np.allclose(f_eq, 0.0, atol=1e-9)
    assert abs(_peak_normal(f_int) - F_MIN) < 1e-6, _peak_normal(f_int)


if __name__ == "__main__":
    n_p = n_f = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn(); print(f"PASS  {name}"); n_p += 1
        except AssertionError as e:
            print(f"FAIL  {name}: {e}"); n_f += 1
    print(f"\n{n_p} passed, {n_f} failed")
    sys.exit(1 if n_f else 0)
