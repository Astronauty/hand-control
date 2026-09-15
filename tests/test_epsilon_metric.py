"""Ferrari-Canny epsilon against closed-form values.

The four-contact cross grasp has an analytic epsilon, which is what makes it the
right fixture: two antipodal pairs on orthogonal axes at radius r resist a pure
torque about z ONLY through tangential friction at moment arm r, and the pyramidal
cone's tangential vertices are single-axis, so the binding face gives `2*mu*r`.
Every other direction is wider. Checking against a number derived independently of
the implementation is the only way to know the hull offsets carry the sign and
normalization the definition wants.

Runs under pytest or standalone (`python tests/test_epsilon_metric.py`), matching
the convention in test_gamma_is_newtons.py.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.epsilon_metric import epsilon_quality, epsilon_subspace  # noqa: E402


def _frame(n):
    """Contact frame with col0 = the given (inward) normal."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    a = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1)
    return np.column_stack([n, t1, np.cross(n, t1)])


def _cross_grasp(r=0.03):
    """Four contacts, two antipodal pairs on the x and y axes. Spans 6D."""
    p = [np.array([-r, 0, 0]), np.array([r, 0, 0]),
         np.array([0, -r, 0]), np.array([0, r, 0])]
    R = [_frame([1, 0, 0]), _frame([-1, 0, 0]),
         _frame([0, 1, 0]), _frame([0, -1, 0])]
    return p, R


def test_epsilon_matches_closed_form_in_mu():
    p, R = _cross_grasp(r=0.03)
    for mu in (0.3, 0.7, 1.0):
        got = epsilon_quality(p, R, [mu] * 4)
        assert got["force_closure"] is True, got
        assert abs(got["epsilon"] - 2.0 * mu * 0.03) < 1e-9, (mu, got["epsilon"])


def test_epsilon_matches_closed_form_in_radius():
    for r in (0.02, 0.03, 0.05):
        p, R = _cross_grasp(r=r)
        eps = epsilon_quality(p, R, [0.7] * 4)["epsilon"]
        assert abs(eps - 2.0 * 0.7 * r) < 1e-9, (r, eps)


def test_epsilon_is_linear_in_gamma():
    """Every cone vertex scales with gamma, so the inscribed radius must too.

    This is why the benchmark scores at a FIXED gamma=1.0: reporting each arm at
    its own solved gamma would confound grasp geometry with squeeze force.
    """
    p, R = _cross_grasp()
    e1 = epsilon_quality(p, R, [0.7] * 4, gamma=1.0)["epsilon"]
    e2 = epsilon_quality(p, R, [0.7] * 4, gamma=2.0)["epsilon"]
    assert abs(e2 - 2.0 * e1) < 1e-9, (e1, e2)


def test_two_contact_pinch_is_degenerate_not_zero():
    """A pinch is rank-5-of-6; epsilon is UNDEFINED, not small.

    Reporting 0.0 here would read as "closure, but weak". The wrench set is flat
    in the grasp-axis torque direction and contains no 6-ball at all -- the same
    structural fact that gates project_grasp_axis_torque in the gamma certificate.
    """
    r = 0.03
    p = [np.array([-r, 0, 0]), np.array([r, 0, 0])]
    R = [_frame([1, 0, 0]), _frame([-1, 0, 0])]
    got = epsilon_quality(p, R, [0.7, 0.7])
    assert got["degenerate"] is True, got
    assert got["epsilon"] is None, got
    assert got["force_closure"] is False, got
    # The force subspace is still full rank and still informative.
    assert epsilon_subspace(p, R, [0.7, 0.7], which="force") > 0.0


def test_tetrahedral_grasp_is_force_closure():
    """A non-degenerate 4-contact case that is not axis-aligned."""
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    got = epsilon_quality([0.03 * v for v in V], [_frame(-v) for v in V],
                          [0.7] * 4)
    assert got["degenerate"] is False, got
    assert got["force_closure"] is True, got
    assert got["epsilon"] > 0.0, got


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
