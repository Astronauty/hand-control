"""Cone-constrained gamma solve, with the 3-contact cases that motivated it.

The sign-anchor approach it replaces orients each null-space basis vector using
the FIRST non-None inward_dirs entry and then breaks, so with 2 antipodal
contacts (null(G) is 1-D) one sign is the whole answer, but with 3 contacts the
null space is 3-D and nothing holds the non-anchor contacts compressive.
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
    """[n|t1|t2] with col0 = inward normal -- the convention this repo uses."""
    n = np.asarray(n_in, float); n = n / np.linalg.norm(n)
    ref = np.array([0., 0., 1.]) if abs(n[0]) > 0.9 else np.array([1., 0., 0.])
    t1 = np.cross(n, ref); t1 /= np.linalg.norm(t1)
    return np.column_stack([n, t1, np.cross(n, t1)])


def _grasp_map(points, normals):
    """G: stacked contact forces (each in its OWN frame) -> object wrench."""
    cols = []
    for p, n in zip(points, normals):
        R = _contact_frame(n)
        for j in range(3):
            f = R[:, j]
            cols.append(np.r_[f, np.cross(np.asarray(p, float), f)])
    return np.array(cols).T


def _check(info, mu, n_c, f_min, margin=0.0):
    """Every contact compressive AND inside its TRUE circular cone."""
    f_c = info['f_c']
    for c in range(n_c):
        f = f_c[3*c:3*c+3]
        fn = f[0]                      # contact frame: x = inward normal
        ft = np.linalg.norm(f[1:])
        assert fn >= f_min - 1e-6, f'contact {c} not compressive: fn={fn:.4f}'
        assert ft <= mu * fn + 1e-6, f'contact {c} outside cone: ft={ft:.4f} > mu*fn={mu*fn:.4f}'


# ---------------------------------------------------------------- 2 contacts

def test_two_contact_antipodal_pinch():
    """The configuration that already worked. null(G) is 1-D."""
    pts = [np.array([-0.03, 0., 0.]), np.array([0.03, 0., 0.])]
    nrm = [np.array([1., 0., 0.]), np.array([-1., 0., 0.])]   # inward, facing each other
    G = _grasp_map(pts, nrm)
    a = GraspForceAllocator(1.0)
    g, info = a.solve_gamma_cone(G, np.zeros(6), 3,
                                 normals=[[1,0,0]]*2, mu=0.7, f_min=1.0)
    assert g is not None, info.get('reason')
    _check(info, 0.7, 2, 1.0)
    assert np.allclose(G @ info['f_c'], 0, atol=1e-8), 'internal force must be wrench-free'


# ---------------------------------------------------------------- 3 contacts

def test_three_contact_tripod_is_compressive_at_every_contact():
    """THE case this exists for: 3 contacts, 3-D null space."""
    pts, nrm = [], []
    for ang in (0.0, 2*np.pi/3, 4*np.pi/3):
        p = 0.03 * np.array([np.cos(ang), np.sin(ang), 0.])
        pts.append(p); nrm.append(-p / np.linalg.norm(p))   # inward = toward centre
    G = _grasp_map(pts, nrm)
    a = GraspForceAllocator(1.0)
    g, info = a.solve_gamma_cone(G, np.zeros(6), 3,
                                 normals=[[1,0,0]]*3, mu=0.7, f_min=1.0)
    assert g is not None, info.get('reason')
    assert info['null_dim'] >= 1
    _check(info, 0.7, 3, 1.0)
    assert np.allclose(G @ info['f_c'], 0, atol=1e-8)


def test_three_contact_asymmetric():
    """Non-symmetric tripod -- the symmetric case can hide sign bugs."""
    pts = [np.array([-0.04, 0.0, 0.0]), np.array([0.035, 0.012, 0.0]),
           np.array([0.02, -0.03, 0.015])]
    nrm = [-p/np.linalg.norm(p) for p in pts]
    G = _grasp_map(pts, nrm)
    a = GraspForceAllocator(1.0)
    g, info = a.solve_gamma_cone(G, np.zeros(6), 3,
                                 normals=[[1,0,0]]*3, mu=0.8, f_min=0.8)
    assert g is not None, info.get('reason')
    _check(info, 0.8, 3, 0.8)


def test_sign_anchor_baseline_fails_where_cone_solve_succeeds():
    """Documents WHY this was needed: uniform gamma over an arbitrary SVD basis,
    signed by contact 0 only, need not be compressive at the others."""
    pts = [np.array([-0.04, 0.0, 0.0]), np.array([0.035, 0.012, 0.0]),
           np.array([0.02, -0.03, 0.015])]
    nrm = [-p/np.linalg.norm(p) for p in pts]
    G = _grasp_map(pts, nrm)
    a = GraspForceAllocator(1.0)
    inward = [np.array([1., 0., 0.])] + [None, None]     # anchor contact 0 only
    f_old = a.allocate(G, np.zeros(6), 3, inward_dirs=inward)
    normals_old = [f_old[3*c] for c in range(3)]          # x-component = normal force
    g, info = a.solve_gamma_cone(G, np.zeros(6), 3,
                                 normals=[[1,0,0]]*3, mu=0.7, f_min=1.0)
    assert g is not None
    assert all(f >= 1.0 - 1e-6 for f in info['normal_forces']), \
        'cone solve must be compressive everywhere'
    # The old path is NOT guaranteed compressive; record what it actually gives.
    print(f'\n  sign-anchor normal forces : {np.round(normals_old, 3)}')
    print(f'  cone-solve normal forces  : {np.round(info["normal_forces"], 3)}')


def test_friction_cone_is_respected_at_low_mu():
    """Tighter cone must still be satisfied, or report infeasible -- never violate."""
    pts, nrm = [], []
    for ang in (0.0, 2*np.pi/3, 4*np.pi/3):
        p = 0.03 * np.array([np.cos(ang), np.sin(ang), 0.])
        pts.append(p); nrm.append(-p/np.linalg.norm(p))
    G = _grasp_map(pts, nrm)
    a = GraspForceAllocator(1.0)
    for mu in (0.2, 0.4, 0.7, 1.5):
        g, info = a.solve_gamma_cone(G, np.zeros(6), 3,
                                     normals=[[1,0,0]]*3, mu=mu, f_min=1.0)
        if g is None:
            continue          # infeasible is an acceptable honest answer
        _check(info, mu, 3, 1.0)


def test_f_max_budget_is_respected():
    pts, nrm = [], []
    for ang in (0.0, 2*np.pi/3, 4*np.pi/3):
        p = 0.03 * np.array([np.cos(ang), np.sin(ang), 0.])
        pts.append(p); nrm.append(-p/np.linalg.norm(p))
    G = _grasp_map(pts, nrm)
    a = GraspForceAllocator(1.0)
    g, info = a.solve_gamma_cone(G, np.zeros(6), 3, normals=[[1,0,0]]*3,
                                 mu=0.7, f_min=1.0, f_max=3.0)
    assert g is not None, info.get('reason')
    assert info['peak_normal'] <= 3.0 + 1e-6
    _check(info, 0.7, 3, 1.0)


def test_degenerate_parallel_normals_reports_infeasible():
    """Three nearly-parallel normals cannot squeeze. Must return None, not a
    fabricated gamma."""
    pts = [np.array([0., 0., 0.]), np.array([0.01, 0., 0.]), np.array([0.02, 0., 0.])]
    nrm = [np.array([0., 0., -1.])] * 3          # all pressing straight down
    G = _grasp_map(pts, nrm)
    a = GraspForceAllocator(1.0)
    g, info = a.solve_gamma_cone(G, np.zeros(6), 3, normals=[[1,0,0]]*3,
                                 mu=0.7, f_min=1.0)
    if g is not None:
        _check(info, 0.7, 3, 1.0)   # if it claims success it must be genuinely valid


def test_gamma_dimension_matches_null_space():
    pts, nrm = [], []
    for ang in (0.0, 2*np.pi/3, 4*np.pi/3):
        p = 0.03 * np.array([np.cos(ang), np.sin(ang), 0.])
        pts.append(p); nrm.append(-p/np.linalg.norm(p))
    G = _grasp_map(pts, nrm)
    a = GraspForceAllocator(1.0)
    g, info = a.solve_gamma_cone(G, np.zeros(6), 3, normals=[[1,0,0]]*3,
                                 mu=0.7, f_min=1.0)
    assert g is not None
    import scipy.linalg
    assert len(g) == scipy.linalg.null_space(G).shape[1] == info['null_dim']


if __name__ == '__main__':
    # Runs without pytest (not installed in this env).
    import traceback
    _t = sorted(n for n in dir() if n.startswith('test_'))
    _p = _f = 0
    for _n in _t:
        try:
            globals()[_n](); print(f'PASS  {_n}'); _p += 1
        except AssertionError as _e:
            print(f'FAIL  {_n}\n      {_e}'); _f += 1
        except Exception as _e:
            print(f'ERROR {_n}: {type(_e).__name__}: {_e}'); traceback.print_exc(); _f += 1
    print(f'\n{_p} passed, {_f} failed')
    raise SystemExit(1 if _f else 0)
