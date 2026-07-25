"""Analyze a --grasp-trace npz (logs/<dir>/grasp_trace.npz) to attribute a lift slip.

Per-step during GRASP it has: t, squeeze_on, ramp, box_xpos, box_xquat, palm_z, norm_force
(per finger), tan_force (per finger), fc_cmd (per finger), slip (per finger), jog_v (3),
q_arm (7), gamma_live.

Prints a phase-segmented timeline and flags the slip-onset step, separating the failure modes:
  - box slips DOWN through fingers  (box_z drops while palm_z rises -> grip can't follow)
  - grip force collapses            (norm_force -> 0 -> contact lost)
  - friction saturates              (tan/(mu*norm) -> ~1 before slip -> true friction slip)
  - jog transient                   (slip jumps exactly when jog_v becomes nonzero)

Usage:
  python simulation/analyze_grasp_trace.py logs/<dir>/grasp_trace.npz [--mu 2.0]
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--mu', type=float, default=2.0)
    ap.add_argument('--every', type=int, default=0,
                    help='print every Nth row (0 = auto ~40 rows)')
    args = ap.parse_args()

    d = np.load(args.npz)
    t = d['t']
    n = len(t)
    box = d['box_xpos']            # (n,3)
    palm_z = d['palm_z']           # (n,)
    nf = d['norm_force']           # (n,2) [index,thumb]
    tf = d['tan_force']            # (n,2)
    fc = d['fc_cmd']               # (n,2)
    slip = d['slip'] * 1e3         # (n,2) mm
    jog = d['jog_v']               # (n,3)
    ramp = d['ramp']
    box_z = box[:, 2]

    print(f"[trace] {n} steps, t={t[0]:.2f}..{t[-1]:.2f}s  ({d['gamma_live'][0]:.1f} gamma)")
    print()

    # Key events.
    jog_mag = np.linalg.norm(jog, axis=1)
    jog_on = np.where(jog_mag > 1e-4)[0]
    i_jog = int(jog_on[0]) if len(jog_on) else None
    # Slip onset: first step where max-finger slip exceeds 2x its pre-jog baseline.
    base_slip = np.median(slip[:max(i_jog, 5)].max(1)) if i_jog else slip[:5].max()
    slip_jump = np.where(slip.max(1) > base_slip + 5.0)[0]
    i_slip = int(slip_jump[0]) if len(slip_jump) else None
    # Contact-lost: first step where either finger normal force drops to ~0 after being >5N.
    got_grip = np.where(nf.max(1) > 5.0)[0]
    i_grip = int(got_grip[0]) if len(got_grip) else None
    lost = None
    if i_grip is not None:
        for i in range(i_grip, n):
            if nf[i].min() < 0.5:
                lost = i; break

    def _tstr(i): return f"{t[i]:.2f}s (step {i})" if i is not None else "never"
    print(f"  jog starts   : {_tstr(i_jog)}   jog_v={jog[i_jog] if i_jog else '-'}")
    print(f"  grip forms   : {_tstr(i_grip)}")
    print(f"  slip onset   : {_tstr(i_slip)}   (baseline {base_slip:.1f}mm + 5mm)")
    print(f"  contact lost : {_tstr(lost)}")
    if i_jog is not None and i_slip is not None:
        dt_ms = (t[i_slip] - t[i_jog]) * 1e3
        print(f"  => slip onset is {dt_ms:+.0f} ms relative to jog start "
              f"({'LIFT-TRIGGERED' if abs(dt_ms) < 200 else 'independent of jog'})")
    print()

    # Friction-cone utilization at slip onset (was the grip actually at its friction limit?).
    if i_slip is not None:
        util = tf[i_slip] / (args.mu * np.maximum(nf[i_slip], 1e-6))
        print(f"  at slip onset: norm={nf[i_slip]} N  tan={tf[i_slip]} N  "
              f"fric_util={np.round(util*100,0)}%")
        print(f"    box_z={box_z[i_slip]*1e3:.1f}mm palm_z={palm_z[i_slip]*1e3:.1f}mm "
              f"rel={box_z[i_slip]*1e3 - palm_z[i_slip]*1e3:.1f}mm")
        if np.max(util) < 0.5:
            print("    -> friction NOT saturated at onset: NOT a classic friction slip. "
                  "The grip loses the box while friction margin remains "
                  "(tracking/stiffness/transient), not because it exceeded mu.")
        else:
            print("    -> friction saturated: the tangential load exceeded mu*N (raise gamma "
                  "or the contact geometry can't route the load).")
    print()

    # LP-vs-MuJoCo contact-model mismatch: recommended (grasp-map) normal/pos vs the ACTUAL
    # MuJoCo contact. A persistent angle/offset means the commanded 'internal' force pair is
    # NOT net-zero in physics -> a residual wrench that drifts the object even when held still.
    if 'norm_ang' in d:
        na = d['norm_ang']; po = d['pos_off']
        # while gripping (both fingers have force)
        grip = (nf.min(1) > 3.0)
        if grip.any():
            na_g = na[grip]; po_g = po[grip]
            na_valid = na_g[np.isfinite(na_g).all(1)] if na_g.ndim > 1 else na_g[np.isfinite(na_g)]
            print("  LP-vs-MuJoCo contact mismatch WHILE GRIPPING (rec normal/pos vs actual):")
            with np.errstate(invalid='ignore'):
                print(f"    normal angle (deg): mean={np.nanmean(na_g):.1f} "
                      f"median={np.nanmedian(na_g):.1f} max={np.nanmax(na_g):.1f}")
                print(f"    contact pos offset (mm): mean={np.nanmean(po_g):.1f} "
                      f"max={np.nanmax(po_g):.1f}")
            _bad = np.nanmedian(na_g)
            if _bad > 15:
                print(f"    -> LARGE normal divergence ({_bad:.0f}deg): the grasp map's normals "
                      "disagree with MuJoCo's real contact, so the internal force isn't net-zero "
                      "-> residual wrench drifts the box. This is the LP-vs-MuJoCo mismatch.")
            else:
                print(f"    -> normals agree ({_bad:.0f}deg): the drift is NOT from normal "
                      "mismatch; look at pos offset / soft-contact tracking.")
    print()

    # Timeline.
    step = args.every or max(1, n // 40)
    print(f"{'t':>6} {'ramp':>4} {'jogz':>6} {'boxz':>7} {'palmz':>7} {'rel':>6} "
          f"{'nf[i,t]':>12} {'slip[i,t]':>12}")
    print('-' * 74)
    for i in range(0, n, step):
        rel = (box_z[i] - palm_z[i]) * 1e3
        print(f"{t[i]:6.2f} {ramp[i]:4.2f} {jog[i,2]:6.3f} {box_z[i]*1e3:7.1f} "
              f"{palm_z[i]*1e3:7.1f} {rel:6.1f} "
              f"[{nf[i,0]:4.1f},{nf[i,1]:4.1f}] [{slip[i,0]:5.1f},{slip[i,1]:5.1f}]")


if __name__ == '__main__':
    main()
