# Raised-contact wrench feasibility — findings

Follow-on to CONTACT_REACHABILITY_FINDINGS.md. That study found IK reachability improves
when the antipodal contacts sit HIGHER on the object face. This one asks whether those
raised contacts are also WRENCH-feasible (can generate the internal forces to hold the
object), and whether a moment-reference change can make them so.

Test scripts (simulation/):
- `test_recommender_contact_lift.py` — IK-only solve (wrench off), does it lift contacts?
- `test_decoupled_lp_on_ik.py`       — IK-only, then LP wrench-check on the converged contacts
Code touched:
- `scripts/3D_minimum_NCF.py`   — added `moment_ref` / `grav_force` mode to the LP
- `simulation/grasp_planner_3d.py` — `w_align` alignment cost, grasp-axis torque projection,
  `n_normal_relinearize=0` for boxes

---

## 1. IK-only lifts the contacts and converges (test_recommender_contact_lift.py)

With the wrench/gamma cost DISABLED and contacts as free decision variables, seeded at the
face center, the IK-only solve converges 11-12/12 and raises the contacts ~16-20 mm above
the object center (mean dz), sub-mm IK error. So the IK part of the recommender is healthy
and intrinsically prefers reachable (higher) contacts — no explicit height bias needed.

| full cube | ik_conv | mean dz | med IK err |
|---|---|---|---|
| 8 cm  | 12/12 | +18.5mm | 0.0mm |
| 10 cm | 12/12 | +20.1mm | 0.3mm |
| 12 cm | 11/12 | +16.4mm | 0.4mm |

## 2. The IK-converged contacts are wrench-INFEASIBLE (test_decoupled_lp_on_ik.py)

Feeding those converged contacts to the standalone LP: **0/12 wrench-feasible** (gamma =
None) at every size, despite IK converging. So the decoupled plan (IK finds contacts, LP
certifies) fails as-is: the IK-reachable contacts are NOT good for internal forces — the
concern was correct, and gamma=None is exactly how the LP flags it.

Diagnosis: the IK-only solve lifts the contacts ASYMMETRICALLY. Example (10 cm cube,
sample 0): p1_O=[0.05, -0.011, +0.001], p2_O=[-0.05, +0.011, -0.020] — a ~20 mm vertical
offset AND ~11 mm y-offset between the two contacts. Normals stay perfectly opposed
(n1=+x, n2=-x, dot=-1), but the contacts are at different heights/positions, so the grasp
is no longer a clean antipodal pair. The IK cost alone has no incentive to keep the two
contacts opposed, only to reach the (independently-seeded) targets.

## 3. Why raising breaks feasibility — the physics

For a symmetric side-pinch on the +/-x faces raised by dz (both contacts same height):
- Contact forces act at height dz. Resisting a lateral force F_x induces a moment dz*F_x
  the two-point geometry cannot cancel -> a pure lateral force becomes unresistable.
- Envelope map (scratchpad): max tolerable lateral accel is a CLIFF — 5+ m/s^2 at dz=0,
  then 0 for ANY dz>0. It is not a gradual trade; leaving the CoM plane at all kills the
  lateral-accel budget.
- VERTICAL load (gravity) is unaffected: a raised symmetric grasp resists gravity fine
  (gamma≈0.77), because vertical force through symmetric contacts induces no moment.

So: **a raised side-pinch can HOLD the object (gravity only) but cannot resist a lateral
disturbance APPLIED AT THE COM.** (It CAN resist a lateral disturbance applied at the grasp
datum — see Section 4; that is a different, and often more appropriate, task definition.)

## 4. The moment-reference / datum idea — implemented, verified, and it DOES render raised
##    contacts feasible under a lateral budget, given the right task definition

Hypothesis (user): specify the lateral-accel disturbance as applied at the grasp DATUM
(the point the accel limits are referenced to), not at the CoM. Then a raised grasp can
resist it.

Implemented as `min_gamma_for_accel_lp(..., moment_ref=<point>, grav_force=<m g>)` in
`scripts/3D_minimum_NCF.py`: re-datums the contact moments to `moment_ref` (set it to the
grasp midpoint), so the disturbance corner FORCES act at that datum with no induced
moment; gravity is added as an explicit constant wrench — its FORCE always, its MOMENT
`(CoM - ref) x (m g)` reference-dependent (honest for asymmetric/general grasps).

CORRECTNESS (part b): for a FIXED load, gamma must be invariant to the moment reference.
After fixing a bug (the first version added only gravity's moment, not its force), the
invariance holds exactly:

    HOLD case (gravity only), raised symmetric grasp (dz=+2cm):
      ref=CoM / graspMid / arbitrary -> gamma = 0.7664 (all EQUAL -> transform correct)
    Sanity: gravity-in-budget (old) == grav_force @ CoM (new) = 0.9695 (exact match).

TWO DISTINCT TASK DEFINITIONS (the key clarification):

    raised symmetric grasp dz=+2cm, a_lat=0.5 m/s^2:
      TASK A: lateral disturbance applied at the CoM   -> gamma = None  (infeasible)
      TASK B: lateral disturbance applied at the DATUM -> gamma = 0.930 (feasible)
    TASK B sweep dz=0..50mm: gamma = 0.93 FLAT (feasible at every height)

These are DIFFERENT physical loads, not a re-datum of one load:
  - Task A (force at CoM): models the free-body inertial reaction when the whole object is
    laterally accelerated. Genuinely unresistable by a raised two-point pinch.
  - Task B (force at the grasp datum): models the disturbance the GRASP must tolerate AT
    THE CONTACT INTERFACE. Physically appropriate for hold-and-transport, where forces
    come through the fingers. `moment_ref`=grasp-midpoint implements exactly this.

So the moment reference does NOT change the feasibility of a *fixed* load (gamma is
invariant), BUT choosing to specify the accel budget AT THE GRASP DATUM is a legitimate,
useful task definition under which raised contacts ARE feasible under a nonzero lateral
budget — gamma flat at 0.93 across all heights. This is the intended use of `moment_ref`.

The decoupled test on the REAL IK contacts still shows 0/12 in both modes — but for a
SEPARATE reason: the IK-only solve lifts the contacts ASYMMETRICALLY (Section 2), so they
are not a clean antipodal pair and fail force closure regardless of task definition. Task B
feasibility (gamma=0.93 flat) was demonstrated on SYMMETRIC raised grasps; the real IK
contacts need the symmetric-opposition fix below before Task B can rescue them.

## Conclusions

1. **The accel-budget DATUM is a task-definition choice, and it gates raised-contact
   feasibility.** For a *fixed* load, gamma is invariant to the moment reference (verified).
   But specifying the lateral disturbance AT THE GRASP DATUM (Task B, `moment_ref`=grasp
   midpoint) vs at the CoM (Task A) is a different, legitimate load — and under Task B a
   symmetric raised grasp is feasible at every height (gamma=0.93 flat). Task B models the
   disturbance the grasp tolerates at the contact interface (appropriate for hold-and-
   transport); Task A models a free-body lateral acceleration of the whole object. Pick the
   one that matches the task.

2. **`moment_ref`/`grav_force` is the mechanism** that implements Task B correctly (and
   handles gravity honestly for asymmetric grasps). It IS the lever for raised-contact
   feasibility under a lateral budget — provided the grasp is a clean antipodal pair.

3. **The remaining blocker for the real IK contacts is asymmetric lift**, not the wrench
   task. The IK must keep the two contacts at the SAME height/position (a clean antipodal
   pair). The current `w_align` cost constrains grasp-axis DIRECTION but not the contact
   OFFSET, so it does not fix the asymmetric lift; a symmetric-contact constraint is needed.
   With that, Task B (`moment_ref`) should render the raised contacts feasible.

## 5. LOOP CLOSED: raised IK contacts are wrench-feasible (12/12) — full stack

The decoupled test on the REAL IK contacts now passes end-to-end. Four pieces together:

  1. IK finds reachable (naturally-lifted) contacts        — 11-12/12 IK convergence
  2. `w_align`  keeps them z-symmetric (equal height)      — antipodal direction
  3. `moment_ref` (Task B) applies lateral disturbance at the grasp datum
  4. `project_grasp_axis_moment` removes the unresistable grasp-axis component of the
     GRAVITY moment (from residual off-CoM common-y drift that w_align is blind to)

    datum + gravity-projection + w_align=2, lateral a=0.5, ANGULAR a=0:
      8 cm : ik 12/12, LP-feasible 12/12, gamma ~0.99
     10 cm : ik 12/12, LP-feasible 12/12, gamma ~1.02
     12 cm : ik 11/12, LP-feasible 12/12, gamma ~1.03
    (was 0/12 LP-feasible before). gamma ~1.0 = efficient grasp.

CRITICAL last condition: the ANGULAR accel budget must also match a hold task (~0). A
nonzero angular budget adds +/-Ty,+/-Tz corners that the raised/off-center grasp cannot
resist — the same force-torque coupling as the lateral case, from the angular side. So the
full statement is: **raised, reachable contacts are wrench-feasible FOR A HOLD/TRANSPORT
TASK** — lateral disturbance referenced at the grasp interface, no commanded angular
acceleration. For genuine object angular acceleration, a two-point pinch is the wrong tool.

Two findings that corrected earlier claims:
- `w_align` does MORE than expected: for face-pinned contacts it drives them to EQUAL
  height (fixes the z-asymmetry), not just grasp-axis direction. But it is mathematically
  BLIND to common-mode y translation (both contacts sliding together), which is why an
  off-CoM-y drift remained — handled by piece 4 (gravity grasp-axis projection) rather than
  by forcing the grasp to center.
- The moment REFERENCE alone does not gate a fixed load (gamma invariant); it is the choice
  to APPLY the disturbance at the datum (a task definition) that makes raised contacts work.

### Also landed this session (kept — physically correct)
- `moment_ref` / `grav_force` / `project_grasp_axis_moment` modes in
  `scripts/3D_minimum_NCF.py` (documented above; verified reference-invariant for fixed
  loads, and the projection is a no-op when the grasp is centered).
- Grasp-axis torque projection in the recommender NLP (`grasp_planner_3d.py`) mirroring
  `solve_gamma_live`; `w_align` alignment cost; `n_normal_relinearize = 0` for boxes.

### Open follow-ups
- Wire the datum + gravity-projection + w_align stack into the LIVE recommender (contacts
  searched, wrench cone on) rather than the decoupled harness, with hold-appropriate
  lateral/angular budgets, and confirm end-to-end.
- w_align is blind to common-mode contact translation; if centering is ever wanted for
  other reasons, a separate midpoint-offset term (not alignment) is required.
