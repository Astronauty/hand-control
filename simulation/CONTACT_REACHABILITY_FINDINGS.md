# Contact reachability & IK-solver study — findings

Follow-on to RECOMMENDER_CONVERGENCE_FINDINGS.md. That work concluded the recommender's
convergence problem was really a REACHABILITY problem: the fixed opposing-face-center
grasp sites are unreachable from many operator poses. This study characterizes what
actually governs reachability and which IK-solver settings best hit reachable contacts.

All tests use the 12 real operator seeds in `samples.jsonl` (q_seed + obj_qpos), the
`obj_red_box` geom resized/repositioned at runtime, and the collision-aware
`ConstrainedIKSolver`. "Reach<5mm" = fingertip site error < 5mm at the target with no
finger penetrating the box (min collision slack > -0.5mm).

Test scripts (simulation/):
- `study_ik_solvers.py`          — SQP vs IPOPT tuning study (2-stage)
- `test_cube_size_sweep.py`      — reach vs box size (--float-height isolates approach)
- `test_contact_height_sweep.py` — reach vs contact height on the face, across sizes
- `test_cik_allsamples.py`, `test_cik_reg_tol_sweep.py` — earlier tolerance/weight sweeps

---

## 1. Best IK-solver settings (study_ik_solvers.py)

Two-stage sweep (coarse 4-sample screen → confirm top-3/method on all 12). Cost weights
were allowed to differ per method.

Key knobs found:
- **SQP needs `tol_du = 0.01`** (tight). Loosening to 0.1 or 1.0 collapsed reach to 0-1/12.
- **SQP wants `tip_weight = 500`, NOT 1000.** At 1000 it over-drove into collision (0/4,
  ~48mm). At 500 it reached 3/4 in stage 1.
- **SQP softplus `alpha = 1500`** was best.
- IPOPT best at `tol = 1e-2`, `tip_weight = 1000`.

Head-to-head on all 12 (contacts at face centers, box on table):

| config | reach<5mm | med_err | max_t |
|---|---|---|---|
| **SQP a1500 tol_du0.01 tip500** | 6/12 | 6.6mm | **0.74s** |
| IPOPT tol1e-2 tip1000 | 6/12 | 5.0mm | 3.33s |

**SQP and IPOPT tie on convergence (6/12); SQP is ~4.5x faster (0.74s vs 3.33s).**
IPOPT edges median error (5.0 vs 6.6mm). For the sub-5s / high-rate / nonzero-posture
target, **SQP (a1500, tol_du0.01, tip500) is the pick** — sub-second, posture kept
(arm 1e-5 / hand 1e-4). This matches the recollection that the SQP variant performed well;
it just needed tip_weight lowered to 500 and alpha raised to 1500. NOTE: production
`--ik-solver` already defaults to sqp, but at tip_weight=100 — worth reconciling.

6/12 was the CEILING for both solvers at face-center contacts — not a tuning limit but a
reachability wall (next sections).

---

## 2. Reach vs cube size (test_cube_size_sweep.py)

Contacts pinned at the +/-x face centers. Two placements:

**Bottom-on-table (realistic):** non-monotonic, sharp peak at the 6 cm cube.

| full cube | 5.0 | 5.6 | 6.0 | 6.4 | 7.0 | 7.6 | 8.0 | 10.0 | 12.0 cm |
|---|---|---|---|---|---|---|---|---|---|
| reach<5mm | 4/12 | 5/12 | **6/12** | 5/12 | 3/12 | 3/12 | 4/12 | 0/12 | 1/12 |

**Center floated to common z=0.043 m (isolates approach height):** the small-cube failure
DISAPPEARS and the trend becomes monotonic — smaller = better.

| full cube | 4.0 | 6.0 | 8.0 | 10.0 | 12.0 cm |
|---|---|---|---|---|---|
| reach (floated) | **6/12** | 3/12 | 5/12 | 1/12 | 1/12 |
| reach (on table)| 0/12 | 6/12 | 4/12 | 0/12 | 1/12 |

The 4 cm cube goes 0/12 (on table) → 6/12 (floated): its on-table failure was ENTIRELY
the low approach height (fingers jam into the table), NOT face-precision. The precision
hypothesis (smaller face → harder to hit) is NOT supported — the ~1 cm LEAP fingertips hit
a small face fine once the approach clears the table.

Collision penetration (`n_penetrate`) was ~0 across all sizes — for face-center contacts,
finger-vs-box collision was never the binding constraint.

---

## 3. Reach vs contact HEIGHT on the face (test_contact_height_sweep.py) — the big result

Contacts on the +/-x faces of an on-table box, RAISED by dz = frac * half-extent toward
the top of the face (dz=0 is face center, dz=0.9h is near the top).

| full cube | dz=0.0h | dz=0.3h | dz=0.6h | dz=0.9h |
|---|---|---|---|---|
| 6 cm  | 6/12 (6.0mm) | 4/12 (11.6) | 8/12 (2.2) | 7/12 (2.4) |
| 8 cm  | 4/12 (25.0) | 4/12 (7.9) | 7/12 (2.8) | 8/12 (0.8) |
| 10 cm | 0/12 (31.3) | 4/12 (19.2) | 5/12 (5.6) | **10/12 (0.0)** |
| 12 cm | 1/12 (45.2) | 0/12 (27.3) | 3/12 (9.4) | **9/12 (0.1)** |

**Raising the contact toward the top of the face is a near-universal fix, strongest for
the LARGEST cubes** — exactly the ones hopeless at face center:
- 10 cm: 0/12 → 10/12;  12 cm: 1/12 → 9/12.
- At dz=0.9h, all sizes reach 7-10/12 with near-zero (0.0-0.1mm) median error.

This REVISES the earlier "finger-opposition span" conclusion. Span is not the dominant
limiter; **contact height relative to the approach is.** A higher contact (a) lets the
fingers approach from ABOVE the box instead of reaching down its side (clearing the
table), and (b) better matches the recorded operator poses, where the hand hovers above
and near the box. Even a 12 cm span becomes reachable when the contact is near the top and
the approach comes from above.

---

## Conclusions & implications

1. **Solver:** SQP with `alpha=1500, tol_du=0.01, tip_weight=500` gives the best
   speed/convergence tradeoff (sub-second, ties IPOPT's 6/12, posture kept nonzero).

2. **The fixed opposing-FACE-CENTER contact convention is the real reachability
   bottleneck** — it fails badly for large objects and low approaches. This is strong
   evidence FOR the recommender's contact-SEARCH design: it should be free to pick
   contacts that are reachable, not pinned to face centers.

3. **Concretely, contacts should be allowed to sit HIGHER on the object** (toward the top,
   away from the table). Permitting the grasp height to float recovered 9-10/12 on large
   cubes that were 0-1/12 at face center. If the recommender / grasp-site convention biases
   contacts upward (or simply doesn't pin them to the vertical center), reachability jumps.

4. **Precision and finger-vs-box collision are NOT the limiters** for these boxes; approach
   height and contact height are. Object size matters mostly through how it interacts with
   the table (a bigger box's face center sits higher, which is why on-table reach peaked at
   6 cm — a coincidence of the center height being in the reachable band).

### Open follow-ups
- Confirm the height effect transfers to the live recommender (contacts searched, not
  pinned) rather than this fixed-contact IK harness.
- The height sweep pinned contacts symmetrically; the recommender could also raise them
  asymmetrically or move off the vertical centerline — untested.
- Reconcile production SQP tip_weight (100) with the 500 found best here.
