"""Placeholder for teleop-compatible task environments (goal #2 of the refactor plan,
see /home/aipexws5/.claude/plans/sleepy-petting-curry.md).

Not built out yet. Future home for e.g. kinova_leap_ycb_pick_place.py — a
YCB-object, teleop-compatible task environment analogous to
kinova_leap_pick_place.py but built on environments/grasp_bench/env.py and
benchmarks/ycb_grasp/scene.py instead of the hardcoded 6/7-primitive scene.
Depends on the ControlPhase state machine and TeleopInputSource abstraction landing
first (see the plan's Phase 3b / Phase 5) so this doesn't just reproduce the
monolith's problems on a different object set.
"""
