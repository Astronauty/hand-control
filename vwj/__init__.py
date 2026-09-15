"""Vector-Wrist-Joint (VWJ) whole-arm-hand retargeting — a 4th teleop baseline.

Clean-room reimplementation of the retargeting method from:

  Xin, Yu, Jiang, Zhang, Li. "Analyzing Key Objectives in Human-to-Robot
  Retargeting for Dexterous Manipulation." arXiv:2506.09384 (2025/2026).
  Project: https://mingrui-yu.github.io/retargeting

This is an INDEPENDENT reimplementation written from the paper and the method's
public description — it does NOT copy the authors' source code (their repository
carries no license, so verbatim vendoring is not permitted). The objective,
weights, and continuous-pinch formulation reproduce the published method; the
forward kinematics use MuJoCo (this repo's existing Gen3+LEAP model) instead of
the authors' pinocchio backend, so the arm/hand kinematics stay consistent with
the simulator the other baselines run in.

Unlike DexPilot / AnyTeleop (finger-only retargeting on a shared wrist-following
arm IK), VWJ solves the WHOLE robot (7 arm + 16 hand joints) in one SLSQP solve
per frame against a single objective:

    L = links_vec + wrist_rot + joint_pos + joint_vel        (Huber losses)

with the fingertip-pinch term continuously (sigmoid-)weighted and rescaled — the
paper's headline change over DexPilot's discrete pinch switch.
"""
