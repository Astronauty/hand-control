"""GLFW keyboard-callback factory for the MuJoCo passive viewer used by the
pick-and-place control loop. Extracted verbatim from kinova_leap_pick_place.py.
"""


def make_key_callback(key_queue):
    """Return a GLFW key callback for the MuJoCo passive viewer.
    Handles only Enter / Q / Esc; target selection is handled via pynput
    (Ctrl+digit) so modifier state is available without fighting GLFW."""
    # Keycodes chosen to NOT collide with the viewer's built-in single-key toggles
    # (mjVISSTRING/mjRNDSTRING shortcuts claim nearly every letter: R=Reflection,
    # K=Skybox, B=Perturb Force, G=Fog, ...). 'N' is the only unbound letter; the
    # viewer only uses digits 0-5 (geom groups), so 6-8 are free. Q still overlaps
    # the Camera-frustum toggle, but we quit on it so the toggle never matters.
    _MAP = {
        257: 'enter',   # GLFW_KEY_ENTER (main keyboard)
        335: 'enter',   # GLFW_KEY_KP_ENTER (numpad)
        81:  'quit',    # Q
        256: 'quit',    # Escape
        78:  'release',  # N — (GRASP) open fingers and return to REACH
        76:  'lock_in',  # L — (contact_aware_teleop) lock in the recommended grasp
                        #     contacts and approach them via RRT
        80:  'rec_vis',  # P — (contact_aware_teleop) preview the recommender's OWN q
                        #     solution (unconstrained NLP, no collision IK)
        79:  'rec_ik_recq_vis',  # O — preview the collision-aware IK to the recommended
                                #     contacts, warm-started from the recommender's q
        73:  'rec_ik_dls_vis',   # I — same collision-aware IK but warm-started from a
                                #     fresh DLS solve, to A/B the two warm-starts
        82:  'record_sample',   # R — (contact_aware_teleop, --record-samples) append the
                                #     current (pose, object, recommendation) to the tuning
                                #     dataset for offline IK-weight sweeping
        54:  'ik_vis',  # 6 — cycle IK config visualization
        55:  'bspheres', # 7 — toggle IK collision bounding-sphere overlay
        56:  'teleop_start', # 8 — (dexpilot) start/re-zero tracking at current pose
        57:  'calib_orient', # 9 — (dexpilot) hold hand to match robot wrist, capture
                             #     the constant orientation correction
        # Multi-pose orientation calibration (dexpilot):
        77:  'calib_next',   # M — pose the robot to the next calibration orientation
        67:  'calib_capture', # C — capture (hand matched to current wrist)
        86:  'calib_solve',  # V — solve the full rotation correction from captures
        259: 'reset',   # Backspace — deliberately shadows the viewer's built-in Reset:
                        # the viewer already mj_resetData'd the shared data from its own
                        # thread; this event lets the control loop re-home its state
                        # machine instead of PD-exploding against the qpos0 arm pose.
        46:  'next_keyframe',  # . (>) — cycle to the NEXT scene <keyframe> (pose_00, ...)
        44:  'prev_keyframe',  # , (<) — cycle to the PREVIOUS scene <keyframe>
                               #     ...and re-home the state machine onto it. NOTE: do NOT use
                               #     [ / ] here — MuJoCo's viewer hard-binds those to camera
                               #     cycling (they'd switch to the 'wrist' fixed camera). The
                               #     passive viewer shares data with this loop, so a GUI keyframe
                               #     Load is clobbered every frame; loading FROM the loop (here)
                               #     is the reliable way to test the recommender/jog from each
                               #     registered box+arm config, and keeps the free camera put.
    }
    def _cb(keycode):
        event = _MAP.get(keycode)
        if event:
            key_queue.put(event)
    return _cb
