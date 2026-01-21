import argparse
import time

import mujoco
import mujoco.viewer


def _camera_id(model: mujoco.MjModel, name: str) -> int:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cam_id < 0:
        raise ValueError(f"Camera '{name}' not found in model. Available cameras: {model.ncam}")
    return cam_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--xml",
        default="models/planar_two_finger_manipulator.xml",
        help="Path to MJCF/XML",
    )
    ap.add_argument("--cam", default="cam0", help="Named camera to use (fixedcam)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    cam_id = _camera_id(model, args.cam)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # --- Set viewer to fixed named camera ---
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

        # --- Enable frame visualization ---
        # Any value other than mjFRAME_NONE draws frames.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY  # or mjFRAME_GEOM / mjFRAME_SITE / ...

        # (Optional) also show joint axes:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        viewer.sync()

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()