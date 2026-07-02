## Installation instructions
Install and source the UV environment
```
conda create -f environment.yml
conda activate hand-control
```

```bash
source .venv/bin/activate
```

## Launch Environments
### 2D Environment
python internal_force_control.py

### 3D Environment
python kinova_leap_pick_place.py

## User Interfaces
### Motion Capture (Mediapipe)
run ui/mediapipe_joint_angles.py


## Pull MuJoCo Menagerie Models
git submodule update --init --recursive