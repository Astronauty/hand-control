from enum import Enum
import numpy as np


class ContactModel(Enum):
    PC   = "frictionless"            # 1 DOF: normal force only
    PCWF = "point_contact_friction"  # 3 DOF: normal + 2 tangential forces
    SFC  = "soft_finger"             # 4 DOF: PCWF + torsional moment about normal


def hat(v):
    """3×3 skew-symmetric matrix such that hat(v) @ u = v × u."""
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])


def spatial_grasp_map(contacts, model=ContactModel.PCWF):
    """Build the spatial grasp map G mapping stacked contact forces/torques to the
    6D object wrench: w_o = G @ f_c

    Args:
        contacts: list of dicts, each with:
            'p': np.array(3)    contact position in object frame
            'R': np.array(3,3)  rotation from contact frame to object frame;
                                columns are [normal_dir, tangent1, tangent2]
        model: ContactModel enum selecting the contact/friction model

    Returns:
        G: np.array(6, dof_per_contact * n_contacts)
           w_o = G @ f_c maps stacked contact forces [f_c1; ...; f_cn]
           to object wrench [fx, fy, fz, mx, my, mz] in the object frame.

    DOF per contact by model:
        PC   — 1: f_c_i = [f_n]
        PCWF — 3: f_c_i = [f_n, f_t1, f_t2]
        SFC  — 4: f_c_i = [f_n, f_t1, f_t2, m_n]

    Contact frame convention:
        column 0 of R_i = inward surface normal (compressive direction)
        columns 1, 2    = tangential directions
    """
    blocks = []
    for c in contacts:
        p = c['p']
        R = c['R']
        W = np.vstack([np.eye(3), hat(p)])        # 6×3: maps force → wrench

        if model == ContactModel.PC:
            G_i = W @ R[:, 0:1]                   # 6×1
        elif model == ContactModel.PCWF:
            G_i = W @ R                           # 6×3
        elif model == ContactModel.SFC:
            n = R[:, 0:1]                         # contact normal (3×1)
            torque_col = np.vstack([np.zeros((3, 1)), n])  # 6×1: pure moment along normal
            G_i = np.hstack([W @ R, torque_col])  # 6×4
        else:
            raise ValueError(f"Unknown contact model: {model}")

        blocks.append(G_i)

    return np.hstack(blocks)


if __name__ == "__main__":
    import scipy.linalg

    # Symmetric 2-finger box grasp: contacts on ±x faces of a 10cm cube.
    # Contact normals point inward (toward object center).
    contacts = [
        {   # right face: normal points in -x
            'p': np.array([0.05, 0.0, 0.0]),
            'R': np.array([[-1, 0, 0],
                           [ 0, 1, 0],
                           [ 0, 0, 1]], dtype=float).T,
        },
        {   # left face: normal points in +x
            'p': np.array([-0.05, 0.0, 0.0]),
            'R': np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]], dtype=float).T,
        },
    ]

    for model in ContactModel:
        G = spatial_grasp_map(contacts, model=model)
        null = scipy.linalg.null_space(G)
        print(f"{model.name:4s}  G: {G.shape}  null-space dim: {null.shape[1]}")
        print(G)
        print()
