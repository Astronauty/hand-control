"""Grasp map computation: maps stacked contact forces to an object wrench, F_o = G @ f_c.

Planar and spatial implementations share the same `compute(contacts) -> G` interface so
callers (IK/force-allocation code, control loops) don't need to branch on dimensionality.
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy.linalg

from scripts.spatial_grasp_map import ContactModel, spatial_grasp_map


class GraspMapComputer(ABC):
    @abstractmethod
    def compute(self, contacts: list[dict]) -> np.ndarray:
        """Build the grasp map G such that w_o = G @ f_c.

        Args:
            contacts: list of {'p': contact position in object frame, 'R': rotation from
                contact frame to object frame (column 0 = inward/compressive normal)}.
        """

    @staticmethod
    def null_space(G: np.ndarray) -> np.ndarray:
        """Basis of G's null space (internal-force directions), shape (contact_dof, k)."""
        return scipy.linalg.null_space(G)


class PlanarGraspMapComputer(GraspMapComputer):
    """2D point contacts with friction (PCWF: 2 force DOF per contact).

    contacts: [{'p': (2,), 'R': (2,2)}, ...]. Supports any number of contacts (>=2).
    Generalizes internal_force_control.py's planar_hat_map/planar_grasp_map_PCWF, which
    were hardcoded to exactly 2 contacts.
    """

    def compute(self, contacts):
        blocks = []
        for c in contacts:
            x, y = c["p"]
            W = np.array([[1.0, 0.0], [0.0, 1.0], [-y, x]])
            blocks.append(W @ c["R"])
        return np.hstack(blocks)


class SpatialGraspMapComputer(GraspMapComputer):
    """3D point contacts. Thin wrapper over the already-generic
    scripts/spatial_grasp_map.spatial_grasp_map, exposing the same GraspMapComputer
    interface as the planar case.

    contacts: [{'p': (3,), 'R': (3,3)}, ...]. Supports any number of contacts.
    """

    def __init__(self, model: ContactModel = ContactModel.PCWF):
        self.model = model

    def compute(self, contacts):
        return spatial_grasp_map(contacts, model=self.model)
