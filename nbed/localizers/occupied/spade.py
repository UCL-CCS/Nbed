"""SPADE Localizer Class."""

import logging

import numpy as np
from pyscf import lib
from scipy import linalg

from .base import LocalizedSystem, OccupiedLocalizer

logger = logging.getLogger(__name__)


class SPADELocalizer(OccupiedLocalizer):
    """Object used to localise molecular orbitals (MOs) using SPADE Localization.

    Running localization returns active and environment systems.

    Args:
        global_scf (scf.StreamObject): PySCF method object.
        n_active_atoms (int): Number of active atoms

    Attributes:
        c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
        c_enviro (np.array): C matrix of localized occupied ennironment MOs
        c_loc_occ_and_virt (np.array): Full localized C_matrix (occpuied and virtual)
        dm_active (np.array): active system density matrix
        dm_enviro (np.array): environment system density matrix
        active_mo_inds (np.array): 1D array of active occupied MO indices
        enviro_mo_inds (np.array): 1D array of environment occupied MO indices
        c_loc_occ (np.array): C matrix of localized occupied MOs

    Methods:
        run: Main function to run localization.
    """

    def __init__(
        self,
        global_scf: lib.StreamObject,
        n_active_atoms: int,
        max_shells: int = 4,
        n_mo_overwrite: tuple[int | None, int | None] | None = None,
    ):
        """Initialize SPADE Localizer object."""
        self.max_shells = max_shells
        self.shells = None
        self.singular_values = None
        self.enviro_selection_condition = None

        super().__init__(
            global_scf,
            n_active_atoms,
            n_mo_overwrite,
        )

    def _localize_spin(
        self,
        c_matrix: np.ndarray,
        occupancy: np.ndarray,
        n_mo_overwrite: int | None = None,
    ) -> LocalizedSystem:
        """Localize orbitals of one spin using SPADE.

        Args:
            c_matrix (np.ndarray): Unlocalized C matrix of occupied orbitals.
            occupancy (np.ndarray): Occupancy of orbitals.
            n_mo_overwrite (int | None): Overwrite the number of active molecular orbitals.

        Returns:
            np.ndarray: Indices of active molecular orbitals
            np.ndarray: Indices of environment molecular orbitals
            np.ndarray: Localized C matrix of active orbitals.
            np.ndarray: Localized C matrix of environment orbitals.
            np.ndarray: Localized C matrix of all occpied orbitals.
        """
        logger.debug("Localising spin with SPADE.")
        logger.debug(f"{c_matrix.shape=}")
        logger.debug(f"{occupancy=}")
        logger.debug(f"{n_mo_overwrite=}")

        # We want the same partition for each spin.
        # It wouldn't make sense to have different spin states be localized differently.

        n_occupied_orbitals = np.count_nonzero(occupancy)
        occupied_orbitals = c_matrix[:, :n_occupied_orbitals]
        logger.debug(f"{n_occupied_orbitals} occupied AOs.")

        n_act_aos = self._global_scf.mol.aoslice_by_atom()[self._n_active_atoms - 1][-1]
        logger.debug(f"{n_act_aos} active AOs.")

        ao_overlap = self._global_scf.get_ovlp()

        # Orbital rotation and partition into subsystems A and B
        # rotation_matrix, sigma = embed.orbital_rotation(occupied_orbitals,
        #    n_act_aos, ao_overlap)

        rotated_orbitals = (
            linalg.fractional_matrix_power(ao_overlap, 0.5) @ occupied_orbitals
        )
        _, sigma, right_vectors = linalg.svd(rotated_orbitals[:n_act_aos, :])

        logger.debug(f"Singular Values: {sigma}")

        # n_act_mos, n_env_mos = embed.orbital_partition(sigma)
        # Prevents an error with argmax
        if len(sigma) == 1:
            n_act_mos = 1
        elif n_mo_overwrite is not None and len(sigma) >= n_mo_overwrite:
            logger.debug(f"Enforcing use of {n_mo_overwrite} MOs")
            n_act_mos: int = n_mo_overwrite
        else:
            value_diffs = sigma[:-1] - sigma[1:]
            logger.debug("Singular value differences %s", value_diffs)
            # It is possible to choose an active subsystem for which all
            # singular values are 1 (i.e. the whole system)
            # we want to avoid numerical error forcing random orbital assignment
            if np.allclose(value_diffs, [0] * len(value_diffs)):
                n_act_mos = len(sigma)
            else:
                n_act_mos: int = np.argmax(value_diffs) + 1

        n_env_mos = n_occupied_orbitals - n_act_mos
        logger.debug(f"{n_act_mos} active MOs.")
        logger.debug(f"{n_env_mos} environment MOs.")

        # get active and enviro indices
        active_mo_inds = np.arange(n_act_mos)
        enviro_mo_inds = np.arange(n_act_mos, n_act_mos + n_env_mos)

        # Defining active and environment orbitals and density
        c_active = occupied_orbitals @ right_vectors.T[:, :n_act_mos]
        c_enviro = occupied_orbitals @ right_vectors.T[:, n_act_mos:]
        c_loc_occ = occupied_orbitals @ right_vectors.T

        # storing condition used to select env system
        if self.enviro_selection_condition is None:
            self.enviro_selection_condition = (sigma, np.zeros(len(sigma)))
        else:
            self.enviro_selection_condition = (
                self.enviro_selection_condition[0],
                sigma,
            )

        return LocalizedSystem(
            active_mo_inds, enviro_mo_inds, c_active, c_enviro, c_loc_occ
        )
