"""SPADE Localizer Class."""

import logging

import numpy as np
from pyscf import lib
from scipy import linalg

from .base import OccupiedLocalizer

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
        active_MO_inds (np.array): 1D array of active occupied MO indices
        enviro_MO_inds (np.array): 1D array of environment occupied MO indices
        _c_loc_occ (np.array): C matrix of localized occupied MOs

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
        self.n_mo_overwrite = (None, None) if n_mo_overwrite is None else n_mo_overwrite
        self.shells = None
        self.singular_values = None
        self.enviro_selection_condition = None

        super().__init__(
            global_scf,
            n_active_atoms,
        )

    def _localize(
        self,
    ) -> tuple[tuple, tuple | None]:
        """Localise orbitals using SPADE.

        Returns:
            active_MO_inds (np.array): 1D array of active occupied MO indices
            enviro_MO_inds (np.array): 1D array of environment occupied MO indices
            c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
            c_enviro (np.array): C matrix of localized occupied ennironment MOs
            c_loc_occ (np.array): full C matrix of localized occupied MOs
        """
        if self.spinless:
            logger.debug("Running SPADE for only one spin.")
            alpha = self._localize_spin(
                self._global_scf.mo_coeff,
                self._global_scf.mo_occ,
                self.n_mo_overwrite[0],
            )
            beta = None
        else:
            alpha = self._localize_spin(
                self._global_scf.mo_coeff[0],
                self._global_scf.mo_occ[0],
                self.n_mo_overwrite[0],
            )
            beta = self._localize_spin(
                self._global_scf.mo_coeff[1],
                self._global_scf.mo_occ[1],
                self.n_mo_overwrite[1],
            )

        return (alpha, beta)

    def _localize_spin(
        self,
        c_matrix: np.ndarray,
        occupancy: np.ndarray,
        n_mo_overwrite: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Localize orbitals of one spin using SPADE.

        Args:
            c_matrix (np.ndarray): Unlocalized C matrix of occupied orbitals.
            occupancy (np.ndarray): Occupancy of orbitals.
            n_mo_overwrite (int): Number of molecular orbitals to use in active region. Overwrite SVD based value.

        Returns:
            np.ndarray: Localized C matrix of occupied orbitals.
        """
        logger.debug("Localising with SPADE.")

        # We want the same partition for each spin.
        # It wouldn't make sense to have different spin states be localized differently.

        n_occupied_orbitals = np.count_nonzero(occupancy)
        occupied_orbitals = c_matrix[:, :n_occupied_orbitals]
        logger.debug(f"{occupancy=}")
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
            n_act_mos: int = np.argmax(value_diffs) + 1

        n_env_mos = n_occupied_orbitals - n_act_mos
        logger.debug(f"{n_act_mos} active MOs.")
        logger.debug(f"{n_env_mos} environment MOs.")

        # get active and enviro indices
        active_MO_inds = np.arange(n_act_mos)
        enviro_MO_inds = np.arange(n_act_mos, n_act_mos + n_env_mos)

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

        return (active_MO_inds, enviro_MO_inds, c_active, c_enviro, c_loc_occ)
