"""SPADE Localizer Class."""

import logging

import numpy as np
import numpy.typing as npt
from pyscf import lib
from scipy import linalg

from ..system import LocalizedSystem
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
        active_occ_inds (np.array): 1D array of active occupied MO indices
        enviro_occ_inds (np.array): 1D array of environment occupied MO indices
        c_loc_occ (np.array): C matrix of localized occupied MOs

    Methods:
        run: Main function to run localization.
    """

    def __init__(
        self,
        global_scf: lib.StreamObject,
        n_active_atoms: int,
        max_shells: int = 4,
        n_mo_overwrite: int | None = None,
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

    def _localize_spin(self, c_matrix, occupancy, n_mo_overwrite=None):
        return super()._localize_spin(c_matrix, occupancy, n_mo_overwrite)

    def localize(
        self,
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
        c_matrix = self._global_scf.mo_coeff
        occupancy = self._global_scf.mo_occ
        n_mo_overwrite = self.n_mo_overwrite

        logger.debug("Localising spin with SPADE.")
        logger.debug(f"{c_matrix.shape=}")
        logger.debug(f"{occupancy=}")
        logger.debug(f"{n_mo_overwrite=}")

        # For spinless systems, occupancy is shape (n)
        # for spin-aware systems it is (2,n)
        # To make the logic consistent, we create a new axis
        # resulting in (1,n) for spinless
        if occupancy.ndim == 1:
            occupancy = occupancy[np.newaxis]

        if c_matrix.ndim == 2:
            c_matrix = c_matrix[np.newaxis]

        # The first thing we do is construct the orbital rotation
        # to transform to the SPADE basis.
        # we use the occupancy of Molecular Orbitals, ignoring spin.
        logger.debug("Constructing SPADE basis rotation.")

        # Find the number of orbitals which are at least partially occupied.
        n_occupied_orbitals = np.count_nonzero(np.sum(occupancy, axis=0))

        occupied_orbitals = c_matrix[..., :n_occupied_orbitals]
        logger.debug(f"{n_occupied_orbitals} occupied AOs.")

        n_act_aos = self._global_scf.mol.aoslice_by_atom()[self._n_active_atoms - 1][-1]
        logger.debug(f"{n_act_aos} active AOs.")

        ao_overlap = self._global_scf.get_ovlp()

        s_half = linalg.fractional_matrix_power(ao_overlap, 0.5)
        logger.debug(f"{s_half.shape=}")
        rotated_orbitals = s_half @ occupied_orbitals

        sigma = np.zeros((c_matrix.shape[0], min(n_occupied_orbitals, n_act_aos)))
        logger.debug(f"{sigma.shape=}")
        right_vectors = np.zeros(
            (c_matrix.shape[0], n_occupied_orbitals, n_occupied_orbitals)
        )
        _left_vectors = np.zeros((c_matrix.shape[0], n_act_aos, n_act_aos))
        for i, orbs in enumerate(rotated_orbitals):
            _left_vectors[i], sigma[i], right_vectors[i] = linalg.svd(
                orbs[:n_act_aos, :]
            )

        # The second part involves partitoning the system into active and environment
        # occupied parts
        # Here it is important that we let spade work on each spin independently.
        logger.debug("Partitoning Electrons.")

        # Prevents an error with argmax
        # It is possible to choose an active subsystem for which all
        # singular values are 1 (i.e. the whole system)
        # we want to avoid numerical error forcing random orbital assignment
        def parition_occupied_spin(sigma: np.ndarray):
            value_diffs = sigma[:-1] - sigma[1:]
            logger.debug("Singular value differences %s", value_diffs)
            if len(value_diffs) == 0:
                max_delta_sigma = 1
            elif np.allclose(value_diffs, np.zeros(value_diffs.shape)):
                max_delta_sigma = sigma.shape[-1]
            else:
                max_delta_sigma: int = np.argmax(value_diffs) + 1
            return max_delta_sigma

        # Closed shell.
        if len(set(np.sum(occupancy, axis=1))) == 1:
            logger.debug("Partitioning closed shell.")
            logger.debug(f"Singular Values: {sigma}")
            max_delta_sigma: npt.NDArray = np.apply_along_axis(
                parition_occupied_spin, axis=-1, arr=sigma
            )
        # Open shell
        else:
            logger.debug("Partitioning Open shell.")
            alpha_n_occupancy = np.count_nonzero(occupancy[0])
            alpha_occupied_orbitals = c_matrix[0, :, : int(alpha_n_occupancy)]
            logger.debug(f"{alpha_n_occupancy=}")
            logger.debug(f"{alpha_occupied_orbitals.shape=}")
            logger.debug(f"{ao_overlap.shape=}")
            logger.debug(f"{n_act_aos=}")
            alpha_rotated_orbitals = (s_half @ alpha_occupied_orbitals)[:n_act_aos, :]
            alpha_sigma = linalg.svdvals(alpha_rotated_orbitals)

            beta_n_occupancy = np.count_nonzero(occupancy[1])
            beta_occupied_orbitals = c_matrix[1, :, : int(beta_n_occupancy)]
            beta_rotated_orbitals = (s_half @ beta_occupied_orbitals)[:n_act_aos, :]
            beta_sigma = linalg.svdvals(beta_rotated_orbitals)
            logger.debug(f"Singular Values: {[alpha_sigma, beta_sigma]}")

            max_delta_sigma = np.array(
                [
                    parition_occupied_spin(alpha_sigma),
                    parition_occupied_spin(beta_sigma),
                ]
            )
        logger.debug(f"Max Difference in Singular Values: {max_delta_sigma}")

        match n_mo_overwrite:
            case int(n) if n <= sigma.shape[-1]:
                n_act_mos = n_mo_overwrite
                logger.debug(f"Enforcing use of {n_act_mos} MOs")
            case int(n) if n > sigma.shape[-1]:
                n_act_mos = sigma.shape[-1]
            case _:
                n_act_mos = np.max(max_delta_sigma)

        n_env_mos = n_occupied_orbitals - n_act_mos
        logger.debug(f"{n_act_mos} active MOs.")
        logger.debug(f"{n_env_mos} environment MOs.")

        # get active and enviro indices
        active_occ_inds = np.zeros(occupancy.shape, dtype=np.bool)
        for i, cutoff in enumerate(max_delta_sigma):
            active_occ_inds[i, :cutoff] = True

        enviro_occ_inds = np.zeros(occupancy.shape, dtype=np.bool)
        lower = max(max_delta_sigma) if n_mo_overwrite is None else n_mo_overwrite
        for i, spin_occ in enumerate(occupancy):
            upper = int(np.sum(spin_occ) - max_delta_sigma[i]) + lower
            enviro_occ_inds[..., lower:upper] = True

        # Defining active and environment orbitals and density
        c_active = occupied_orbitals @ right_vectors.swapaxes(-2, -1)[..., :n_act_mos]
        c_enviro = occupied_orbitals @ right_vectors.swapaxes(-2, -1)[..., n_act_mos:]
        c_loc_occ = occupied_orbitals @ right_vectors.swapaxes(-2, -1)

        # storing condition used to select env system
        self.enviro_selection_condition = sigma

        if occupancy.shape[0] == 1:
            logger.debug("Returning single spin Occupancy.")
            active_occ_inds = active_occ_inds[0]
            enviro_occ_inds = enviro_occ_inds[0]

        if c_loc_occ.shape[0] == 1:
            logger.debug("Returning single spin C Matrix.")
            c_active = c_active[0]
            c_enviro = c_enviro[0]
            c_loc_occ = c_loc_occ[0]

        return LocalizedSystem(
            active_occ_inds,
            enviro_occ_inds,
            c_active,
            c_enviro,
            c_loc_occ,
        )
