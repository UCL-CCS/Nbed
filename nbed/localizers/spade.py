"""SPADE Localizer Class."""

import logging
from typing import Optional, Tuple

import numpy as np
from pyscf import gto
from scipy import linalg

from .base import Localizer

logger = logging.getLogger(__name__)


class SPADELocalizer(Localizer):
    """Localizer Class to carry out SPADE"""

    def __init__(
        self,
        pyscf_scf: gto.Mole,
        n_active_atoms: int,
        localization_method: str,
        occ_cutoff: Optional[float] = 0.95,
        virt_cutoff: Optional[float] = 0.95,
        run_virtual_localization: Optional[bool] = False,
    ):
        super().__init__(
            pyscf_scf,
            n_active_atoms,
            localization_method,
            occ_cutoff=occ_cutoff,
            virt_cutoff=virt_cutoff,
            run_virtual_localization=run_virtual_localization,
        )

    def _localize(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Localise orbitals using SPADE.

        Returns:
            active_MO_inds (np.array): 1D array of active occupied MO indices
            enviro_MO_inds (np.array): 1D array of environment occupied MO indices
            c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
            c_enviro (np.array): C matrix of localized occupied ennironment MOs
            c_loc_occ (np.array): full C matrix of localized occupied MOs
        """
        logger.info("Localising with SPADE.")
        n_occupied_orbitals = np.count_nonzero(self._pyscf_scf.mo_occ == 2)
        occupied_orbitals = self._pyscf_scf.mo_coeff[:, :n_occupied_orbitals]

        n_act_aos = self._pyscf_scf.mol.aoslice_by_atom()[self._n_active_atoms - 1][-1]
        logger.debug(f"{n_act_aos} active AOs.")

        ao_overlap = self._pyscf_scf.get_ovlp()

        # Orbital rotation and partition into subsystems A and B
        # rotation_matrix, sigma = embed.orbital_rotation(occupied_orbitals,
        #    n_act_aos, ao_overlap)

        rotated_orbitals = (
            linalg.fractional_matrix_power(ao_overlap, 0.5) @ occupied_orbitals
        )
        _, sigma, right_vectors = linalg.svd(rotated_orbitals[:n_act_aos, :])

        logger.debug(f"Singular Values: {sigma}")

        # n_act_mos, n_env_mos = embed.orbital_partition(sigma)
        value_diffs = sigma[:-1] - sigma[1:]
        n_act_mos = np.argmax(value_diffs) + 1
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

        return active_MO_inds, enviro_MO_inds, c_active, c_enviro, c_loc_occ
