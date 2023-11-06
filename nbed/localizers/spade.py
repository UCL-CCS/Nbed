"""SPADE Localizer Class."""

import logging
from typing import Optional, Tuple, Union

import numpy as np
from pyscf import dft, gto, scf
from pyscf.lib import StreamObject
from scipy import linalg

from .base import Localizer

logger = logging.getLogger(__name__)


class SPADELocalizer(Localizer):
    """Object used to localise molecular orbitals (MOs) using SPADE Localization.

    Running localization returns active and environment systems.

    Args:
        pyscf_rks (gto.Mole): PySCF molecule object
        n_active_atoms (int): Number of active atoms
        localization_method (str): String of orbital localization method (spade, pipekmezey, boys, ibo)
        occ_cutoff (float): Threshold for selecting occupied active region (only requried if
                                spade localization is NOT used)
        virt_cutoff (float): Threshold for selecting unoccupied (virtual) active region (required for
                                spade approach too!)
        run_virtual_localization (bool): optional flag on whether to perform localization of virtual orbitals.
                                         Note if False appends canonical virtual orbs to C_loc_occ_and_virt matrix

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
        pyscf_scf: gto.Mole,
        n_active_atoms: int,
        max_shells: int = 4,
    ):
        """Initialize SPADE Localizer object."""
        super().__init__(
            pyscf_scf,
            n_active_atoms,
        )
        self.max_shells = max_shells

    def _localize_spin(
        self, c_matrix: np.ndarray, occupancy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Localize orbitals of one spin using SPADE.

        Args:
            c_matrix (np.ndarray): Unlocalized C matrix of occupied orbitals.
            occupancy (np.ndarray): Occupancy of orbitals.

        Returns:
            np.ndarray: Localized C matrix of occupied orbitals.
        """
        logger.debug("Localising with SPADE.")

        # We want the same partition for each spin.
        # It wouldn't make sense to have different spin states be localized differently.

        n_occupied_orbitals = np.count_nonzero(occupancy)
        occupied_orbitals = c_matrix[:, :n_occupied_orbitals]

        n_act_aos = self._global_ks.mol.aoslice_by_atom()[self._n_active_atoms - 1][-1]
        logger.debug(f"{n_act_aos} active AOs.")

        ao_overlap = self._global_ks.get_ovlp()

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
        else:
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

        # storing condition used to select env system
        self.enviro_selection_condition = sigma

        return (active_MO_inds, enviro_MO_inds, c_active, c_enviro, c_loc_occ)

    def localize_virtual(self, local_scf: StreamObject) -> StreamObject:
        """Localise virtual (unoccupied) obitals using concentric localization.

        [1] D. Claudino and N. J. Mayhall, "Simple and Efficient Truncation of Virtual
        Spaces in Embedded Wave Functions via Concentric Localization", Journal of Chemical
        Theory and Computation, vol. 15, no. 11, pp. 6085-6096, Nov. 2019,
        doi: 10.1021/ACS.JCTC.9B00682.

        Args:
            local_scf (StreamObject): SCF object with occupied orbitals localized.

        Returns:
            StreamObject: Fully Localized SCF object.
        """
        logger.debug("Localising virtual orbital spin with concentric localization.")

        projected_mol = gto.mole.Mole()
        projected_mol.atom = local_scf.mol.atom
        projected_mol.basis = local_scf.mol.basis  # can be anything
        projected_mf = scf.RKS(projected_mol)
        n_act_proj_aos = projected_mol.aoslice_by_atom()[self._n_active_atoms - 1][-1]
        logger.debug(f"{n_act_proj_aos=}")

        projected_overlap = projected_mf.get_ovlp(local_scf.mol)[
            :n_act_proj_aos, :n_act_proj_aos
        ]
        overlap_two_basis = gto.intor_cross(
            "int1e_ovlp_sph", local_scf.mol, projected_mol
        )[:n_act_proj_aos, :]

        occupancy = local_scf.mo_occ
        virtual_mos = np.where(occupancy == 0)[0]
        logger.debug(f"{virtual_mos=}")

        effective_virt = local_scf.mo_coeff[:, virtual_mos]
        logger.debug(f"N effective viruals: {effective_virt.shape}")

        left = np.linalg.inv(projected_overlap) @ overlap_two_basis @ effective_virt

        _, sigma, right_vectors = np.linalg.svd(
            left.T @ overlap_two_basis @ effective_virt
        )
        logger.debug(f"Singular values: {sigma}")

        # We'll iteratively build up the total C matrix
        active_mos = np.where(occupancy > 0)[0]
        logger.debug(f"Using active MOs for CL: {active_mos}")
        c_total = local_scf.mo_coeff[:, active_mos]
        logger.debug(f"{c_total.shape=}")

        shell_size = np.sum(sigma[:n_act_proj_aos] >= 1e-15)
        logger.debug(f"{shell_size=}")

        v_span = right_vectors.T[:, :shell_size]  # 0 but instability
        v_ker = right_vectors.T[:, shell_size:]
        logger.debug(f"{v_span.shape=}")

        c_ispan = effective_virt @ v_span
        # We'll transform this in the loop
        c_iker = effective_virt

        c_total = np.hstack((c_total, c_ispan))

        # keep track of the number of orbitals in each shell
        shells = []
        shells.append(c_total.shape[1])

        fock_operator = local_scf.get_fock()
        # why use the overlap for the first shell and then the fock for the rest?

        for ishell in range(1, self.max_shells + 1):

            logger.debug(f"{v_ker.shape[1]=}")
            if v_ker.shape[1] > 0:
                c_iker = c_iker @ v_ker
            else:
                # This means that all virtual orbitals have been included.
                logger.debug("No kernel, ending CL.")
                break

            logger.debug("Beginning Concentric Localization Iteration")
            logger.debug(f"{c_ispan.shape=}, {fock_operator.shape=}, {c_iker.shape=}")
            _, sigma, right_vectors = linalg.svd(c_total.T @ fock_operator @ c_iker)
            logger.debug(f"Singular values: {sigma}")
            logger.debug(f"{right_vectors.shape=}")

            shell_size = np.sum(sigma[:n_act_proj_aos] >= 1e-15)
            logger.debug(f"{shell_size=}")
            if shell_size == 0:
                logger.debug("Empty shell, ending CL.")
                break

            v_span = right_vectors.T[:, :shell_size]
            v_ker = right_vectors.T[:, shell_size:]

            logger.debug(f"{v_span.shape=}")

            c_total = np.hstack((c_total, c_iker @ v_span))
            logger.debug("Adding shell to total C matrix.")
            shells.append(c_total.shape[1])
            logger.debug(f"{c_total.shape=}")

        self.shells = shells
        logger.debug(f"Shell indices: {shells}")

        local_scf.mo_coeff = c_total
        logger.debug("Completed Concentric Localization.")
        return local_scf
