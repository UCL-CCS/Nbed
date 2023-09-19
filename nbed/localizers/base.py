"""Base Localizer Class."""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from pyscf import scf
from pyscf.lib import StreamObject
from pyscf.lo import vvo

# from ..utils import restricted_float_percentage

logger = logging.getLogger(__name__)


class Localizer(ABC):
    """Object used to localise molecular orbitals (MOs) using different localization schemes.

    Running localization returns active and environment systems.

    Note:
    The major improvement of IBOs over PM orbitals is that they are based on IAO charges instead of the erratic
    Mulliken charges. As a result, IBOs are always well-defined.  (Ref: J. Chem. Theory Comput. 2013, 9, 4834−4843)

    Args:
        global_ks (gto.Mole): PySCF molecule object
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
        global_ks: StreamObject,
        n_active_atoms: int,
        occ_cutoff: Optional[float] = 0.95,
        virt_cutoff: Optional[float] = 0.95,
        run_virtual_localization: Optional[bool] = False,
    ):
        """Initialise class."""
        logger.debug("Initialising Localizer.")
        if global_ks.mo_coeff is None:
            logger.debug("SCF method not initialised, running now...")
            global_ks.run()
            logger.debug("SCF method initialised.")

        self._global_ks = global_ks
        self._n_active_atoms = n_active_atoms

        self._occ_cutoff = self._valid_threshold(occ_cutoff)
        self._virt_cutoff = self._valid_threshold(virt_cutoff)
        self._run_virtual_localization = run_virtual_localization
        self._restricted_scf = isinstance(self._global_ks, scf.hf.RHF)

        # Run the localization procedure
        self.run()

    def _valid_threshold(self, threshold: float):
        """Checks if threshold is within 0-1 range (percentage).

        Args:
            threshold (float): input number between 0 and 1 (inclusive)

        Returns:
            threshold (float): input percentage
        """
        if 0.0 <= threshold <= 1.0:
            logger.debug("Localizer threshold valid.")
            return threshold
        else:
            logger.error("Localizer threshold not valid.")
            raise ValueError(f"threshold: {threshold} is not in range [0,1] inclusive")

    def _localize(
        self,
    ) -> Tuple[Tuple, Union[Tuple, None]]:
        """Localise orbitals using SPADE.

        Returns:
            active_MO_inds (np.array): 1D array of active occupied MO indices
            enviro_MO_inds (np.array): 1D array of environment occupied MO indices
            c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
            c_enviro (np.array): C matrix of localized occupied ennironment MOs
            c_loc_occ (np.array): full C matrix of localized occupied MOs
        """
        if self._restricted_scf:
            alpha = self._localize_spin(
                self._global_ks.mo_coeff, self._global_ks.mo_occ
            )
            beta = None
        else:
            alpha = self._localize_spin(
                self._global_ks.mo_coeff[0], self._global_ks.mo_occ[0]
            )
            beta = self._localize_spin(
                self._global_ks.mo_coeff[1], self._global_ks.mo_occ[1]
            )

        return (alpha, beta)

    @abstractmethod
    def _localize_spin(self, c_matrix: np.ndarray, occupancy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Localize orbitals of one spin.

        Args:
            c_matrix (np.ndarray): Unlocalized C matrix of occupied orbitals.
            occupancy (np.ndarray): Occupancy of orbitals.

        Returns:
            np.ndarray: Localized C matrix of occupied orbitals.
        """
        pass

    def _check_values(self) -> None: # Needs clarification
        """Check that output values make sense."""
        logger.debug("Checking density matrix partition.")
        # checking denisty matrix parition makes sense:
        dm_localised_full_system = 2 * self._c_loc_occ @ self._c_loc_occ.conj().T
        bool_density_flag = np.allclose(
            dm_localised_full_system, self.dm_active + self.dm_enviro
        )
        logger.debug(f"y_active + y_enviro = y_total is: {bool_density_flag}")

        if not bool_density_flag:
            raise ValueError("gamma_full != gamma_active + gamma_enviro")

        # check number of electrons is still the same after orbitals have been localized (change of basis)
        s_ovlp = self._global_ks.get_ovlp()
        n_active_electrons = np.trace(self.dm_active @ s_ovlp)
        n_enviro_electrons = np.trace(self.dm_enviro @ s_ovlp)

        if self._restricted_scf is False:
            n_active_electrons += np.trace(self.beta_dm_active @ s_ovlp)
            n_enviro_electrons += np.trace(self.beta_dm_enviro @ s_ovlp)

        n_all_electrons = self._global_ks.mol.nelectron
        electron_number_match = np.isclose(
            (n_active_electrons + n_enviro_electrons), n_all_electrons
        )
        logger.debug(f"N total electrons: {n_all_electrons}")
        logger.debug(
            f"N active electrons + N env electrons = {electron_number_match}"
        )
        if not electron_number_match:
            message = "Number of electrons in localized orbitals is not consistent."
            logger.error(message)
            raise ValueError(message)

    def _localize_virtual_orbs(self) -> None:
        """Localise virtual (unoccupied) orbitals using different localization schemes in PySCF.

        Args:
            global_ks (StreamObject): PySCF molecule object
            n_active_atoms (int): Number of active atoms
            virt_cutoff (float): Threshold for selecting unoccupied (virtual) active regio

        Returns:
            c_virtual_loc (np.array): C matrix of localized virtual MOs (columns define MOs)
            active_virtual_MO_inds (np.array): 1D array of active virtual MO indices
            enviro_virtual_MO_inds (np.array): 1D array of environment virtual MO indices
        """
        logger.debug("Localizing virtual orbitals.")
        n_occupied_orbitals = np.count_nonzero(self._global_ks.mo_occ == 2)
        c_std_occ = self._global_ks.mo_coeff[:, :n_occupied_orbitals]
        c_std_virt = self._global_ks.mo_coeff[:, self._global_ks.mo_occ < 2]

        c_virtual_loc = vvo.vvo(
            self._global_ks.mol, c_std_occ, c_std_virt, iaos=None, s=None, verbose=None
        )

        ao_slice_matrix = self._global_ks.mol.aoslice_by_atom()

        # TODO: Check the following:
        # S_ovlp = global_ks.get_ovlp()
        # S_half = sp.linalg.fractional_matrix_power(S_ovlp , 0.5)
        # C_loc_occ_ORTHO = S_half@C_loc_occ_full
        # run numerator_all and denominator_all in ortho basis

        # find indices of AO of active atoms
        ao_active_inds = np.arange(
            ao_slice_matrix[0, 2], ao_slice_matrix[self._n_active_atoms - 1, 3]
        )

        # active AOs coeffs for a given MO j
        numerator_all = np.einsum("ij->j", (c_virtual_loc[ao_active_inds, :]) ** 2)
        # all AOs coeffs for a given MO j
        denominator_all = np.einsum("ij->j", c_virtual_loc ** 2)

        active_percentage_MO = numerator_all / denominator_all

        logger.debug("Virtual orbitals localized.")
        logger.debug(f"(active_AO^2)/(all_AO^2): {np.around(active_percentage_MO,4)}")
        logger.debug(f"threshold for active part: {self._virt_cutoff}")

        # NOT IN USE
        # add constant occupied index
        # active_virtual_MO_inds = (
        #     np.where(active_percentage_MO > self.virt_cutoff)[0] + c_std_occ.shape[1]
        # )
        # enviro_virtual_MO_inds = np.array(
        #     [
        #         i
        #         for i in range(
        #             c_std_occ.shape[1], c_std_occ.shape[1] + c_virtual_loc.shape[1]
        #         )
        #         if i not in active_virtual_MO_inds
        #     ]
        # )

        return c_virtual_loc

    def run(self, sanity_check: bool = False) -> None:
        """Function that runs localization.

        Args:
            sanity_check (bool): optional flag to check denisty matrices and electron number after orbital localization
                                 makes sense
        """
        alpha, beta = self._localize()

        (
            self.active_MO_inds,
            self.enviro_MO_inds,
            self.c_active,
            self.c_enviro,
            self._c_loc_occ,
        ) = alpha

        self.dm_active = self.c_active @ self.c_active.T
        self.dm_enviro = self.c_enviro @ self.c_enviro.T

        # For resticted methods
        if beta is None:
            self.dm_active *= 2.0
            self.dm_enviro *= 2.0
            self.beta_active_MO_inds = None
            self.beta_enviro_MO_inds = None
            self.beta_c_active = None
            self.beta_c_enviro = None
            self._beta_c_loc_occ = None
            self.beta_dm_active = np.zeros(self.dm_active.shape)
            self.beta_dm_enviro = np.zeros(self.dm_enviro.shape)
        else:
            (
                self.beta_active_MO_inds,
                self.beta_enviro_MO_inds,
                self.beta_c_active,
                self.beta_c_enviro,
                self._beta_c_loc_occ,
            ) = beta

            self.beta_dm_active = self.beta_c_active @ self.beta_c_active.T
            self.beta_dm_enviro = self.beta_c_enviro @ self.beta_c_enviro.T

        if sanity_check is True:
            self._check_values()

        if self._run_virtual_localization is True:
            logger.error("Virtual localization is not implemented.")
            # c_virtual = self._localize_virtual_orbs()
            # logger.error("Defualting to unlocalized virtual orbitals.")
            # c_virtual = self._global_ks.mo_coeff[:, self._global_ks.mo_occ < 2]
        else:
            logger.debug("Not localizing virtual orbitals.")
            # appends standard virtual orbitals from SCF calculation (NOT localized in any way)
            # c_virtual = self._global_ks.mo_coeff[:, self._global_ks.mo_occ < 2]

        # Unused
        # self.c_loc_occ_and_virt = np.hstack((self._c_loc_occ, c_virtual))

        logger.debug("Localization complete.")
