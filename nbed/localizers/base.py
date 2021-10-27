"""Base Localizer Class."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from cached_property import cached_property

import numpy as np
import scipy as sp
from pyscf import gto, lo
from pyscf.lib import StreamObject
from pyscf.lo import vvo
from scipy import linalg

logger = logging.getLogger(__name__)


class Localizer(ABC):
    """Object used to localise molecular orbitals (MOs) using different localization schemes.

    Running localisation returns active and environment systems.

    Note:
    The major improvement of IBOs over PM orbitals is that they are based on IAO charges instead of the erratic
    Mulliken charges. As a result, IBOs are always well-defined.  (Ref: J. Chem. Theory Comput. 2013, 9, 4834−4843)

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
        pyscf_rks: StreamObject,
        n_active_atoms: int,
        occ_cutoff: Optional[float] = 0.95,
        virt_cutoff: Optional[float] = 0.95,
        run_virtual_localization: Optional[bool] = False,
    ):

        if pyscf_rks.mo_coeff is None:
            logger.debug("SCF method not initialised, running now...")
            pyscf_rks.run()
            logger.debug("SCF method initialised.")

        if self.output not in ["qiskit", "pennylane", "openfermion"]:
            logger.error(
                "Invalid output format %s,. Choose from 'qiskit', 'pennylane' or 'openfermion'."
            )
            config_valid = False

        self._pyscf_rks = pyscf_rks
        self._n_active_atoms = n_active_atoms
        self._occ_cutoff = occ_cutoff
        self._virt_cutoff = virt_cutoff
        self._run_virtual_localization = run_virtual_localization

        # Run the localization procedure
        self.run()

    @cached_property
    def localized_rks(self) -> StreamObject:
        """Localize the input RKS object."""

        local_rks = self._global_rks

        hcore_std = local_rks.get_hcore()
        local_rks.get_hcore = (
            lambda *args: self._local_basis_transform.conj().T
            @ hcore_std
            @ self._local_basis_transform
        )

        local_rks.get_veff = (
            lambda mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1: self._rks_veff(
                local_rks, self._local_basis_transform, dm=dm, check_result=True
            )
        )

        # overwrite C matrix with localised orbitals
        local_rks.mo_coeff = self.localized_system.c_loc_occ_and_virt
        dm_loc = local_rks.make_rdm1(
            mo_coeff=local_rks.mo_coeff, mo_occ=local_rks.mo_occ
        )

        # fock_locbasis = _local_rks.get_hcore() + _local_rks.get_veff(dm=dm_loc)
        fock_locbasis = local_rks.get_fock(dm=dm_loc)

        # orbital_energies_std = _local_rks.mo_energy
        orbital_energies_loc = np.diag(
            local_rks.mo_coeff.conj().T @ fock_locbasis @ local_rks.mo_coeff
        )
        local_rks.mo_energy = orbital_energies_loc

        # # check electronic energy matches standard global calc
        # local_rks_total_energy_loc = local_rks.energy_tot(dm=dm_loc)
        # if not np.isclose(self._local_rks.e_tot, local_rks_total_energy_loc):
        #     raise ValueError(
        #         "electronic energy of standard calculation not matching localized calculation"
        #     )

        # check if mo energies match
        # orbital_energies_std = _local_rks.mo_energy
        # if not np.allclose(orbital_energies_std, orbital_energies_loc):
        #     raise ValueError('orbital energies of standard calc not matching localized calc')

        return local_rks

    @abstractmethod
    def _localize(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Abstract method which should handle localization.

        Returns:
            active_MO_inds (np.array): 1D array of active occupied MO indices
            enviro_MO_inds (np.array): 1D array of environment occupied MO indices
            c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
            c_enviro (np.array): C matrix of localized occupied ennironment MOs
            c_loc_occ (np.array): full C matrix of localized occupied MOs
        """
        pass

    def _check_values(self) -> None:
        """Check that output values make sense."""
        # checking denisty matrix parition makes sense:
        dm_localised_full_system = 2 * self._c_loc_occ @ self._c_loc_occ.conj().T
        bool_density_flag = np.allclose(
            dm_localised_full_system, self.dm_active + self.dm_enviro
        )
        logger.debug(f"y_active + y_enviro = y_total is: {bool_density_flag}")
        if not bool_density_flag:
            raise ValueError("gamma_full != gamma_active + gamma_enviro")

        # check number of electrons is still the same after orbitals have been localized (change of basis)
        s_ovlp = self._pyscf_rks.get_ovlp()
        n_active_electrons = np.trace(self.dm_active @ s_ovlp)
        n_enviro_electrons = np.trace(self.dm_enviro @ s_ovlp)
        n_all_electrons = self._pyscf_rks.mol.nelectron
        bool_flag_electron_number = np.isclose(
            (n_active_electrons + n_enviro_electrons), n_all_electrons
        )
        logger.debug(
            f"N_active_elec + N_environment_elec = N_total_elec is: {bool_flag_electron_number}"
        )
        if not bool_flag_electron_number:
            raise ValueError("number of electrons in localized orbitals is incorrect")

    def _localize_virtual_orbs(self) -> None:
        """Localise virtual (unoccupied) orbitals using different localization schemes in PySCF.

        Args:
            pyscf_rks (StreamObject): PySCF molecule object
            n_active_atoms (int): Number of active atoms
            virt_cutoff (float): Threshold for selecting unoccupied (virtual) active regio

        Returns:
            c_virtual_loc (np.array): C matrix of localized virtual MOs (columns define MOs)
            active_virtual_MO_inds (np.array): 1D array of active virtual MO indices
            enviro_virtual_MO_inds (np.array): 1D array of environment virtual MO indices
        """
        logger.debug("Localizing virtual orbitals.")
        n_occupied_orbitals = np.count_nonzero(self._pyscf_rks.mo_occ == 2)
        c_std_occ = self._pyscf_rks.mo_coeff[:, :n_occupied_orbitals]
        c_std_virt = self._pyscf_rks.mo_coeff[:, self._pyscf_rks.mo_occ < 2]

        c_virtual_loc = vvo.vvo(
            self._pyscf_rks.mol, c_std_occ, c_std_virt, iaos=None, s=None, verbose=None
        )

        ao_slice_matrix = self._pyscf_rks.mol.aoslice_by_atom()

        # TODO: Check the following:
        # S_ovlp = pyscf_rks.get_ovlp()
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
        """Function that runs localisation

        Args:
            sanity_check (bool): optional flag to check denisty matrices and electron number after orbital localization
                                 makes sense
        """
        (
            self.active_MO_inds,
            self.enviro_MO_inds,
            self.c_active,
            self.c_enviro,
            self._c_loc_occ,
        ) = self._localize()

        self.dm_active = 2.0 * self.c_active @ self.c_active.T
        self.dm_enviro = 2.0 * self.c_enviro @ self.c_enviro.T

        if sanity_check is True:
            self._check_values()

        if self._run_virtual_localization is True:
            c_virtual = self._localize_virtual_orbs()
        else:
            # appends standard virtual orbitals from SCF calculation (NOT localized in any way)
            c_virtual = self._pyscf_rks.mo_coeff[:, self._pyscf_rks.mo_occ < 2]

        self.c_loc_occ_and_virt = np.hstack((self._c_loc_occ, c_virtual))
