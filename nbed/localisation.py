"""Orbital localisation methods."""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np
import scipy as sp
from pyscf import gto, lo
from pyscf.lib import StreamObject
from pyscf.lo import vvo
from scipy import linalg

logger = logging.getLogger(__name__)


def orb_change_basis_operator(
    pyscf_scf: StreamObject,
    c_all_localized_and_virt: np.array,
    sanity_check: Optional[bool] = False,
) -> np.ndarray:
    """Construct operator to change basis.

    Get operator that changes from standard canonical orbitals (C_matrix standard) to
    localized orbitals (C_matrix_localized)

    Args:
        pyscf_scf (StreamObject): PySCF molecule object
        c_all_localized_and_virt (np.array): C_matrix of localized orbitals (includes occupied and virtual)
        sanity_check (bool): optional flag to check if change of basis is working properly

    Returns:
        matrix_std_to_loc (np.array): Matrix that maps from standard (canonical) MOs to localized MOs
    """
    s_mat = pyscf_scf.get_ovlp()
    s_half = sp.linalg.fractional_matrix_power(s_mat, 0.5)

    # find orthogonal orbitals
    ortho_std = s_half @ pyscf_scf.mo_coeff
    ortho_loc = s_half @ c_all_localized_and_virt

    # Build change of basis operator (maps between orthonormal basis (canonical and localized)
    unitary_ORTHO_std_onto_loc = np.einsum("ik,jk->ij", ortho_std, ortho_loc)

    if sanity_check:
        if np.allclose(unitary_ORTHO_std_onto_loc @ ortho_loc, ortho_std) is not True:
            raise ValueError(
                "Change of basis incorrect... U_ORTHO_std_onto_loc*C_ortho_loc !=  C_ortho_STD"
            )

        if (
            np.allclose(
                unitary_ORTHO_std_onto_loc.conj().T @ unitary_ORTHO_std_onto_loc,
                np.eye(unitary_ORTHO_std_onto_loc.shape[0]),
            )
            is not True
        ):
            raise ValueError("Change of basis (U_ORTHO_std_onto_loc) is not Unitary!")

    s_neg_half = sp.linalg.fractional_matrix_power(s_mat, -0.5)

    # move back into non orthogonal basis
    matrix_std_to_loc = s_neg_half @ unitary_ORTHO_std_onto_loc @ s_half

    if sanity_check:
        if (
            np.allclose(
                matrix_std_to_loc @ c_all_localized_and_virt, pyscf_scf.mo_coeff
            )
            is not True
        ):
            raise ValueError(
                "Change of basis incorrect... U_std*C_std !=  C_loc_occ_and_virt"
            )

    return matrix_std_to_loc


class Localizer(ABC):
    """Object used to localise molecular orbitals (MOs) using different localization schemes.

    Running localisation returns active and environment systems.

    Note:
    The major improvement of IBOs over PM orbitals is that they are based on IAO charges instead of the erratic
    Mulliken charges. As a result, IBOs are always well-defined.  (Ref: J. Chem. Theory Comput. 2013, 9, 4834−4843)

    Args:
        pyscf_scf (gto.Mole): PySCF molecule object
        n_active_atoms (int): Number of active atoms
        localization_method (str): String of orbital localization method (spade, pipekmezey, boys, ibo)
        occ_THRESHOLD (float): Threshold for selecting occupied active region (only requried if
                                spade localization is NOT used)
        virt_cutoff (float): Threshold for selecting unoccupied (virtual) active region (required for
                                spade approach too!)
        run_virtual_localization (bool): optional flag on whether to perform localization of virtual orbitals.
                                         Note if False appends canonical virtual orbs to C_loc_occ_and_virt matrix

    Attributes:
        c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
        c_enviro (np.array): C matrix of localized occupied ennironment MOs
        c_loc_occ (np.array): C matrix of localized occupied MOs
        dm_active (np.array): active system density matrix
        dm_enviro (np.array): environment system density matrix
        active_MO_inds (np.array): 1D array of active occupied MO indices
        enviro_MO_inds (np.array): 1D array of environment occupied MO indices
        c_loc_occ_and_virt (np.array): Full localized C_matrix (occpuied and virtual)
        active_virtual_MO_inds (np.array): 1D array of active virtual MO indices (set to None if
                                           run_virtual_localization is False)
        enviro_virtual_MO_inds (np.array): 1D array of environment virtual MO indices
                                           (set to None if run_virtual_localization is False)
    """

    def __init__(
        self,
        pyscf_scf: StreamObject,
        n_active_atoms: int,
        occ_THRESHOLD: Optional[float] = 0.95,
        virt_cutoff: Optional[float] = 0.95,
        run_virtual_localization: Optional[bool] = False,
    ):

        if pyscf_scf.mo_coeff is None:
            logger.debug("SCF method not initialised, running now...")
            pyscf_scf.run()
            logger.debug("SCF method initialised.")

        self.pyscf_scf = pyscf_scf
        self.n_active_atoms = n_active_atoms
        self.occ_THRESHOLD = occ_THRESHOLD
        self.virt_cutoff = virt_cutoff
        self.run_virtual_localization = run_virtual_localization

        # attributes
        self.c_active: np.ndarray = None
        self.c_enviro: np.ndarray = None
        self.c_loc_occ: np.ndarray = None
        self.c_loc_occ_and_virt: np.ndarray = None

        self.dm_active: np.ndarray = None
        self.dm_enviro: np.ndarray = None

        self.active_MO_inds: List[int] = None
        self.enviro_MO_inds: List[int] = None
        self.active_virtual_MO_inds: List[int] = None
        self.enviro_virtual_MO_inds: List[int] = None

    @abstractmethod
    def localize(self) -> None:
        """Abstract method which should handle localization.
        Assigns:
            c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
            c_enviro (np.array): C matrix of localized occupied ennironment MOs
            c_loc_occ (np.array): full C matrix of localized occupied MOs
            dm_active (np.array): active system density matrix
            dm_enviro (np.array): environment system density matrix
            active_MO_inds (np.array): 1D array of active occupied MO indices
            enviro_MO_inds (np.array): 1D array of environment occupied MO indices
        """
        pass

    def check_values(self) -> None:
        """Check that output values make sense."""
        # checking denisty matrix parition makes sense:
        dm_localised_full_system = 2 * self.c_loc_occ @ self.c_loc_occ.conj().T
        bool_density_flag = np.allclose(
            dm_localised_full_system, self.dm_active + self.dm_enviro
        )
        logger.debug(f"y_active + y_enviro = y_total is: {bool_density_flag}")
        if not bool_density_flag:
            raise ValueError("gamma_full != gamma_active + gamma_enviro")

        # check number of electrons is still the same after orbitals have been localized (change of basis)
        s_ovlp = self.pyscf_scf.get_ovlp()
        n_active_electrons = np.trace(self.dm_active @ s_ovlp)
        n_enviro_electrons = np.trace(self.dm_enviro @ s_ovlp)
        n_all_electrons = self.pyscf_scf.mol.nelectron
        bool_flag_electron_number = np.isclose(
            (n_active_electrons + n_enviro_electrons), n_all_electrons
        )
        logger.debug(
            f"N_active_elec + N_environment_elec = N_total_elec is: {bool_flag_electron_number}"
        )
        if not bool_flag_electron_number:
            raise ValueError("number of electrons in localized orbitals is incorrect")

    def localize_virtual_orbs(self) -> None:
        """Localise virtual (unoccupied) orbitals using different localization schemes in PySCF.

        Args:
            pyscf_scf (StreamObject): PySCF molecule object
            n_active_atoms (int): Number of active atoms
            virt_cutoff (float): Threshold for selecting unoccupied (virtual) active regio

        Returns:
            c_virtual_loc (np.array): C matrix of localized virtual MOs (columns define MOs)
            active_virtual_MO_inds (np.array): 1D array of active virtual MO indices
            enviro_virtual_MO_inds (np.array): 1D array of environment virtual MO indices
        """

        n_occupied_orbitals = np.count_nonzero(self.pyscf_scf.mo_occ == 2)
        c_std_occ = self.pyscf_scf.mo_coeff[:, :n_occupied_orbitals]
        c_std_virt = self.pyscf_scf.mo_coeff[:, self.pyscf_scf.mo_occ < 2]

        c_virtual_loc = vvo.vvo(
            self.pyscf_scf.mol, c_std_occ, c_std_virt, iaos=None, s=None, verbose=None
        )

        ao_slice_matrix = self.pyscf_scf.mol.aoslice_by_atom()

        # TODO: Check the following:
        # S_ovlp = pyscf_scf.get_ovlp()
        # S_half = sp.linalg.fractional_matrix_power(S_ovlp , 0.5)
        # C_loc_occ_ORTHO = S_half@C_loc_occ_full
        # run numerator_all and denominator_all in ortho basis

        # find indices of AO of active atoms
        ao_active_inds = np.arange(
            ao_slice_matrix[0, 2], ao_slice_matrix[self.n_active_atoms - 1, 3]
        )

        # active AOs coeffs for a given MO j
        numerator_all = np.einsum("ij->j", (c_virtual_loc[ao_active_inds, :]) ** 2)
        # all AOs coeffs for a given MO j
        denominator_all = np.einsum("ij->j", c_virtual_loc ** 2)

        active_percentage_MO = numerator_all / denominator_all

        logger.debug(f"(active_AO^2)/(all_AO^2): {np.around(active_percentage_MO,4)}")
        logger.debug(f"threshold for active part: {self.virt_cutoff}")

        # add constant occupied index
        active_virtual_MO_inds = (
            np.where(active_percentage_MO > self.virt_cutoff)[0] + c_std_occ.shape[1]
        )
        enviro_virtual_MO_inds = np.array(
            [
                i
                for i in range(
                    c_std_occ.shape[1], c_std_occ.shape[1] + c_virtual_loc.shape[1]
                )
                if i not in active_virtual_MO_inds
            ]
        )

    def run(self, sanity_check: bool = False):
        """Function that runs localisation

        Args:
            sanity_check (bool): optional flag to check denisty matrices and electron number after orbital localization
                                 makes sense

        Returns:
            None
        """
        self.localize()

        if sanity_check is True:
            self.check_values()

        if self.run_virtual_localization is True:
            c_virtual = self.localize_virtual_orbs()
        else:
            # appends standard virtual orbitals from SCF calculation (NOT localized in any way)
            self.active_virtual_MO_inds = None
            self.enviro_virtual_MO_inds = None
            c_virtual = self.pyscf_scf.mo_coeff[:, self.pyscf_scf.mo_occ < 2]

        self.c_loc_occ_and_virt = np.hstack((self.c_loc_occ, c_virtual))

        return None


class SpadeLocalizer(Localizer):
    """Localizer Class to carry out SPADE"""

    def __init__(
        self,
        pyscf_scf: gto.Mole,
        n_active_atoms: int,
        localization_method: str,
        occ_THRESHOLD: Optional[float] = 0.95,
        virt_cutoff: Optional[float] = 0.95,
        run_virtual_localization: Optional[bool] = False,
    ):
        super().__init__(
            pyscf_scf,
            n_active_atoms,
            localization_method,
            occ_THRESHOLD=occ_THRESHOLD,
            virt_cutoff=virt_cutoff,
            run_virtual_localization=run_virtual_localization,
        )

    def localize(self) -> None:
        """Localise orbitals using SPADE.

        Assigns:
            c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
            c_enviro (np.array): C matrix of localized occupied ennironment MOs
            c_loc_occ (np.array): full C matrix of localized occupied MOs
            dm_active (np.array): active system density matrix
            dm_enviro (np.array): environment system density matrix
            active_MO_inds (np.array): 1D array of active occupied MO indices
            enviro_MO_inds (np.array): 1D array of environment occupied MO indices
        """
        logger.info("Localising with SPADE.")
        n_occupied_orbitals = np.count_nonzero(self.pyscf_scf.mo_occ == 2)
        occupied_orbitals = self.pyscf_scf.mo_coeff[:, :n_occupied_orbitals]

        n_act_aos = self.pyscf_scf.mol.aoslice_by_atom()[self.n_active_atoms - 1][-1]
        logger.debug(f"{n_act_aos} active AOs.")

        ao_overlap = self.pyscf_scf.get_ovlp()

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
        self.active_MO_inds = np.arange(n_act_mos)
        self.enviro_MO_inds = np.arange(n_act_mos, n_act_mos + n_env_mos)

        # Defining active and environment orbitals and density
        self.c_active = occupied_orbitals @ right_vectors.T[:, :n_act_mos]
        self.c_enviro = occupied_orbitals @ right_vectors.T[:, n_act_mos:]
        self.dm_active = 2.0 * self.c_active @ self.c_active.T
        self.dm_enviro = 2.0 * self.c_enviro @ self.c_enviro.T

        self.c_loc_occ = occupied_orbitals @ right_vectors.T


class PySCFLocalizer(Localizer):
    """Class to run localization using PySCF functions."""

    def __init__(
        self,
        pyscf_scf: StreamObject,
        n_active_atoms: int,
        localization_method: str,
        occ_THRESHOLD: Optional[float] = 0.95,
        virt_cutoff: Optional[float] = 0.95,
        run_virtual_localization: Optional[bool] = False,
    ):
        super().__init__(
            pyscf_scf,
            n_active_atoms,
            occ_THRESHOLD=occ_THRESHOLD,
            virt_cutoff=virt_cutoff,
            run_virtual_localization=run_virtual_localization,
        )
        self.method = localization_method.lower()

    def localize(self) -> None:
        """Localise orbitals using PySCF localization schemes.

        Args:
            method (str): String of orbital localization method: 'pipekmezey', 'boys' or 'ibo'
        """

        n_occupied_orbitals = np.count_nonzero(self.pyscf_scf.mo_occ == 2)
        c_std_occ = self.pyscf_scf.mo_coeff[:, :n_occupied_orbitals]

        if self.method == "pipekmezey":
            # Localise orbitals using Pipek-Mezey localization scheme.
            # This maximizes the sum of orbital-dependent partial charges on the nuclei.

            pipmez = lo.PipekMezey(self.pyscf_scf.mol, c_std_occ)

            # The atomic population projection scheme.
            # 'mulliken' 'meta-lowdin', 'iao', 'becke'
            pipmez.pop_method = "meta-lowdin"

            # run localization
            self.c_loc_occ = pipmez.kernel()

        elif self.method == "boys":
            #  Minimizes the spatial extent of the orbitals by minimizing a certain function.
            boys_SCF = lo.boys.Boys(self.pyscf_scf.mol, c_std_occ)
            self.c_loc_occ = boys_SCF.kernel()

        elif self.method == "ibo":
            # Intrinsic bonding orbitals.
            iaos = lo.iao.iao(self.pyscf_scf.mol, c_std_occ)
            # Orthogonalize IAO
            iaos = lo.vec_lowdin(iaos, self.pyscf_scf.get_ovlp())
            self.c_loc_occ = lo.ibo.ibo(
                self.pyscf_scf.mol, c_std_occ, locmethod="IBO", iaos=iaos
            )
        else:
            raise ValueError(f"unknown localization method {self.method}.")

        ao_slice_matrix = self.pyscf_scf.mol.aoslice_by_atom()

        # TODO: Check the following:
        # S_ovlp = pyscf_scf.get_ovlp()
        # S_half = sp.linalg.fractional_matrix_power(S_ovlp , 0.5)
        # C_loc_occ_ORTHO = S_half@C_loc_occ_full
        # run numerator_all and denominator_all in ortho basis

        # find indices of AO of active atoms
        ao_active_inds = np.arange(
            ao_slice_matrix[0, 2], ao_slice_matrix[self.n_active_atoms - 1, 3]
        )
        # active AOs coeffs for a given MO j
        numerator_all = np.einsum("ij->j", (self.c_loc_occ[ao_active_inds, :]) ** 2)

        # all AOs coeffs for a given MO j
        denominator_all = np.einsum("ij->j", self.c_loc_occ ** 2)

        MO_active_percentage = numerator_all / denominator_all

        logger.debug(f"(active_AO^2)/(all_AO^2): {np.around(MO_active_percentage, 4)}")
        logger.debug(f"threshold for active part: {self.occ_THRESHOLD}")

        active_MO_inds = np.where(MO_active_percentage > self.occ_THRESHOLD)[0]
        enviro_MO_inds = np.array(
            [i for i in range(self.c_loc_occ.shape[1]) if i not in active_MO_inds]
        )

        # define active MO orbs and environment
        #    take MO (columns of C_matrix) that have high dependence from active AOs
        self.c_active = self.c_loc_occ[:, active_MO_inds]
        self.c_enviro = self.c_loc_occ[:, enviro_MO_inds]

        self.n_act_mos = len(active_MO_inds)
        self.n_env_mos = len(enviro_MO_inds)

        logger.debug(f"{self.n_act_mos} active MOs.")
        logger.debug(f"{self.n_env_mos} environment MOs.")

        self.dm_active = 2.0 * self.c_active @ self.c_active.T
        self.dm_enviro = 2.0 * self.c_enviro @ self.c_enviro.T
