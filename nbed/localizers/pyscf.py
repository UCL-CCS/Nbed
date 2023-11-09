"""PySCF Localizer Class."""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
from pyscf import lo
from pyscf.lib import StreamObject
from pyscf.lo import vvo

from .base import Localizer

logger = logging.getLogger(__name__)


class PySCFLocalizer(Localizer, ABC):
    """Object used to localise molecular orbitals (MOs) using PySCF localization functions.

    Running localization returns active and environment systems.

    Note:
    The major improvement of IBOs over PM orbitals is that they are based on IAO charges instead of the erratic
    Mulliken charges. As a result, IBOs are always well-defined.  (Ref: J. Chem. Theory Comput. 2013, 9, 4834−4843)


    Args:
        global_scf (gto.Mole): PySCF molecule object
        n_active_atoms (int): Number of active atoms
        occ_cutoff (float): Threshold for selecting occupied active region (only requried if
                                spade localization is NOT used)
        virt_cutoff (float): Threshold for selecting unoccupied (virtual) active region (required for
                                spade approach too!)

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
        global_scf: StreamObject,
        n_active_atoms: int,
        occ_cutoff: Optional[float] = 0.95,
        virt_cutoff: Optional[float] = 0.95,
    ):
        """Initialize PySCF Localizer."""
        self.occ_cutoff = self._valid_threshold(occ_cutoff)
        self.virt_cutoff = self._valid_threshold(virt_cutoff)
        super().__init__(
            global_scf,
            n_active_atoms,
        )

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

    @abstractmethod
    def _pyscf_method(self, c_std_occ):
        """Abstract method containing the PySCF method to use.

        Args:
            c_std_occ (np.ndarray): Unlocalized C matrix of occupied orbitals.
        """
        pass

    def _localize_spin(
        self, c_matrix: np.ndarray, occupancy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Localize orbitals of one spin using PySCF.

        Args:
            c_matrix (np.ndarray): Unlocalized C matrix of occupied orbitals.
            occupancy (np.ndarray): Occupancy of orbitals.

        Returns:
            np.ndarray: Localized C matrix of occupied orbitals.
        """
        n_occupied_orbitals = np.count_nonzero(occupancy)
        c_std_occ = c_matrix[:, :n_occupied_orbitals]

        c_loc_occ = self._pyscf_method(c_std_occ)

        ao_slice_matrix = self._global_scf.mol.aoslice_by_atom()

        # TODO: Check the following:
        # S_ovlp = pyscf_scf.get_ovlp()
        # S_half = sp.linalg.fractional_matrix_power(S_ovlp , 0.5)
        # C_loc_occ_ORTHO = S_half@C_loc_occ_full
        # run numerator_all and denominator_all in ortho basis

        # find indices of AO of active atoms
        ao_active_inds = np.arange(
            ao_slice_matrix[0, 2], ao_slice_matrix[self._n_active_atoms - 1, 3]
        )
        # active AOs coeffs for a given MO j
        numerator_all = np.einsum("ij->j", (c_loc_occ[ao_active_inds, :]) ** 2)

        # all AOs coeffs for a given MO j
        denominator_all = np.einsum("ij->j", c_loc_occ**2)

        mo_active_share = numerator_all / denominator_all

        logger.debug(f"(active_AO^2)/(all_AO^2): {np.around(mo_active_share, 4)}")
        logger.debug(f"threshold for active part: {self.occ_cutoff}")

        active_MO_inds = np.where(mo_active_share > self.occ_cutoff)[0]
        # print(active_MO_inds)

        all_ao_shares_same_bool = np.allclose(
            np.zeros_like(mo_active_share),
            mo_active_share - mo_active_share.sum() / len(mo_active_share),
        )

        if all_ao_shares_same_bool:
            # case for highly symmetric molecules
            # overlap is the same everywhere hence everything goes into env or act part

            # edge case put half and half!
            logger.warning(
                "AO subsystem selection % same everywhere. Splitting half and half"
            )
            print(f"mo_active_share: {mo_active_share}")
            active_MO_inds = np.array(range(0, c_loc_occ.shape[1] // 2), dtype=int)
        elif len(active_MO_inds) == 0:
            # if no active indices, then take largest possible overlap
            mo_active_percentage_inshare = mo_active_share.argsort()[::-1]
            active_MO_inds = mo_active_percentage_inshare[:1]  # take first element
            logger.warning("no active AOs - forcing one to be active")
            print(f"active system %: {mo_active_share[active_MO_inds][0]} \n")

        enviro_MO_inds = np.array(
            [i for i in range(c_loc_occ.shape[1]) if i not in active_MO_inds]
        )

        # define active MO orbs and environment
        #    take MO (columns of C_matrix) that have high dependence from active AOs
        c_active = c_loc_occ[:, active_MO_inds]

        if len(enviro_MO_inds) == 0:
            # case for when no environement
            logger.warning("No environment electronic density")
            c_enviro = np.zeros((c_active.shape[0], 1))
        else:
            c_enviro = c_loc_occ[:, enviro_MO_inds]

        # storing condition used to select env system
        self.enviro_selection_condition = mo_active_share

        logger.debug("PySCF localization complete.")
        return active_MO_inds, enviro_MO_inds, c_active, c_enviro, c_loc_occ

    def _localize_virtual_spin(
        self, c_matrix: np.ndarray, virt_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Localise virtual (unoccupied) orbitals using different localization schemes in PySCF.

        Args:
            global_scf (StreamObject): PySCF molecule object
            n_active_atoms (int): Number of active atoms
            virt_threshold (float): Threshold for selecting unoccupied (virtual) active MOs.

        Returns:
            c_virtual_loc (np.array): C matrix of localized virtual MOs (columns define MOs)
            active_virtual_MO_inds (np.array): 1D array of active virtual MO indices
            enviro_virtual_MO_inds (np.array): 1D array of environment virtual MO indices
        """
        logger.debug("Localizing virtual orbitals.")
        n_occupied_orbitals = np.count_nonzero(self._global_scf.mo_occ == 2)
        c_std_occ = self._global_scf.mo_coeff[:, :n_occupied_orbitals]
        c_std_virt = self._global_scf.mo_coeff[:, self._global_scf.mo_occ < 2]

        c_virtual_loc = vvo.vvo(
            self._global_scf.mol, c_std_occ, c_std_virt, iaos=None, s=None, verbose=None
        )

        ao_slice_matrix = self._global_scf.mol.aoslice_by_atom()

        # TODO: Check the following:
        # S_ovlp = global_scf.get_ovlp()
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
        denominator_all = np.einsum("ij->j", c_virtual_loc**2)

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

    def localize_virtual(local_scf: StreamObject) -> StreamObject:
        """Localise virtual (unoccupied) obitals using PySCF method.

        [1] D. Claudino and N. J. Mayhall, "Simple and Efficient Truncation of Virtual
        Spaces in Embedded Wave Functions via Concentric Localization", Journal of Chemical
        Theory and Computation, vol. 15, no. 11, pp. 6085-6096, Nov. 2019,
        doi: 10.1021/ACS.JCTC.9B00682.

        Args:
            local_scf (StreamObject): SCF object with occupied orbitals localized.

        Returns:
            StreamObject: Fully Localized SCF object.
        """
        raise NotImplementedError(
            "Virtual orbital localization not implemented for PySCF methods."
        )
        return local_scf


class PMLocalizer(PySCFLocalizer):
    """Object used to localise molecular orbitals (MOs) using Pipek-Mezey localization.

    Running localization returns active and environment systems.

    Args:
        global_scf (gto.Mole): PySCF molecule object
        n_active_atoms (int): Number of active atoms
        occ_cutoff (float): Threshold for selecting occupied active region (only requried if
                                spade localization is NOT used)
        virt_cutoff (float): Threshold for selecting unoccupied (virtual) active region (required for
                                spade approach too!)

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
        global_scf: StreamObject,
        n_active_atoms: int,
        occ_cutoff: Optional[float] = 0.95,
        virt_cutoff: Optional[float] = 0.95,
    ):
        """Initialize Localizer."""
        super().__init__(
            global_scf,
            n_active_atoms,
            occ_cutoff=occ_cutoff,
            virt_cutoff=virt_cutoff,
        )

    def _pyscf_method(self, c_std_occ: np.ndarray) -> np.ndarray:
        """Abstract method containing the PySCF method to use.

        Args:
            c_std_occ (np.ndarray): Unlocalized C matrix of occupied orbitals.
        """
        logger.debug("Using Pipek-Mezey method.")
        # Localise orbitals using Pipek-Mezey localization scheme.
        # This maximizes the sum of orbital-dependent partial charges on the nuclei.

        pipmez = lo.PipekMezey(self._global_scf.mol, c_std_occ)

        # The atomic population projection scheme.
        # 'mulliken', 'meta-lowdin', 'iao', 'becke'
        pipmez.pop_method = "meta-lowdin"

        # run localization
        return pipmez.kernel()


class BOYSLocalizer(PySCFLocalizer):
    """Object used to localise molecular orbitals (MOs) using BOYS localization.

    Running localization returns active and environment systems.

    Args:
        global_scf (gto.Mole): PySCF molecule object
        n_active_atoms (int): Number of active atoms
        localization_method (str): String of orbital localization method (spade, pipekmezey, boys, ibo)
        occ_cutoff (float): Threshold for selecting occupied active region (only requried if
                                spade localization is NOT used)
        virt_cutoff (float): Threshold for selecting unoccupied (virtual) active region (required for
                                spade approach too!)

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
        global_scf: StreamObject,
        n_active_atoms: int,
        occ_cutoff: Optional[float] = 0.95,
        virt_cutoff: Optional[float] = 0.95,
    ):
        """Initialize Localizer."""
        super().__init__(
            global_scf,
            n_active_atoms,
            occ_cutoff=occ_cutoff,
            virt_cutoff=virt_cutoff,
        )

    def _pyscf_method(self, c_std_occ: np.ndarray) -> np.ndarray:
        """Abstract method containing the PySCF method to use.

        Args:
            c_std_occ (np.ndarray): Unlocalized C matrix of occupied orbitals.
        """
        logger.debug("Using BOYS method.")
        #  Minimizes the spatial extent of the orbitals by minimizing a certain function.
        boys_SCF = lo.boys.Boys(self._global_scf.mol, c_std_occ)
        return boys_SCF.kernel()


class IBOLocalizer(PySCFLocalizer):
    """Object used to localise molecular orbitals (MOs) using IBO localization.

    Running localization returns active and environment systems.

    Args:
        global_scf (gto.Mole): PySCF molecule object
        n_active_atoms (int): Number of active atoms
        occ_cutoff (float): Threshold for selecting occupied active region (only requried if
                                spade localization is NOT used)
        virt_cutoff (float): Threshold for selecting unoccupied (virtual) active region (required for
                                spade approach too!)

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
        global_scf: StreamObject,
        n_active_atoms: int,
        occ_cutoff: Optional[float] = 0.95,
        virt_cutoff: Optional[float] = 0.95,
    ):
        """Initialise Localizer."""
        super().__init__(
            global_scf,
            n_active_atoms,
            occ_cutoff=occ_cutoff,
            virt_cutoff=virt_cutoff,
        )

    def _pyscf_method(self, c_std_occ: np.ndarray) -> np.ndarray:
        """Abstract method containing the PySCF method to use.

        Args:
            c_std_occ (np.ndarray): Unlocalized C matrix of occupied orbitals.
        """
        logger.debug("Using IBO method.")
        # Intrinsic bonding orbitals.
        iaos = lo.iao.iao(self._global_scf.mol, c_std_occ)
        # Orthogonalize IAO
        iaos = lo.vec_lowdin(iaos, self._global_scf.get_ovlp())
        c_loc_occ = lo.ibo.ibo(
            self._global_scf.mol, c_std_occ, locmethod="IBO", iaos=iaos
        )
        return c_loc_occ
