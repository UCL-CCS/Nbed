"""
File to contain localisations.
"""

from typing import Callable, Tuple

import numpy as np
from scipy import linalg
import logging
from pyscf import gto, lo
from pyscf.lo import vvo
import scipy as sp

logger = logging.getLogger(__name__)


def spade(scf_method: Callable, n_active_atoms: int
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Localise orbitals using SPADE.

    Args:
        scf_method (gto.Mole): PySCF SCF mol object
        n_active_atoms (int): Number of active atoms
    Returns:
        c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
        c_enviro (np.array): C matrix of localized occupied ennironment MOs
        c_loc_occ_full (np.array): full C matrix of localized occupied MOs
        dm_active (np.array): active system density matrix
        dm_enviro (np.array): environment system density matrix
        active_MO_inds (np.array): 1D array of active occupied MO indices
        enviro_MO_inds (np.array): 1D array of environment occupied MO indices
    """

    logger.info("Localising with SPADE.")
    n_occupied_orbitals = np.count_nonzero(scf_method.mo_occ == 2)
    occupied_orbitals = scf_method.mo_coeff[:, :n_occupied_orbitals]

    n_act_aos = scf_method.mol.aoslice_by_atom()[n_active_atoms - 1][-1]
    logger.debug(f"{n_act_aos} active AOs.")

    ao_overlap = scf_method.get_ovlp()

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
    enviro_MO_inds = np.arange(n_act_mos, n_act_mos+n_env_mos)

    # Defining active and environment orbitals and density
    c_active = occupied_orbitals @ right_vectors.T[:, :n_act_mos]
    c_enviro = occupied_orbitals @ right_vectors.T[:, n_act_mos:]
    dm_active = 2.0 * c_active @ c_active.T
    dm_enviro = 2.0 * c_enviro @ c_enviro.T

    c_loc_occ_full = occupied_orbitals @ right_vectors.T

    return c_active, c_enviro, c_loc_occ_full, dm_active, dm_enviro, active_MO_inds, enviro_MO_inds


def pyscf_localization(pyscf_scf: gto.Mole,
                       localization_method: str) -> Tuple[np.ndarray]:
    """
    Localise orbitals using PySCF localization schemes.

    Args:
        pyscf_scf (gto.Mole): PySCF SCF mol object
        localization_method (str): String of orbital localization method (pipekmezey, boys, ibo)

    Returns:
        c_loc_occ (np.array): C matrix of OCCUPIED localized molecular orbitals (defined by columns of matrix)

    """
    if pyscf_scf.mo_coeff is None:
        raise ValueError('SCF calculation has not been performed. No optimized C_matrix')

    n_occupied_orbitals = np.count_nonzero(pyscf_scf.mo_occ == 2)
    c_std_occ = pyscf_scf.mo_coeff[:, :n_occupied_orbitals]

    if localization_method.lower() == 'pipekmezey':
        # Localise orbitals using Pipek-Mezey localization scheme.
        # This maximizes the sum of orbital-dependent partial charges on the nuclei.

        pipmez = lo.PipekMezey(pyscf_scf.mol, c_std_occ)

        # The atomic population projection scheme.
        # 'mulliken' 'meta-lowdin', 'iao', 'becke'
        pipmez.pop_method = 'meta-lowdin'

        # run localization
        c_loc_occ = pipmez.kernel()

    elif localization_method.lower() == 'boys':
        #  Boy localization method minimizes the spatial extent of the orbitals by minimizing a certain function
        boys_SCF = lo.boys.Boys(pyscf_scf.mol, c_std_occ)
        c_loc_occ = boys_SCF.kernel()

    elif localization_method.lower() == 'ibo':
        # intrinsic bonding orbitals
        iaos = lo.iao.iao(pyscf_scf.mol, c_std_occ)
        # Orthogonalize IAO
        iaos = lo.vec_lowdin(iaos, pyscf_scf.get_ovlp())
        c_loc_occ = lo.ibo.ibo(pyscf_scf.mol, c_std_occ, locmethod='IBO', iaos=iaos)
    else:
        raise ValueError(f'unknown localization method {localization_method}')

    return c_loc_occ


def localize_molecular_orbs(pyscf_scf: gto.Mole, n_active_atoms: int,
                            localization_method: str, occ_THRESHOLD: float = 0.95, virt_THRESHOLD: float = 0.95,
                            sanity_check: bool = False, run_virtual_localization: bool = False) \
        -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
    """
    Localise molecular orbitals (MOs) using different localization schemes.
    Funtion returns active and environment systems.


    Note:
    The major improvement of IBOs over PM orbitals is that they are based on IAO charges instead of the erratic
    Mulliken charges. As a result, IBOs are always well-defined.  (Ref: J. Chem. Theory Comput. 2013, 9, 4834−4843)

    Args:
        pyscf_scf (gto.Mole): PySCF molecule object
        n_active_atoms (int): Number of active atoms
        localization_method (str): String of orbital localization method (spade, pipekmezey, boys, ibo)
        occ_THRESHOLD (float): Threshold for selecting occupied active region (only requried if
                                spade localization is NOT used)
        virt_THRESHOLD (float): Threshold for selecting unoccupied (virtual) active region (required for
                                spade approach too!)
        sanity_check (bool): optional flag to check denisty matrices and electron number after orbital localization
                             makes sense
        run_virtual_localization (bool): optional flag on whether to perform localization of virtual orbitals.
                                         Note if False appends cannonical virtual orbs to C_loc_occ_and_virt matrix

    Returns:
        c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
        c_enviro (np.array): C matrix of localized occupied ennironment MOs
        c_loc_occ_full (np.array): full C matrix of localized occupied MOs
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
    if localization_method.lower() == 'spade':
        c_active, c_enviro, c_loc_occ_full, dm_active, dm_enviro, active_MO_inds, enviro_MO_inds = spade(pyscf_scf,
                                                                                                         n_active_atoms
                                                                                                         )
    else:
        c_loc_occ_full = pyscf_localization(pyscf_scf, localization_method)
        
        ao_slice_matrix = pyscf_scf.mol.aoslice_by_atom()

        # TODO: Check the following:
        # S_ovlp = pyscf_scf.get_ovlp()
        # S_half = sp.linalg.fractional_matrix_power(S_ovlp , 0.5)
        # C_loc_occ_ORTHO = S_half@C_loc_occ_full 
        # run numerator_all and denominator_all in ortho basis

        # find indices of AO of active atoms
        ao_active_inds = np.arange(ao_slice_matrix[0, 2], ao_slice_matrix[n_active_atoms-1, 3])
        # active AOs coeffs for a given MO j
        numerator_all = np.einsum('ij->j', (c_loc_occ_full[ao_active_inds, :])**2)

        # all AOs coeffs for a given MO j
        denominator_all = np.einsum('ij->j', c_loc_occ_full**2)

        MO_active_percentage = numerator_all/denominator_all

        logger.debug(f"(active_AO^2)/(all_AO^2): {np.around(MO_active_percentage,4)}")
        logger.debug(f"threshold for active part: {occ_THRESHOLD}")

        active_MO_inds = np.where(MO_active_percentage > occ_THRESHOLD)[0]
        enviro_MO_inds = np.array([i for i in range(c_loc_occ_full.shape[1]) if i not in active_MO_inds])

        # define active MO orbs and environment
        #    take MO (columns of C_matrix) that have high dependence from active AOs
        c_active = c_loc_occ_full[:, active_MO_inds]
        c_enviro = c_loc_occ_full[:, enviro_MO_inds]

        n_act_mos = len(active_MO_inds)
        n_env_mos = len(enviro_MO_inds)

        logger.debug(f"{n_act_mos} active MOs.")
        logger.debug(f"{n_env_mos} environment MOs.")

        dm_active = 2.0 * c_active @ c_active.T
        dm_enviro = 2.0 * c_enviro @ c_enviro.T

    if sanity_check is True:
        # checking denisty matrix parition makes sense:
        dm_localised_full_system = 2 * c_loc_occ_full @ c_loc_occ_full.conj().T
        bool_density_flag = np.allclose(dm_localised_full_system, dm_active + dm_enviro)
        logger.debug(f'y_active + y_enviro = y_total is: {bool_density_flag}')
        if not bool_density_flag:
            raise ValueError('gamma_full != gamma_active + gamma_enviro')

        # check number of electrons is still the same after orbitals have been localized (change of basis)
        s_ovlp = pyscf_scf.get_ovlp()
        n_active_electrons = np.trace(dm_active@s_ovlp)
        n_enviro_electrons = np.trace(dm_enviro@s_ovlp)
        n_all_electrons = pyscf_scf.mol.nelectron
        bool_flag_electron_number = np.isclose((n_active_electrons + n_enviro_electrons), n_all_electrons)
        logger.debug(f'N_active_elec + N_environment_elec = N_total_elec is: {bool_flag_electron_number}')
        if not bool_flag_electron_number:
            raise ValueError('number of electrons in localized orbitals is incorrect')

    if run_virtual_localization is True:
        (c_virtual_loc, active_virtual_MO_inds,
         enviro_virtual_MO_inds) = localize_virtual_orbs(pyscf_scf,
                                                         n_active_atoms,
                                                         virt_THRESHOLD=virt_THRESHOLD)

        c_loc_occ_and_virt = np.hstack((c_loc_occ_full,
                                        c_virtual_loc))
    else:
        # appends standard virtual orbitals from SCF calculation (NOT localized in any way)
        active_virtual_MO_inds, enviro_virtual_MO_inds = None, None
        c_std_virtual = pyscf_scf.mo_coeff[:, pyscf_scf.mo_occ < 2]
        c_loc_occ_and_virt = np.hstack((c_loc_occ_full,
                                        c_std_virtual))

    return (c_active, c_enviro, c_loc_occ_full, dm_active, dm_enviro, active_MO_inds, enviro_MO_inds,
            c_loc_occ_and_virt, active_virtual_MO_inds, enviro_virtual_MO_inds)


def localize_virtual_orbs(pyscf_scf: gto.Mole, n_active_atoms: int, virt_THRESHOLD: float = 0.95) \
                           -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Localise virtual (unoccupied) orbitals using different localization schemes in PySCF


    Args:
        pyscf_scf (gto.Mole): PySCF molecule object
        n_active_atoms (int): Number of active atoms
        virt_THRESHOLD (float): Threshold for selecting unoccupied (virtual) active regio

    Returns:
        c_virtual_loc (np.array): C matrix of localized virtual MOs (columns define MOs)
        active_virtual_MO_inds (np.array): 1D array of active virtual MO indices
        enviro_virtual_MO_inds (np.array): 1D array of environment virtual MO indices

    """
    if pyscf_scf.mo_coeff is None:
        raise ValueError('SCF calculation has not been performed. No optimized C_matrix')

    n_occupied_orbitals = np.count_nonzero(pyscf_scf.mo_occ == 2)
    c_std_occ = pyscf_scf.mo_coeff[:, :n_occupied_orbitals]
    c_std_virt = pyscf_scf.mo_coeff[:, pyscf_scf.mo_occ < 2]

    c_virtual_loc = vvo.vvo(pyscf_scf.mol,
                            c_std_occ,
                            c_std_virt,
                            iaos=None,
                            s=None,
                            verbose=None)

    ao_slice_matrix = pyscf_scf.mol.aoslice_by_atom()

    # TODO: Check the following:
    # S_ovlp = pyscf_scf.get_ovlp()
    # S_half = sp.linalg.fractional_matrix_power(S_ovlp , 0.5)
    # C_loc_occ_ORTHO = S_half@C_loc_occ_full 
    # run numerator_all and denominator_all in ortho basis

    # find indices of AO of active atoms
    ao_active_inds = np.arange(ao_slice_matrix[0, 2], ao_slice_matrix[n_active_atoms-1, 3])

    # active AOs coeffs for a given MO j
    numerator_all = np.einsum('ij->j', (c_virtual_loc[ao_active_inds, :])**2)
    # all AOs coeffs for a given MO j
    denominator_all = np.einsum('ij->j', c_virtual_loc**2)

    active_percentage_MO = numerator_all/denominator_all

    logger.debug(f"(active_AO^2)/(all_AO^2): {np.around(active_percentage_MO,4)}")
    logger.debug(f"threshold for active part: {virt_THRESHOLD}")

    # add constant occupied index 
    active_virtual_MO_inds = np.where(active_percentage_MO > virt_THRESHOLD)[0] + c_std_occ.shape[1]
    enviro_virtual_MO_inds = np.array([i for i in range(c_std_occ.shape[1],
                                                        c_std_occ.shape[1]+c_virtual_loc.shape[1])
                                       if i not in active_virtual_MO_inds])

    return c_virtual_loc, active_virtual_MO_inds, enviro_virtual_MO_inds


def orb_change_basis_operator(pyscf_scf: gto.Mole,
                              c_all_localized_and_virt: np.array,
                              sanity_check=False) -> np.ndarray:

    """
    Get operator that changes from standard cannoncial orbitals (C_matrix standard) to
    Localized orbitals (C_matrix_localized)

    Args:
        pyscf_scf (gto.Mole): PySCF molecule object
        c_all_localized_and_virt (np.array): C_matrix of localized orbitals (includes occupied and virtual)
        sanity_check (bool): optional flag to check if change of basis is working properly

    Returns:
        matrix_std_to_loc (np.array): Matrix that maps from standard (cannonical) MOs to localized MOs
    """

    s_mat = pyscf_scf.get_ovlp()
    s_half = sp.linalg.fractional_matrix_power(s_mat, 0.5)

    # find orthogonal orbitals
    ortho_std = s_half @ pyscf_scf.mo_coeff
    ortho_loc = s_half @ c_all_localized_and_virt

    # Build change of basis operator (maps between orthonormal basis (cannoncial and localized)
    unitary_ORTHO_std_onto_loc = np.einsum('ik,jk->ij', ortho_std, ortho_loc)

    if sanity_check:
        if np.allclose(unitary_ORTHO_std_onto_loc @ ortho_loc, ortho_std) is not True:
            raise ValueError('Change of basis incorrect... U_ORTHO_std_onto_loc*C_ortho_loc !=  C_ortho_STD')

        if np.allclose(unitary_ORTHO_std_onto_loc.conj().T@unitary_ORTHO_std_onto_loc,
                       np.eye(unitary_ORTHO_std_onto_loc.shape[0])) is not True:
            raise ValueError('Change of basis (U_ORTHO_std_onto_loc) is not Unitary!')

    s_neg_half = sp.linalg.fractional_matrix_power(s_mat, -0.5)

    # move back into non orthogonal basis
    matrix_std_to_loc = s_neg_half @ unitary_ORTHO_std_onto_loc @ s_half

    if sanity_check:
        if np.allclose(matrix_std_to_loc@c_all_localized_and_virt,
                       pyscf_scf.mo_coeff) is not True:
            raise ValueError('Change of basis incorrect... U_std*C_std !=  C_loc_occ_and_virt')

    return matrix_std_to_loc
