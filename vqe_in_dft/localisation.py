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


def spade(
    scf_method: Callable, N_active_atoms: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Localise orbitals using SPADE.
    """
    logger.info("Localising with SPADE.")
    n_occupied_orbitals = np.count_nonzero(scf_method.mo_occ == 2)
    occupied_orbitals = scf_method.mo_coeff[:, :n_occupied_orbitals]

    n_act_aos = scf_method.mol.aoslice_by_atom()[N_active_atoms - 1][-1]
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
    active_MO_inds  = np.arange(n_act_mos)
    enviro_MO_inds = np.arange(n_act_mos, n_act_mos+n_env_mos)

    # Defining active and environment orbitals and density
    C_active = occupied_orbitals @ right_vectors.T[:, :n_act_mos]
    C_enviro = occupied_orbitals @ right_vectors.T[:, n_act_mos:]
    dm_active = 2.0 * C_active @ C_active.T
    dm_enviro = 2.0 * C_enviro @ C_enviro.T

    C_loc_occ_full = occupied_orbitals @ right_vectors.T


    return C_active, C_enviro, C_loc_occ_full, dm_active, dm_enviro, active_MO_inds, enviro_MO_inds


def PySCF_localization(PySCF_scf_obj: gto.Mole,
                localization_method:str) -> Tuple[np.ndarray]:
    """
    Localise orbitals using PySCF localization schemes.

    Args:
        PySCF_scf_obj (gto.Mole)
        localization_method (str): String of orbital localization method (pipekmezey, boys, ibo)

    Returns:
        C_loc_occ (np.array): C matrix of OCCUPIED localized molecular orbitals (defined by columns of matrix)

    """
    if PySCF_scf_obj.mo_coeff is None:
        raise ValueError('SCF calculation has not been performed. No optimized C_matrix')

    n_occupied_orbitals = np.count_nonzero(PySCF_scf_obj.mo_occ == 2)
    C_std_occ = PySCF_scf_obj.mo_coeff[:, :n_occupied_orbitals]

    if localization_method.lower() == 'pipekmezey':
        # Localise orbitals using Pipek-Mezey localization scheme.
        # This maximizes the sum of orbital-dependent partial charges on the nuclei.

        PM = lo.PipekMezey(PySCF_scf_obj.mol, C_std_occ)
        # The atomic population projection scheme.
        PM.pop_method = 'meta-lowdin' #'mulliken' 'meta-lowdin', 'iao', 'becke'
        C_loc_occ = PM.kernel() # run localization

    elif localization_method.lower() == 'boys':
        #  Boy localization method minimizes the spatial extent of the orbitals by minimizing a certain function
        boys_SCF = lo.boys.Boys(PySCF_scf_obj.mol, C_std_occ)
        C_loc_occ  = boys_SCF.kernel()

    elif localization_method.lower() == 'ibo':
        # intrinsic bonding orbitals
        iaos = lo.iao.iao(PySCF_scf_obj.mol, C_std_occ)
        # Orthogonalize IAO
        iaos = lo.vec_lowdin(iaos, PySCF_scf_obj.get_ovlp())
        C_loc_occ = lo.ibo.ibo(PySCF_scf_obj.mol, C_std_occ, locmethod='IBO', iaos=iaos)#.kernel()
    else:
        raise ValueError(f'unknown localization method {localization_method}')

    return C_loc_occ




def Localize_orbital_orbs(PySCF_scf_obj: gto.Mole, N_active_atoms: int,
                localization_method: str, occ_THRESHOLD: float = 0.95, virt_THRESHOLD: float = 0.95,
                 sanity_check: bool=False, run_virtual_localization: bool=False
                 ) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
    """
    Localise orbitals using different localization schemes. Funtion returns active and environment systems


    Note:
    The major improvement of IBOs over PM orbitals is that they are based on IAO charges instead of the erratic Mulliken charges
    As a result, IBOs are always well-defined.  (Ref: J. Chem. Theory Comput. 2013, 9, 4834−4843)

    Args:
        PySCF_scf_obj (gto.Mole): PySCF molecule object
        N_active_atoms (int): Number of active atoms
        localization_method (str): String of orbital localization method (spade, pipekmezey, boys, ibo)
        occ_THRESHOLD (float): Threshold for selecting occupied active region (only requried if spade localization is NOT used)
        virt_THRESHOLD (float): Threshold for selecting unoccupied (virtual) active region (required for spade approach too!)
        sanity_check (bool): optional flag to check denisty matrices and electron number after orbital localization makes sense
        run_virtual_localization (bool): optional flag on whether to perform localization of virtual orbitals. Note if False appends cannonical virtual orbs to C_loc_occ_and_virt matrix

    Returns:
        C_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
        C_enviro (np.array): C matrix of localized occupied ennironment MOs  
        C_loc_occ_full (np.array): full C matrix of localized occupied MOs  
        dm_active (np.array): active system density matrix
        dm_enviro (np.array): environment system density matrix
        active_MO_inds (np.array): 1D array of active occupied MO indices
        enviro_MO_inds (np.array): 1D array of environment occupied MO indices
        C_loc_occ_and_virt (np.array): Full localized C_matrix (occpuied and virtual)
        active_virtual_MO_inds (np.array): 1D array of active virtual MO indices (set to None if run_virtual_localization is False)
        enviro_virtual_MO_inds (np.array): 1D array of environment virtual MO indices (set to None if run_virtual_localization is False)

    """
    if localization_method.lower() == 'spade':
        C_active, C_enviro, C_loc_occ_full, dm_active, dm_enviro, active_MO_inds, enviro_MO_inds = spade(PySCF_scf_obj,
                                                                                                            N_active_atoms
                                                                                                            )
    else:
        C_loc_occ_full = PySCF_localization(PySCF_scf_obj, localization_method)
        
        AO_slice_matrix = PySCF_scf_obj.mol.aoslice_by_atom()

        # TODO: Check the following:
        # S_ovlp = PySCF_scf_obj.get_ovlp()
        # S_half = sp.linalg.fractional_matrix_power(S_ovlp , 0.5)
        # C_loc_occ_ORTHO = S_half@C_loc_occ_full 
        # run numerator_all and denominator_all in ortho basis

        # find indices of AO of active atoms
        ao_active_inds = np.arange(AO_slice_matrix[0,2], AO_slice_matrix[N_active_atoms-1,3])
        
        numerator_all = np.einsum('ij->j', (C_loc_occ_full[ao_active_inds, :])**2) # active AOs coeffs for a given MO j
        denominator_all = np.einsum('ij->j', C_loc_occ_full**2) # all AOs coeffs for a given MO j

        MO_active_percentage = numerator_all/denominator_all

        logger.debug(f"(active_AO^2)/(all_AO^2): {np.around(MO_active_percentage,4)}")
        logger.debug(f"threshold for active part: {occ_THRESHOLD}")


        active_MO_inds = np.where(MO_active_percentage>occ_THRESHOLD)[0]
        enviro_MO_inds = np.array([i for i in range(C_loc_occ_full.shape[1]) if i not in active_MO_inds]) # get all non active MOs

        # define active MO orbs and environment
        C_active = C_loc_occ_full[:, active_MO_inds] # take MO (columns of C_matrix) that have high dependence from active AOs
        C_enviro = C_loc_occ_full[:, enviro_MO_inds]

        n_act_mos = len(active_MO_inds)
        n_env_mos = len(enviro_MO_inds)

        logger.debug(f"{n_act_mos} active MOs.")
        logger.debug(f"{n_env_mos} environment MOs.")

        dm_active = 2.0 * C_active @ C_active.T
        dm_enviro = 2.0 * C_enviro @ C_enviro.T


    if sanity_check is True:
        # checking denisty matrix parition makes sense:
        dm_localised_full_system = 2* C_loc_occ_full@ C_loc_occ_full.conj().T
        bool_density_flag = np.allclose(dm_localised_full_system, dm_active + dm_enviro)
        logger.debug(f'y_active + y_enviro = y_total is: {bool_density_flag}')
        if not bool_density_flag:
            raise ValueError('gamma_full != gamma_active + gamma_enviro')

        ## check number of electrons is still the same after orbitals have been localized (change of basis)
        S_ovlp = PySCF_scf_obj.get_ovlp()
        N_active_electrons = np.trace(dm_active@S_ovlp)
        N_enviro_electrons = np.trace(dm_enviro@S_ovlp)
        N_all_electrons = PySCF_scf_obj.mol.nelectron
        bool_flag_electron_number = np.isclose(( N_active_electrons + N_enviro_electrons), N_all_electrons)
        logger.debug(f'N_active_elec + N_environment_elec = N_total_elec is: {bool_flag_electron_number}')
        if not bool_flag_electron_number:
            raise ValueError('number of electrons in localized orbitals is incorrect')


    if run_virtual_localization is True:
        C_virtual_loc, active_virtual_MO_inds, enviro_virtual_MO_inds = Localize_virtual_orbs(PySCF_scf_obj, N_active_atoms, virt_THRESHOLD= virt_THRESHOLD)

        C_loc_occ_and_virt = np.hstack((C_loc_occ_full,
                                     C_virtual_loc))
    else:
        # appends standard virtual orbitals from SCF calculation (NOT localized in any way)
        active_virtual_MO_inds, enviro_virtual_MO_inds = None, None
        C_std_virtual = PySCF_scf_obj.mo_coeff[:,PySCF_scf_obj.mo_occ<2]
        C_loc_occ_and_virt = np.hstack((C_loc_occ_full,
                             C_std_virtual))

    return C_active, C_enviro, C_loc_occ_full, dm_active, dm_enviro, active_MO_inds, enviro_MO_inds, C_loc_occ_and_virt, active_virtual_MO_inds, enviro_virtual_MO_inds


def Localize_virtual_orbs(PySCF_scf_obj: gto.Mole, N_active_atoms: int,
                          virt_THRESHOLD:float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Localise virtual (unoccupied) orbitals using different localization schemes in PySCF


    Args:
        PySCF_scf_obj (gto.Mole): PySCF molecule object
        N_active_atoms (int): Number of active atoms
        virt_THRESHOLD (float): Threshold for selecting unoccupied (virtual) active regio

    Returns:
        C_virtual_loc (np.array): C matrix of localized virtual MOs (columns define MOs)
        active_virtual_MO_inds (np.array): 1D array of active virtual MO indices
        enviro_virtual_MO_inds (np.array): 1D array of environment virtual MO indices

    """
    if PySCF_scf_obj.mo_coeff is None:
        raise ValueError('SCF calculation has not been performed. No optimized C_matrix')

    n_occupied_orbitals = np.count_nonzero(PySCF_scf_obj.mo_occ == 2)
    C_std_occ = PySCF_scf_obj.mo_coeff[:, :n_occupied_orbitals]
    C_std_virt = PySCF_scf_obj.mo_coeff[:,PySCF_scf_obj.mo_occ<2]

    C_virtual_loc = vvo.vvo(PySCF_scf_obj.mol,
                        C_std_occ,
                        C_std_virt,
                        iaos=None, 
                        s=None,
                        verbose=None)


    S_ovlp = PySCF_scf_obj.get_ovlp()
    AO_slice_matrix = PySCF_scf_obj.mol.aoslice_by_atom()

    # TODO: Check the following:
    # S_half = sp.linalg.fractional_matrix_power(S_ovlp , 0.5)
    # C_loc_occ_ORTHO = S_half@C_loc_occ_full 
    # run numerator_all and denominator_all in ortho basis

    # find indices of AO of active atoms
    ao_active_inds = np.arange(AO_slice_matrix[0,2], AO_slice_matrix[N_active_atoms-1,3])
    
    numerator_all = np.einsum('ij->j', (C_virtual_loc[ao_active_inds, :])**2) # active AOs coeffs for a given MO j
    denominator_all = np.einsum('ij->j', C_virtual_loc**2) # all AOs coeffs for a given MO j

    MO_active_percentage = numerator_all/denominator_all

    logger.debug(f"(active_AO^2)/(all_AO^2): {np.around(MO_active_percentage,4)}")
    logger.debug(f"threshold for active part: {virt_THRESHOLD}")

    # add constant occupied index 
    active_virtual_MO_inds = np.where(MO_active_percentage>virt_THRESHOLD)[0] + C_std_occ.shape[1]
    enviro_virtual_MO_inds = np.array([i for i in range(C_std_occ.shape[1], C_std_occ.shape[1]+C_virtual_loc.shape[1]) if i not in active_virtual_MO_inds]) 

    return C_virtual_loc, active_virtual_MO_inds, enviro_virtual_MO_inds


def Get_orb_change_basis_operator(PySCF_scf_obj: gto.Mole, N_active_atoms: int,
                                  C_all_localized_and_virt: np.array,
                                  sanity_check=False) -> np.ndarray:

    """
    Localise virtual (unoccupied) orbitals using different localization scheme.
    This maximizes the sum of orbital-dependent partial charges on the nuclei.

    Args:
        PySCF_scf_obj (gto.Mole): PySCF molecule object
        N_active_atoms (int): Number of active atoms
        C_all_localized_and_virt (np.array): C_matrix of localized orbitals (includes occupied and virtual)
        sanity_check (bool): optional flag to check if change of basis is working properly

    Returns:
        U_std (np.array): Matrix that maps from standard (cannonical) MOs to localized MOs 
    """

    S_mat = PySCF_scf_obj.get_ovlp()
    S_half = sp.linalg.fractional_matrix_power(PySCF_scf_obj.get_ovlp() , 0.5)

    ortho_std = S_half@ PySCF_scf_obj.mo_coeff
    ortho_loc = S_half@ C_all_localized_and_virt

    # ortho_loc[:,0].dot(ortho_loc[:,2])

    U_ORTHO_std_onto_loc = np.einsum('ik,jk->ij', ortho_std,ortho_loc)


    if sanity_check:
        if np.allclose(U_ORTHO_std_onto_loc @ ortho_loc, ortho_std) is not True:
            raise ValueError('Change of basis incorrect... U_ORTHO_std_onto_loc*C_ortho_loc !=  C_ortho_STD')

        if np.allclose(U_ORTHO_std_onto_loc.conj().T@U_ORTHO_std_onto_loc, np.eye(U_ORTHO_std_onto_loc.shape[0])) is not True:
            raise ValueError('Change of basis (U_ORTHO_std_onto_loc) is not Unitary!')


    S_neg_half = sp.linalg.fractional_matrix_power(PySCF_scf_obj.get_ovlp() , -0.5)
    U_std = S_neg_half @ U_ORTHO_std_onto_loc @ S_half # move back into non orthogonal basis

    if sanity_check:
        if np.allclose(U_std@C_all_localized_and_virt,
                      PySCF_scf_obj.mo_coeff) is not True:
            raise ValueError('Change of basis incorrect... U_std*C_std !=  C_loc_occ_and_virt')

    return U_std


