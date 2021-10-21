"""Main embedding functionality."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy as sp
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import jordan_wigner
from pyscf import ao2mo, cc, gto, lib, scf, fci
from pyscf.dft import numint
from pyscf.dft.rks import get_veff as rks_get_veff
from pyscf.lib import StreamObject
from nbed.ham_converter import HamiltonianConverter
from nbed.utils import parse, setup_logs
from nbed.localisation import localize_molecular_orbs, orb_change_basis_operator
import warnings

logger = logging.getLogger(__name__)
setup_logs()

def get_Hcore_new_basis(h_core: np.array, unitary_rot: np.ndarray) -> np.array:
    """
    Function to get H_core in new basis

    Args:
        h_core (np.ndarray): standard core Hamiltonian
        unitary_rot (np.ndarray): Operator to change basis  (in this code base this should be: cannonical basis to
        localized basis)
    Returns:
        H_core_rot (np.ndarray): core Hamiltonian in new basis
    """
    H_core_rot = unitary_rot.conj().T @ h_core @ unitary_rot
    return H_core_rot


def get_new_RKS_Veff(
    pyscf_RKS: StreamObject,
    unitary_rot: np.ndarray,
    dm=None,
    check_result: bool = False,
) -> lib.tag_array:
    """
    Function to get V_eff in new basis.  Note this function is based on: pyscf.dft.rks.get_veff

    Note in RKS calculation Veff = J + Vxc
    Whereas for RHF calc it is Veff = J - 0.5k

    Args:
        pyscf_RKS (StreamObject): PySCF RKS obj
        unitary_rot (np.ndarray): Operator to change basis  (in this code base this should be: cannonical basis
                                to localized basis)
        dm (np.ndarray): Optional input density matrix. If not defined, finds whatever is available from pyscf_RKS_obj
        check_result (bool): Flag to check result against PySCF functions

    Returns:
        output (lib.tag_array): Tagged array containing J, K, E_coloumb, E_xcorr, Vxc
    """
    if dm is None:
        if pyscf_RKS.mo_coeff is not None:
            dm = pyscf_RKS.make_rdm1(pyscf_RKS.mo_coeff, pyscf_RKS.mo_occ)
        else:
            dm = pyscf_RKS.init_guess_by_1e()

    # Evaluate RKS/UKS XC functional and potential matrix on given meshgrids
    # for a set of density matrices.
    nelec, exc, vxc = numint.nr_vxc(pyscf_RKS.mol, pyscf_RKS.grids, pyscf_RKS.xc, dm)

    # definition in new basis
    vxc = unitary_rot.conj().T @ vxc @ unitary_rot

    v_eff = rks_get_veff(pyscf_RKS, dm=dm)
    if v_eff.vk is not None:
        k_mat = unitary_rot.conj().T @ v_eff.vk @ unitary_rot
        j_mat = unitary_rot.conj().T @ v_eff.vj @ unitary_rot
        vxc += j_mat - k_mat * 0.5
    else:
        j_mat = unitary_rot.conj().T @ v_eff.vj @ unitary_rot
        k_mat = None
        vxc += j_mat

    if check_result is True:
        veff_check = unitary_rot.conj().T @ v_eff.__array__() @ unitary_rot
        if not np.allclose(vxc, veff_check):
            raise ValueError("Veff in new basis does not match rotated PySCF value.")

    # note J matrix is in new basis!
    ecoul = np.einsum("ij,ji", dm, j_mat).real * 0.5
    # this ecoul term changes if the full density matrix is NOT
    #    (aka for dm_active and dm_enviroment we get different V_eff under different bases!)

    output = lib.tag_array(vxc, ecoul=ecoul, exc=v_eff.exc, vj=j_mat, vk=k_mat)
    return output


def calc_RKS_components_from_dm(
    pyscf_RKS: StreamObject, dm_matrix: np.ndarray, check_E_with_pyscf: bool = True
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the components of subsystem energy from a RKS DFT calculation.

    For a given density matrix this function returns the electronic energy, exchange correlation energy and
    J,K, V_xc matrices.

    Args:
        pyscf_RKS (StreamObject): PySCF RKS object
        dm_matrix (np.ndarray): density matrix (to calculate all matrices from)
        check_E_with_pyscf (bool): optional flag to check manual energy calc against PySCF calc
    Returns:
        Energy_elec (float): DFT energy defubed by input density matrix
        e_xc (float): exchange correlation energy defined by input density matrix
        J_mat (np.ndarray): J_matrix defined by input density matrix
        K_mat (np.ndarray): K_matrix defined by input density matrix
        v_xc (np.ndarray): V_exchangeCorrelation matrix defined by input density matrix (note Coloumbic
                         contribution (J_mat) has been subtracted to give this term)
    """

    # It seems that PySCF lumps J and K in the J array
    two_e_term = pyscf_RKS.get_veff(dm=dm_matrix)
    j_mat = two_e_term.vj
    k_mat = np.zeros_like(j_mat)

    e_xc = two_e_term.exc
    v_xc = two_e_term - j_mat

    energy_elec = (
        np.einsum("ij,ji->", pyscf_RKS.get_hcore(), dm_matrix)
        + two_e_term.ecoul
        + two_e_term.exc
    )

    if check_E_with_pyscf:
        energy_elec_pyscf = pyscf_RKS.energy_elec(dm=dm_matrix)[0]
        if not np.isclose(energy_elec_pyscf, energy_elec):
            raise ValueError("Energy calculation incorrect")

    return energy_elec, e_xc, j_mat, k_mat, v_xc


def get_new_RHF_Veff(
    pyscf_RHF: StreamObject, unitary_rot: np.ndarray, dm=None, hermi: int = 1
) -> np.ndarray:
    """
    Function to get V_eff in new basis.

    Note in RKS calculation Veff = J + Vxc
    Whereas for RHF calc it is Veff = J - 0.5k

    Args:
        pyscf_RHF (StreamObject): PySCF RHF obj
        unitary_rot (np.ndarray): Operator to change basis  (in this code base this should be: cannonical basis
                                to localized basis)
        dm (np.ndarray): Optional input density matrix. If not defined, finds whatever is available from pyscf_RKS_obj
        hermi (int): TODO
    """
    if dm is None:
        if pyscf_RHF.mo_coeff is not None:
            dm = pyscf_RHF.make_rdm1(pyscf_RHF.mo_coeff, pyscf_RHF.mo_occ)
        else:
            dm = pyscf_RHF.init_guess_by_1e()

    # if pyscf_RHF._eri is None:
    #     pyscf_RHF._eri = pyscf_RHF.mol.intor('int2e', aosym='s8')

    vj, vk = pyscf_RHF.get_jk(mol=pyscf_RHF.mol, dm=dm, hermi=hermi)
    v_eff = vj - vk * 0.5

    # v_eff = pyscf_obj.get_veff(dm=dm)
    v_eff_new = unitary_rot.conj().T @ v_eff @ unitary_rot

    return v_eff_new

# from pyscf.scf.hf import get_jk as get_jk_HFock
# def get_new_RHF_Veff(
#     pyscf_RHF: StreamObject, unitary_rot: np.ndarray, dm=None, hermi: int = 1
# ) -> np.ndarray:
#     """
#     Function to get V_eff in new basis.
#
#     Note in RKS calculation Veff = J + Vxc
#     Whereas for RHF calc it is Veff = J - 0.5k
#
#     Args:
#         pyscf_RHF (StreamObject): PySCF RHF obj
#         unitary_rot (np.ndarray): Operator to change basis  (in this code base this should be: cannonical basis
#                                 to localized basis)
#         dm (np.ndarray): Optional input density matrix. If not defined, finds whatever is available from pyscf_RKS_obj
#         hermi (int): Whether J, K matrix is hermitian
#                         0: not hermitian and not symmetric
#                         1: hermitian or symmetric
#                         2: anti-hermitian
#     """
#     if dm is None:
#         if pyscf_RHF.mo_coeff is not None:
#             dm = pyscf_RHF.make_rdm1(pyscf_RHF.mo_coeff, pyscf_RHF.mo_occ)
#         else:
#             dm = pyscf_RHF.init_guess_by_1e()
#
#     vj, vk = get_jk_HFock(pyscf_RHF.mol, dm, hermi=hermi, vhfopt=None, with_j=True, with_k=True, omega=None)
#     # vj, vk = pyscf_RHF.get_jk(dm=dm, hermi=hermi)
#     v_eff = vj - vk * 0.5
#
#     # v_eff = pyscf_obj.get_veff(dm=dm)
#     v_eff_new = unitary_rot.conj().T @ v_eff @ unitary_rot
#
#     return v_eff_new

def get_cross_terms_DFT(
    pyscf_RKS: StreamObject,
    dm_active: np.ndarray,
    dm_enviro: np.ndarray,
    j_env: np.ndarray,
    j_act: np.ndarray,
    e_xc_act: float,
    e_xc_env: float,
) -> float:
    """
    Get two electron cross term energy. As Veff = J + Vxc, need Colombic cross term energy (J_cross)
    and XC cross term energy

    Args:
        pyscf_RKS (StreamObject): PySCF RKS object
        dm_active (np.ndarray): density matrix of active subsystem
        dm_enviro (np.ndarray): density matrix of enironment subsystem
        j_env (np.ndarray): J_matrix defined by enviornemnt density
        j_act (np.ndarray): J_matrix defined by active density
        e_xc_act (float): exchange correlation energy defined by input active density matrix
        e_xc_env (float): exchange correlation energy defined by input enviornemnt density matrix

    Returns:
        two_e_cross (float): two electron energy from cross terms (includes exchange correlation
                             and Coloumb contribution)
    """
    two_e_term_total = pyscf_RKS.get_veff(dm=dm_active + dm_enviro)
    e_xc_total = two_e_term_total.exc

    j_cross = 0.5 * (
        np.einsum("ij,ij", dm_active, j_env) + np.einsum("ij,ij", dm_enviro, j_act)
    )
    k_cross = 0.0

    xc_cross = e_xc_total - e_xc_act - e_xc_env

    # overall two_electron cross energy
    two_e_cross = j_cross + k_cross + xc_cross

    return two_e_cross


def get_enivornment_projector(
    c_loc_occ_and_virt: np.ndarray,
    s_half: np.ndarray,
    active_MO_inds: np.ndarray,
    enviro_MO_inds: np.ndarray,
    return_in_ortho_basis: bool = False,
) -> np.ndarray:
    """ Get projector onto environement MOs

    P_env = Σ_{i ∈ env} |MO_i> <MO_i|

    Args:
        c_loc_occ_and_virt (np.ndarray): C_matrix of localized MO (virtual and occupied)
        s_half (np.ndarray): AO overlap matrix to the power of 1/2
        active_MO_inds (np.ndarray): 1D array of active MO indices
        enviro_MO_inds (np.ndarray): 1D array of enviornemnt MO indices
        return_in_ortho_basis (bool): Whether to return projector in orthogonal basis or standard basis

    Returns:
        projector (np.ndarray): Operator that projects environment MOs onto themselves and active MOs onto zero vector
    """
    # 1. Get orthogonal C matrix (localized)
    c_loc_ortho = s_half @ c_loc_occ_and_virt

    # 2. Define projector that projects MO orbs of subsystem B onto themselves and system A onto zero state!
    #    (do this in orthongoal basis!)
    #    note we only take MO environment indices!
    ortho_proj = np.einsum(
        "ik,jk->ij", c_loc_ortho[:, enviro_MO_inds], c_loc_ortho[:, enviro_MO_inds]
    )

    # env projected onto itself
    logger.info(
        f"""Are subsystem B (env) projected onto themselves in ORTHO basis: {
            np.allclose(ortho_proj @ c_loc_ortho[:, enviro_MO_inds],
            c_loc_ortho[:, enviro_MO_inds])}"""
    )

    # act projected onto zero vec
    logger.info(
        f"""Is subsystem A traced out  in ORTHO basis?: {
            np.allclose(ortho_proj @ c_loc_ortho[:, active_MO_inds],
            np.zeros_like(c_loc_ortho[:, active_MO_inds]))}"""
    )

    # 3. Define projector in standard (non-orthogonal basis)
    projector = s_half @ ortho_proj @ s_half

    logger.info(
        f"""Are subsystem B (env) projected onto themselves in ORTHO basis: {
            np.allclose(projector @ c_loc_occ_and_virt[:, enviro_MO_inds],
            c_loc_occ_and_virt[:, enviro_MO_inds])}"""
    )

    logger.info(
        f"""Is subsystem A traced out  in ORTHO basis?: {
            np.allclose(projector@c_loc_occ_and_virt[:, active_MO_inds],
            np.zeros_like(c_loc_occ_and_virt[:, active_MO_inds]))}"""
    )
    if return_in_ortho_basis:
        return ortho_proj
    else:
        return projector


def get_active_indices(
    scf_method: StreamObject,
    n_act_mos: int,
    n_env_mos: int,
    qubits: Optional[int] = None,
) -> np.ndarray:
    """Return an array of active indices for QHam construction.

    Args:
        scf_method (StreamObject): A pyscf self consisten method.
        n_act_mos (int): Number of active-space moleclar orbitals.
        n_env_mos (int): Number of environment moleclar orbitals.
        qubits (int): Number of qubits to be used in final calclation.

    Returns:
        np.ndarray: A 1D array of integer indices.
    """
    # Find the active indices
    active_indices = [i for i in range(len(scf_method.mo_occ) - n_env_mos)]

    # This is not the best way to simplify.
    # TODO some more sophisticated thing with frozen core
    # rather than just cutting high level MOs
    if qubits:
        # Check that the reduction is sensible
        # Needs 1 qubit per spin state
        if qubits < 2 * n_act_mos:
            raise Exception(f"Not enouch qubits for active MOs, minimum {2*n_act_mos}.")

        logger.info("Restricting to low level MOs for %s qubits.", qubits)
        active_indices = active_indices[: qubits // 2]

    return np.array(active_indices)


# def get_qubit_hamiltonian(
#     scf_method: StreamObject, active_indices: List[int]
# ) -> object:
#     """Return the qubit hamiltonian.
#
#     Args:
#         scf_method (StreamObject): A pyscf self-consistent method.
#         active_indices (list[int]): A list of integer indices of active moleclar orbitals.
#
#     Returns:
#         object: A qubit hamiltonian.
#     """
#     n_orbs = len(active_indices)
#
#     mo_coeff = scf_method.mo_coeff[:, active_indices]
#
#     one_body_integrals = mo_coeff.T @ scf_method.get_hcore() @ mo_coeff
#
#     # temp_scf.get_hcore = lambda *args, **kwargs : initial_h_core
#     scf_method.mol.incore_anyway is True
#
#     # Get two electron integrals in compressed format.
#     two_body_compressed = ao2mo.kernel(scf_method.mol, mo_coeff)
#
#     two_body_integrals = ao2mo.restore(
#         1, two_body_compressed, n_orbs  # no permutation symmetry
#     )
#
#     # Openfermion uses pysicist notation whereas pyscf uses chemists
#     two_body_integrals = np.asarray(two_body_integrals.transpose(0, 2, 3, 1), order="C")
#
#     one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
#         one_body_integrals, two_body_integrals
#     )
#
#     molecular_hamiltonian = InteractionOperator(
#         0, one_body_coefficients, 0.5 * two_body_coefficients
#     )
#
#     Qubit_Hamiltonian = jordan_wigner(molecular_hamiltonian)
#
#     return Qubit_Hamiltonian

from openfermion.ops.representations import get_active_space_integrals
def get_qubit_hamiltonian(
    scf_method: StreamObject,
) -> object:
    """Return the qubit hamiltonian.

    Args:
        scf_method (StreamObject): A pyscf self-consistent method.
        frozen_indices (list[int]): A list of integer indices of frozen moleclar orbitals.

    Returns:
        object: A qubit hamiltonian.
    """

    # C_matrix containing orbitals to be considered
    # if there are any environment orbs that have been projected out... these should NOT be present in the
    # scf_method.mo_coeff array (aka columns should be deleted!)
    c_matrix_active = scf_method.mo_coeff

    # one body terms
    one_body_integrals = c_matrix_active.T @ scf_method.get_hcore() @ c_matrix_active
    # one_body_integrals = c_matrix.T @ scf_method.get_hcore() @ c_matrix

    # two body terms
    # two_body_compressed = ao2mo.kernel(scf_method.mol,
    #                                    c_matrix)

    two_body_compressed = ao2mo.kernel(scf_method.mol,
                                       c_matrix_active)
    # get electron repulsion integrals
    eri = ao2mo.restore(
        1,
        two_body_compressed,
        n_orbs  # no permutation symmetry
    )

    # Openfermion uses pysicist notation whereas pyscf uses chemists
    two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order="C")

    core_constant, one_body_ints_reduced, two_body_ints_reduced = 0, one_body_integrals, two_body_integrals
    # core_constant, one_body_ints_reduced, two_body_ints_reduced = get_active_space_integrals(
    #                                                                                        one_body_integrals,
    #                                                                                        two_body_integrals,
    #                                                                                        occupied_indices=None,
    #                                                                                        active_indices=active_mo_inds
    #                                                                                         )

    print(f'core constant: {core_constant}')

    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
        one_body_ints_reduced, two_body_ints_reduced
    )

    molecular_hamiltonian = InteractionOperator(
        core_constant, one_body_coefficients, 0.5 * two_body_coefficients
    )

    Qubit_Hamiltonian = jordan_wigner(molecular_hamiltonian)

    return Qubit_Hamiltonian

def huzinaga_RHF(scf_method: StreamObject,
                 DFT_potential: np.ndarray,
                 enviro_proj_ortho_basis: np.ndarray,
                 s_neg_half: np.ndarray,
                 s_half: np.ndarray,
                 dm_conv_tol: float = 1e-6,
                 dm_initial_guess:Optional[np.ndarray] = None
                 ):
    """Manual RHF calculation that is implemented using the huzinaga operator

    Note this function uses lowdin (symmetric) orthogonalization only! (PySCF sometimes uses meta-lowdin and NAO). Also
    the intial density matrix guess is based on the modified core Hamilotnian (containing projector and DFT potential)
    PySCF has other methods for initial guess that aren't available here. Manual guess can also be given).

    TODO: make a subclass from PySCF RHF object. Can include all this functionality there. Problems in this function
    can occur due to DIIS and other clever PySCF methods not being available.

    Args:
        scf_method (StreamObjecty):PySCF RHF object (containing info about max cycles and convergence tolerence)
        DFT_potential (np.ndarray): DFT active and environment two body terms - DFT active environemnt two body term
        enviro_proj_ortho_basis (np.ndarray): Projector onto environment space (defined in orthogonal basis)
        s_neg_half (np.ndarray): AO overlap matrix to the power of -1/2
        s_half (np.ndarray): AO overlap matrix to the power of 1/2
        dm_conv_tol (float): density matrix convergence tolerance
        dm_initial_guess (np.ndarray): Optional initial guess density matrix
    Returns:
        conv_flag (bool): Flag to indicate whether SCF has converged or not
        e_total (float): RHF energy (includes nuclear energy)
        mo_coeff_std (np.ndarray): Optimized C_matrix (columns are optimized moelcular orbtials)
        mo_energy (np.ndarray): 1D array of molecular orbital energies
        dm_mat (np.ndarray): Converged density matrix
        huzinaga_op_std (np.ndarray): Huzinaga operator in standard basis (same basis as Fock operator).
    """
    #
    H_core_and_DFT_pot = scf_method.get_hcore() + DFT_potential

    if dm_initial_guess is None:
        # initial dm guess (using modified Hcore)
        fock_ortho = s_neg_half @ H_core_and_DFT_pot @ s_neg_half

        huzinaga_op_ortho = -1*(fock_ortho@enviro_proj_ortho_basis + enviro_proj_ortho_basis@fock_ortho)
        huzinaga_op_std = s_half @ huzinaga_op_ortho @ s_half

        fock_ortho += huzinaga_op_ortho

        mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
        mo_coeff_std = s_neg_half @ mo_coeff_ortho
        mo_occ = scf_method.get_occ(mo_energy, mo_coeff_std)
        dm_mat = scf_method.make_rdm1(mo_coeff=mo_coeff_std, mo_occ=mo_occ)
    else:
        dm_mat = dm_initial_guess
        vhf = scf_method.get_veff(dm=dm_mat)
        fock = vhf + H_core_and_DFT_pot
        fock_ortho = s_neg_half @ fock @ s_neg_half
        huzinaga_op_ortho = -1 * (fock_ortho @ enviro_proj_ortho_basis + enviro_proj_ortho_basis @ fock_ortho)
        huzinaga_op_std = s_half @ huzinaga_op_ortho @ s_half
        fock_ortho += huzinaga_op_ortho

    # SCF parameters
    nuclear_energy = scf_method.energy_nuc()
    max_iter = scf_method.max_cycle
    energy_conv_tol = scf_method.conv_tol
    dm_mat_old = np.zeros_like(dm_mat)
    rhf_energy_prev = 0
    conv_flag = False

    for i in range(max_iter):
        # build fock matrix
        vhf = scf_method.get_veff(dm=dm_mat)

        # Find RHF energy
        e_hcore_and_dft = np.einsum('ij,ji->', H_core_and_DFT_pot, dm_mat)
        e_coul = 0.5 * np.einsum('ij,ji->', vhf, dm_mat)
        e_huz = np.einsum('ij,ji->', huzinaga_op_std, dm_mat)
        rhf_energy = e_hcore_and_dft + e_coul + e_huz

        norm_dm_diff = np.linalg.norm(dm_mat-dm_mat_old)
        # check convergence
        if (np.abs(rhf_energy-rhf_energy_prev) < energy_conv_tol) and (norm_dm_diff < dm_conv_tol):
            conv_flag = True
            break

        # else continue alg
        fock = H_core_and_DFT_pot + vhf
        fock_ortho = s_neg_half @ fock @ s_neg_half
        huzinaga_op_ortho = -1 * (fock_ortho @ enviro_proj_ortho_basis + enviro_proj_ortho_basis @ fock_ortho)
        fock_ortho += huzinaga_op_ortho
        huzinaga_op_std = s_half @ huzinaga_op_ortho @ s_half

        mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
        mo_coeff_std = s_neg_half @ mo_coeff_ortho
        mo_occ = scf_method.get_occ(mo_energy, mo_coeff_std)
        dm_mat, dm_mat_old = scf_method.make_rdm1(mo_coeff=mo_coeff_std, mo_occ=mo_occ), dm_mat
        rhf_energy_prev = rhf_energy

    if conv_flag is False:
        warnings.warn('SCF has NOT converged')

    e_total = rhf_energy + nuclear_energy

    return conv_flag, e_total, mo_coeff_std, mo_energy, dm_mat, huzinaga_op_std


def nbed_driver(
    geometry: Path,
    n_active_atoms: int,
    basis: str,
    xc_functional: str,
    output: str,
    convergence: float = 1e-6,
    localization_method: str = "spade",
    projector_method: str = 'mu_shift',
    mu_level_shift: float = 1e6,
    run_ccsd_emb: bool = False,
    run_fci_emb: bool = False,
    run_global_fci: bool = False,
    max_ram_memory: int = 4000,
    pyscf_print_level: int =1,
    qubits: Optional[int] = None,
    savefile: Optional[Path] = None,
) -> Tuple[object, float]:
    """Function to return the embedding Qubit Hamiltonian.

    Args:
        geometry (Path): A path to an .xyz file describing moleclar geometry.
        n_active_atoms (int): The number of atoms to include in the active region.
        basis (str): The name of an atomic orbital basis set to use for chemistry calculations.
        xc_functonal (str): The name of an Exchange-Correlation functional to be used for DFT.
        output (str): one of "Openfermion" (TODO other options)
        convergence (float): The convergence tolerance for energy calculations.
        localization_method (str): Orbital Localisation method to use. One of 'spade', 'mullikan', 'boys' or 'ibo'.
        projector_method (str): Projector method to use. One of 'mu_shift', 'huzinaga'.
        mu_level_shift (float): Level shift parameter to use for mu-projector.
        run_ccsd_emb (bool): Whether or not to find the CCSD energy of embbeded system for reference.
        run_fci_emb (bool): Whether or not to find the FCI energy of embbeded system for reference.
        max_ram_memory (int): Amount of RAM memery in MB available for PySCF calculation
        pyscf_print_level (int): Amount of information PySCF prints
        qubits (int): The number of qubits available for the output hamiltonian.

    Returns:
        object: A Qubit Hamiltonian of some kind
        float: The classical contribution to the total energy.

    """
    logger.debug("Construcing molecule.")
    mol: gto.Mole = gto.Mole(atom=geometry, basis=basis, charge=0).build()

    e_nuc = mol.energy_nuc()
    logger.debug(f"Nuclear energy: {e_nuc}.")

    global_rks = scf.RKS(mol)
    global_rks.conv_tol = convergence
    global_rks.xc = xc_functional
    global_rks.max_memory = max_ram_memory
    global_rks.verbose = pyscf_print_level
    global_rks.kernel()
    global_rks_total_energy = global_rks.e_tot
    # Function names must be the same as the imput choices.
    logger.debug(f"Using {localization_method} localisation method.")

    (c_active,
     c_enviro,
     c_loc_occ_full,
     dm_active,
     dm_enviro,
     active_MO_inds,
     enviro_MO_inds,
     c_loc_occ_and_virt,
     active_virtual_MO_inds,
     enviro_virtual_MO_inds) = localize_molecular_orbs(global_rks,
                                                       n_active_atoms,
                                                       localization_method,
                                                       occ_THRESHOLD = 0.95,
                                                       virt_THRESHOLD = 0.95,
                                                       sanity_check = True,
                                                       run_virtual_localization=False)

    logger.debug("Write global molecule in localized basis")
    change_basis_matrix = orb_change_basis_operator(global_rks,
                                                    c_loc_occ_and_virt,
                                                    sanity_check=True)

    hcore_std = global_rks.get_hcore()
    global_rks.get_hcore = lambda *args: get_Hcore_new_basis(hcore_std,
                                                             change_basis_matrix)
    global_rks.get_veff = lambda mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1: get_new_RKS_Veff(
                                                           global_rks,
                                                           change_basis_matrix,
                                                           dm=dm,
                                                           check_result=True)
    # overwrite_with_localized_terms
    global_rks.mo_coeff = c_loc_occ_and_virt
    dm_loc = global_rks.make_rdm1(mo_coeff=global_rks.mo_coeff,
                                  mo_occ=global_rks.mo_occ)
    # fock_loc_basis = global_rks.get_hcore() + global_rks.get_veff(dm=dm_loc)
    fock_loc_basis = global_rks.get_fock(dm=dm_loc)

    # orbital_energies_std = global_rks.mo_energy
    orbital_energies_loc = np.diag(global_rks.mo_coeff.conj().T @ fock_loc_basis @ global_rks.mo_coeff)
    global_rks.mo_energy = orbital_energies_loc

    # check electronic energy matches standard global calc
    global_rks_total_energy_loc = global_rks.energy_tot(dm=dm_loc)
    if not np.isclose(global_rks_total_energy, global_rks_total_energy_loc):
        raise ValueError('electronic energy of standard calculation not matching localized calculation')

    # check if mo energies match
    # orbital_energies_std = global_rks.mo_energy
    # if not np.allclose(orbital_energies_std, orbital_energies_loc):
    #     raise ValueError('orbital energies of standard calc not matching localized calc')

    logger.debug("Calculating active and environment subsystem terms.")
    e_act, e_xc_act, j_act, k_act, v_xc_act = calc_RKS_components_from_dm(global_rks,
                                                                          dm_active,
                                                                          check_E_with_pyscf=True)
    e_env, e_xc_env, j_env, k_env, v_xc_env = calc_RKS_components_from_dm(global_rks,
                                                                          dm_enviro,
                                                                          check_E_with_pyscf=True)
    # Computing cross subsystem terms
    logger.debug("Calculating two electron cross subsystem energy.")
    two_e_cross = get_cross_terms_DFT(global_rks,
                                      dm_active,
                                      dm_enviro,
                                      j_env,
                                      j_act,
                                      e_xc_act,
                                      e_xc_env)

    energy_DFT_components = e_act + e_env + two_e_cross + e_nuc
    if not np.isclose(energy_DFT_components, global_rks_total_energy):
        raise ValueError('DFT energy of localized components not matching supersystem DFT')

    logger.debug("Define Hartree-Fock object")
    embedded_mol: gto.Mole = gto.Mole(atom=geometry,
                                      basis=basis,
                                      charge=0).build()

    logger.debug("re-defining total number of electrons to only include active system")
    embedded_mol.nelectron = 2 * len(active_MO_inds)

    embedded_RHF_MU = scf.RHF(embedded_mol)
    embedded_RHF_MU.max_memory = max_ram_memory
    embedded_RHF_MU.conv_tol = convergence
    embedded_RHF_MU.verbose = pyscf_print_level

    # ADD CHANGE OF BASIS
    # TODO: need to check if change of basis here is necessary (START)
    H_core_std_MU = embedded_RHF_MU.get_hcore()
    embedded_RHF_MU.get_hcore = lambda *args: get_Hcore_new_basis(H_core_std_MU, unitary_rot=change_basis_matrix)
    embedded_RHF_MU.get_veff = lambda mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1: get_new_RHF_Veff(
                                                                       embedded_RHF_MU,
                                                                       change_basis_matrix,
                                                                       dm=dm,
                                                                       hermi=hermi)
    # TODO: need to check this (END)

    logger.debug("Get global DFT potential to optimize embedded calc in.")
    g_act_and_env = global_rks.get_veff(dm=(dm_active+dm_enviro))
    g_act = global_rks.get_veff(dm=dm_active)
    DFT_potential = g_act_and_env - g_act

    logger.debug("Calculating projector onto subsystem B.")
    s_mat = global_rks.get_ovlp()
    s_half = sp.linalg.fractional_matrix_power(s_mat, 0.5)


    ### MU SHIFT
    enviro_projector_MU = get_enivornment_projector(c_loc_occ_and_virt,
                                                 s_half,
                                                 active_MO_inds,
                                                 enviro_MO_inds,
                                                 return_in_ortho_basis=False)
    v_emb_MU = (mu_level_shift * enviro_projector_MU) + DFT_potential
    hcore_std = embedded_RHF_MU.get_hcore()
    embedded_RHF_MU.get_hcore = lambda *args: hcore_std + v_emb_MU

    logger.debug("Running embedded RHF calculation.")
    embedded_RHF_MU.kernel()
    print(f'embedded HF energy MU_SHIFT: {embedded_RHF_MU.e_tot}, converged: {embedded_RHF_MU.converged}')
    dm_active_embedded_MU = embedded_RHF_MU.make_rdm1(mo_coeff=embedded_RHF_MU.mo_coeff,
                                                mo_occ=embedded_RHF_MU.mo_occ)

    shift = mol.nao - len(enviro_MO_inds)
    frozen_orb_inds_MU = [mo_i for mo_i in range(shift, mol.nao)]

    ccsd = cc.CCSD(embedded_RHF_MU)
    ccsd.conv_tol = convergence
    ccsd.max_memory = max_ram_memory
    ccsd.verbose = pyscf_print_level

    # Set which orbitals are to be frozen
    ccsd.frozen = frozen_orb_inds_MU
    e_ccsd_corr, t1, t2 = ccsd.kernel()

    wf_correction_MU = np.einsum("ij,ij", v_emb_MU, dm_active)
    e_wf_emb_MU = (ccsd.e_hf + e_ccsd_corr) + e_env + two_e_cross - wf_correction_MU
    print("CCSD Energy:\n\t%s", e_wf_emb_MU)

    # instead of freezing orbs like:
    # fci_scf_MU.frozen = frozen_orb_inds_MU
    # doesn't work as this will add energy constant. Instead delete environment part:
    active_inds = [mo_i for mo_i in range(embedded_RHF_MU.mo_coeff.shape[1]) if mo_i not in frozen_orb_inds_MU]
    embedded_RHF_MU.mo_coeff = embedded_RHF_MU.mo_coeff[:, active_inds]
    embedded_RHF_MU.mo_energy = embedded_RHF_MU.mo_energy[active_inds]
    embedded_RHF_MU.mo_occ = embedded_RHF_MU.mo_occ[active_inds]

    fci_scf_MU = fci.FCI(embedded_RHF_MU)
    fci_scf_MU.conv_tol = convergence
    fci_scf_MU.verbose = pyscf_print_level
    fci_scf_MU.max_memory = max_ram_memory
    fci_scf_MU.run()
    fci_emb_energy_MU = fci_scf_MU.e_tot
    e_wf_fci_emb_MU = (fci_emb_energy_MU) + e_env + two_e_cross - wf_correction_MU
    print("FCI MU Energy:\n\t%s", e_wf_fci_emb_MU)

    ### HUZINAGA
    embedded_RHF_HUZ = scf.RHF(embedded_mol)
    embedded_RHF_HUZ.max_memory = max_ram_memory
    embedded_RHF_HUZ.conv_tol = convergence
    embedded_RHF_HUZ.verbose = pyscf_print_level
    # embedded_RHF_HUZ.max_cycle = 500

    # ADD CHANGE OF BASIS
    # TODO: need to check if change of basis here is necessary (START)
    H_core_std_HUZ = embedded_RHF_HUZ.get_hcore()
    embedded_RHF_HUZ.get_hcore = lambda *args: get_Hcore_new_basis(H_core_std_HUZ, unitary_rot=change_basis_matrix)
    embedded_RHF_HUZ.get_veff = lambda mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1: get_new_RHF_Veff(
                                                                       embedded_RHF_HUZ,
                                                                       change_basis_matrix,
                                                                       dm=dm,
                                                                       hermi=hermi)
    # TODO: (END)

    s_neg_half = sp.linalg.fractional_matrix_power(s_mat, -0.5)
    enviro_projector_ortho = get_enivornment_projector(c_loc_occ_and_virt,
                                                       s_half,
                                                       active_MO_inds,
                                                       enviro_MO_inds,
                                                       return_in_ortho_basis=True)

    (conv_flag,
     energy_hf,
     c_active_embedded,
     mo_embedded_energy,
     dm_active_embedded,
     huzinaga_op_std) = huzinaga_RHF(embedded_RHF_HUZ,
                                     DFT_potential,
                                     enviro_projector_ortho,
                                     s_neg_half,
                                     s_half,
                                     dm_conv_tol=1e-6,
                                     dm_initial_guess=None) # TODO: use dm_active_embedded_MU (use mu answer to initialize!)

    print(f'embedded HF energy HUZINAGA: {energy_hf}, converged: {conv_flag}')
    # write results to pyscf object
    hcore_std = embedded_RHF_HUZ.get_hcore()
    v_emb_HUZ = huzinaga_op_std + DFT_potential
    embedded_RHF_HUZ.get_hcore = lambda *args: hcore_std + v_emb_HUZ
    embedded_RHF_HUZ.mo_coeff = c_active_embedded
    embedded_RHF_HUZ.mo_occ = embedded_RHF_HUZ.get_occ(mo_embedded_energy, c_active_embedded)
    embedded_RHF_HUZ.mo_energy = mo_embedded_energy

    n_act_mo = len(active_MO_inds)
    n_env_mo = len(enviro_MO_inds)
    frozen_orb_inds_HUZ = [i for i in range(n_act_mo, n_act_mo+n_env_mo)]

    # energy_rhf_active_embedded = embedded_RHF_HUZ.energy_tot(dm=dm_active_embedded)

    ccsd = cc.CCSD(embedded_RHF_HUZ)
    ccsd.conv_tol = convergence
    ccsd.max_memory = max_ram_memory
    ccsd.verbose = pyscf_print_level
    # Set which orbitals are to be frozen
    ccsd.frozen = frozen_orb_inds_HUZ
    e_ccsd_corr, t1, t2 = ccsd.kernel()

    wf_correction_HUZ = np.einsum("ij,ij", v_emb_HUZ, dm_active)
    e_wf_emb_HUZ = (ccsd.e_hf + e_ccsd_corr) + e_env + two_e_cross - wf_correction_HUZ
    print("CCSD Energy:\n\t%s", e_wf_emb_HUZ)


    active_inds = [mo_i for mo_i in range(embedded_RHF_HUZ.mo_coeff.shape[1]) if mo_i not in frozen_orb_inds_HUZ]
    embedded_RHF_HUZ.mo_coeff = embedded_RHF_HUZ.mo_coeff[:, active_inds]
    embedded_RHF_HUZ.mo_energy = embedded_RHF_HUZ.mo_energy[active_inds]
    embedded_RHF_HUZ.mo_occ = embedded_RHF_HUZ.mo_occ[active_inds]

    fci_scf_HUZ = fci.FCI(embedded_RHF_HUZ)
    # fci_scf_HUZ.frozen = frozen_orb_inds_HUZ
    fci_scf_HUZ.conv_tol = convergence
    fci_scf_HUZ.verbose = pyscf_print_level
    fci_scf_HUZ.max_memory = max_ram_memory
    fci_scf_HUZ.run()
    fci_emb_energy_HUZ = fci_scf_HUZ.e_tot
    e_wf_fci_emb_HUZ = (fci_emb_energy_HUZ) + e_env + two_e_cross - wf_correction_HUZ
    print("FCI HUZ Energy:\n\t%s", e_wf_fci_emb_HUZ)


    print(f'difference between CCSD Huz and Mu calcs: {np.abs(e_wf_emb_HUZ-e_wf_emb_MU)}')
    print(f'difference between FCI Huz and Mu calcs: {np.abs(e_wf_fci_emb_HUZ - e_wf_fci_emb_MU)}')

    # WF Method
    # Calculate the energy of embedded A
    # embedded_scf.get_hcore = lambda *args, **kwargs: h_core

    q_ham_MU = get_qubit_hamiltonian(embedded_RHF_MU)

    q_ham_HUZ = get_qubit_hamiltonian(embedded_RHF_HUZ)

    # converter_MU = HamiltonianConverter(q_ham_MU)
    # q_ham_MU = converter_MU.convert(output)
    # converter_HUZ = HamiltonianConverter(q_ham_HUZ)
    # q_ham_HUZ = converter_HUZ.convert(output)
    # if savefile:
    #     converter_MU.save(savefile)
    #     converter_HUZ.save(savefile)

    print(f'num e emb: {2 * len(active_MO_inds)}')
    print(active_MO_inds)
    print(enviro_MO_inds)
    classical_energy_MU = e_env + two_e_cross + e_nuc - wf_correction_MU
    classical_energy_HUZ = e_env + two_e_cross + e_nuc - wf_correction_HUZ

    output = {'mu_shift': {'H':q_ham_MU,
                           'energy_classical': classical_energy_MU},
              'huzinaga': {'H':q_ham_HUZ,
                           'energy_classical': classical_energy_HUZ}
              }

    if run_global_fci:
        mol_full: gto.Mole = gto.Mole(atom=geometry, basis=basis, charge=0).build()

        global_HF = scf.RHF(mol_full)
        global_HF.conv_tol = convergence
        global_HF.xc = xc_functional
        global_HF.max_memory = max_ram_memory
        global_HF.verbose = pyscf_print_level
        global_HF.kernel()

        global_fci = fci.FCI(global_HF)
        global_fci.conv_tol = convergence
        global_fci.verbose = pyscf_print_level
        global_fci.max_memory = max_ram_memory
        global_fci.run()
        global_fci_energy = global_fci.e_tot

        print(f'global FCI: {global_fci_energy}')


    return output

# def nbed_driver(
#     geometry: Path,
#     n_active_atoms: int,
#     basis: str,
#     xc_functional: str,
#     output: str,
#     convergence: float = 1e-6,
#     localization_method: str = "spade",
#     projector_method: str = 'mu_shift',
#     mu_level_shift: float = 1e6,
#     run_ccsd_emb: bool = False,
#     run_fci_emb: bool = False,
#     max_ram_memory: int = 4000,
#     pyscf_print_level: int =1,
#     qubits: Optional[int] = None,
#     savefile: Optional[Path] = None,
# ) -> Tuple[object, float]:
#     """Function to return the embedding Qubit Hamiltonian.
#
#     Args:
#         geometry (Path): A path to an .xyz file describing moleclar geometry.
#         n_active_atoms (int): The number of atoms to include in the active region.
#         basis (str): The name of an atomic orbital basis set to use for chemistry calculations.
#         xc_functonal (str): The name of an Exchange-Correlation functional to be used for DFT.
#         output (str): one of "Openfermion" (TODO other options)
#         convergence (float): The convergence tolerance for energy calculations.
#         localization_method (str): Orbital Localisation method to use. One of 'spade', 'mullikan', 'boys' or 'ibo'.
#         projector_method (str): Projector method to use. One of 'mu_shift', 'huzinaga'.
#         mu_level_shift (float): Level shift parameter to use for mu-projector.
#         run_ccsd_emb (bool): Whether or not to find the CCSD energy of embbeded system for reference.
#         run_fci_emb (bool): Whether or not to find the FCI energy of embbeded system for reference.
#         max_ram_memory (int): Amount of RAM memery in MB available for PySCF calculation
#         pyscf_print_level (int): Amount of information PySCF prints
#         qubits (int): The number of qubits available for the output hamiltonian.
#
#     Returns:
#         object: A Qubit Hamiltonian of some kind
#         float: The classical contribution to the total energy.
#
#     """
#     logger.debug("Construcing molecule.")
#     mol: gto.Mole = gto.Mole(atom=geometry, basis=basis, charge=0).build()
#
#     e_nuc = mol.energy_nuc()
#     logger.debug(f"Nuclear energy: {e_nuc}.")
#
#     global_rks = scf.RKS(mol)
#     global_rks.conv_tol = convergence
#     global_rks.xc = xc_functional
#     global_rks.max_memory = max_ram_memory
#     global_rks.verbose = pyscf_print_level
#     global_rks.kernel()
#     global_rks_total_energy = global_rks.e_tot
#     # Function names must be the same as the imput choices.
#     logger.debug(f"Using {localization_method} localisation method.")
#
#     (c_active,
#      c_enviro,
#      c_loc_occ_full,
#      dm_active,
#      dm_enviro,
#      active_MO_inds,
#      enviro_MO_inds,
#      c_loc_occ_and_virt,
#      active_virtual_MO_inds,
#      enviro_virtual_MO_inds) = localize_molecular_orbs(global_rks,
#                                                        n_active_atoms,
#                                                        localization_method,
#                                                        occ_THRESHOLD = 0.95,
#                                                        virt_THRESHOLD = 0.95,
#                                                        sanity_check = False,
#                                                        run_virtual_localization=False)
#
#     logger.debug("Write global molecule in localized basis")
#     change_basis_matrix = orb_change_basis_operator(global_rks,
#                                                     c_loc_occ_and_virt,
#                                                     sanity_check=True)
#
#     hcore_std = global_rks.get_hcore()
#     global_rks.get_hcore = lambda *args: get_Hcore_new_basis(hcore_std,
#                                                              change_basis_matrix)
#     global_rks.get_veff = lambda mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1: get_new_RKS_Veff(
#                                                            global_rks,
#                                                            change_basis_matrix,
#                                                            dm=dm,
#                                                            check_result=True)
#     # overwrite_with_localized_terms
#     global_rks.mo_coeff = c_loc_occ_and_virt
#     dm_loc = global_rks.make_rdm1(mo_coeff=global_rks.mo_coeff,
#                                   mo_occ=global_rks.mo_occ)
#     # fock_loc_basis = global_rks.get_hcore() + global_rks.get_veff(dm=dm_loc)
#     fock_loc_basis = global_rks.get_fock(dm=dm_loc)
#
#     # orbital_energies_std = global_rks.mo_energy
#     orbital_energies_loc = np.diag(global_rks.mo_coeff.conj().T @ fock_loc_basis @ global_rks.mo_coeff)
#     global_rks.mo_energy = orbital_energies_loc
#
#     # check electronic energy matches standard global calc
#     global_rks_total_energy_loc = global_rks.energy_tot(dm=dm_loc)
#     if not np.isclose(global_rks_total_energy, global_rks_total_energy_loc):
#         raise ValueError('electronic energy of standard calculation not matching localized calculation')
#
#     # check if mo energies match
#     # orbital_energies_std = global_rks.mo_energy
#     # if not np.allclose(orbital_energies_std, orbital_energies_loc):
#     #     raise ValueError('orbital energies of standard calc not matching localized calc')
#
#     logger.debug("Calculating active and environment subsystem terms.")
#     e_act, e_xc_act, j_act, k_act, v_xc_act = calc_RKS_components_from_dm(global_rks,
#                                                                           dm_active,
#                                                                           check_E_with_pyscf=True)
#     e_env, e_xc_env, j_env, k_env, v_xc_env = calc_RKS_components_from_dm(global_rks,
#                                                                           dm_enviro,
#                                                                           check_E_with_pyscf=True)
#     # Computing cross subsystem terms
#     logger.debug("Calculating two electron cross subsystem energy.")
#     two_e_cross = get_cross_terms_DFT(global_rks,
#                                       dm_active,
#                                       dm_enviro,
#                                       j_env,
#                                       j_act,
#                                       e_xc_act,
#                                       e_xc_env)
#
#     logger.debug("Define Hartree-Fock object")
#     embedded_mol: gto.Mole = gto.Mole(atom=geometry,
#                                       basis=basis,
#                                       charge=0).build()
#
#     logger.debug("re-defining total number of electrons to only include active system")
#     embedded_mol.nelectron = 2 * len(active_MO_inds)
#     embedded_RHF = scf.RHF(embedded_mol)
#     embedded_RHF.max_memory = max_ram_memory
#     embedded_RHF.conv_tol = convergence
#     embedded_RHF.verbose = pyscf_print_level
#
#     logger.debug("Get global DFT potential to optimize embedded calc in.")
#     g_act_and_env = global_rks.get_veff(dm=(dm_active+dm_enviro))
#     g_act = global_rks.get_veff(dm=dm_active)
#     DFT_potential = g_act_and_env - g_act
#
#     logger.debug("Calculating projector onto subsystem B.")
#     s_mat = global_rks.get_ovlp()
#     s_half = sp.linalg.fractional_matrix_power(s_mat, 0.5)
#
#     if projector_method == 'mu_shift':
#         enviro_projector = get_enivornment_projector(c_loc_occ_and_virt,
#                                                      s_half,
#                                                      active_MO_inds,
#                                                      enviro_MO_inds,
#                                                      return_in_ortho_basis=False)
#
#         v_emb = (mu_level_shift * enviro_projector) + DFT_potential
#
#         hcore_std = embedded_RHF.get_hcore()
#         embedded_RHF.get_hcore = lambda *args: hcore_std + v_emb
#
#         logger.debug("Running embedded RHF calculation.")
#         embedded_RHF.kernel()
#
#         print(f'embedded HF energy : {embedded_RHF.e_tot}, converged: {embedded_RHF.converged}')
#         dm_active_embedded = embedded_RHF.make_rdm1(mo_coeff=embedded_RHF.mo_coeff,
#                                                     mo_occ=embedded_RHF.mo_occ)
#
#         shift = mol.nao - len(enviro_MO_inds)
#         frozen_orb_inds = [mo_i for mo_i in range(shift, mol.nao)]
#
#     elif projector_method == 'huzinaga':
#         s_neg_half = sp.linalg.fractional_matrix_power(s_mat, -0.5)
#         enviro_projector_ortho = get_enivornment_projector(c_loc_occ_and_virt,
#                                                            s_half,
#                                                            active_MO_inds,
#                                                            enviro_MO_inds,
#                                                            return_in_ortho_basis=True)
#
#         (conv_flag,
#          energy_hf,
#          c_active_embedded,
#          mo_embedded_energy,
#          dm_active_embedded,
#          huzinaga_op_std) = huzinaga_RHF(embedded_RHF,
#                                          DFT_potential,
#                                          enviro_projector_ortho,
#                                          s_neg_half,
#                                          s_half,
#                                          dm_conv_tol=1e-6)
#
#         print(f'embedded HF energy : {energy_hf}, converged: {conv_flag}')
#         # write results to pyscf object
#         hcore_std = embedded_RHF.get_hcore()
#         v_emb = huzinaga_op_std + DFT_potential
#         embedded_RHF.get_hcore = lambda *args: hcore_std + v_emb
#         embedded_RHF.mo_coeff = c_active_embedded
#         embedded_RHF.mo_occ = embedded_RHF.get_occ(mo_embedded_energy, c_active_embedded)
#         embedded_RHF.mo_energy = mo_embedded_energy
#
#         print(mo_embedded_energy)
#         n_act_mo = len(active_MO_inds)
#         n_env_mo = len(enviro_MO_inds)
#         frozen_orb_inds = [i for i in range(n_act_mo, n_act_mo+n_env_mo)]
#     else:
#         raise ValueError(f'unknown projector method: {projector_method}')
#
#     energy_rhf_active_embedded = embedded_RHF.energy_tot(dm=dm_active_embedded)
#
#     wf_correction = np.einsum("ij,ij", v_emb, dm_active)
#
#     # shift = mol.nao - len(enviro_MO_inds)
#     # frozen_orb_inds = [mo_i for mo_i in range(shift, mol.nao)]
#     if run_ccsd_emb:
#         # Run CCSD as WF method
#         ccsd = cc.CCSD(embedded_RHF)
#         ccsd.conv_tol = convergence
#         ccsd.max_memory = max_ram_memory
#         ccsd.verbose = pyscf_print_level
#
#         # Set which orbitals are to be frozen
#         ccsd.frozen = frozen_orb_inds
#         try:
#             e_ccsd_corr, t1, t2 = ccsd.kernel()
#             if not np.isclose(energy_rhf_active_embedded, ccsd.e_hf):
#                 raise ValueError('CCSD hartree fock calc not matching RHF calculation')
#         except np.linalg.LinAlgError as e:
#             print("\n====CCSD ERROR====\n")
#             print(e)
#
#         # Add up the parts again
#         e_wf_emb = (ccsd.e_hf + e_ccsd_corr) + e_env + two_e_cross - wf_correction
#         print("CCSD Energy:\n\t%s", e_wf_emb)
#
#     if run_fci_emb:
#         # Note for VQE calc to match this one must set frozen orbitals to None!
#         fci_scf = fci.FCI(embedded_RHF)
#         fci_scf.frozen = frozen_orb_inds
#         fci_scf.conv_tol = convergence
#         fci_scf.verbose = pyscf_print_level
#         fci_scf.max_memory = max_ram_memory
#         fci_scf.run()
#         fci_emb_energy = fci_scf.e_tot
#
#         # choose active orbs and electrons (full space!) CASSCF calc!
#         # ncas, nelecas = (embedded_RHF.mo_coeff.shape[1], embedded_RHF.mol.nelectron)
#         # fci_scf = embedded_RHF.CASSCF(ncas, nelecas, frozen=frozen_orb_inds)
#         # fci_scf.conv_tol = convergence
#         # fci_scf.verbose = pyscf_print_level
#         # fci_scf.max_memory = max_ram_memory
#         # fci_scf.run()
#         # fci_emb_energy = fci_scf.e_tot
#
#         # Add up the parts again
#         e_wf_fci_emb = (fci_emb_energy) + e_env + two_e_cross - wf_correction
#         print("FCI Energy:\n\t%s", e_wf_fci_emb)
#
#     # WF Method
#     # Calculate the energy of embedded A
#     # embedded_scf.get_hcore = lambda *args, **kwargs: h_core
#
#     q_ham = get_qubit_hamiltonian(embedded_RHF,
#                                   frozen_orb_inds)
#
#     # converter = HamiltonianConverter(q_ham)
#     # q_ham = converter.convert(output)
#     #
#     # if savefile:
#     #     converter.save(savefile)
#
#     print(f'num e emb: {2 * len(active_MO_inds)}')
#     print(active_MO_inds)
#     print(enviro_MO_inds)
#     classical_energy = e_env + two_e_cross + e_nuc - wf_correction
#
#     return q_ham, classical_energy


def cli() -> None:
    """CLI Interface."""
    setup_logs()
    args = parse()
    qham, e_classical = nbed_driver(
        geometry=args["geometry"],
        active_atoms=args["active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        output=args["output"],
        localisation=args["localisation"],
        convergence=args["convergence"],
        run_ccsd=args["ccsd"],
        qubits=args["qubits"],
        savefile=args["savefile"],
    )
    print("Qubit Hamiltonian:")
    print(qham)
    print(f"Classical Energy (Ha): {e_classical}")


if __name__ == "__main__":
    cli()
