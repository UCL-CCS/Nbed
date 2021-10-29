"""Main embedding functionality."""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy as sp
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import jordan_wigner, bravyi_kitaev, bravyi_kitaev_tree
from openfermion import QubitOperator

from pyscf import ao2mo, cc, fci, gto, lib, scf
from pyscf.dft import numint
from pyscf.dft.rks import get_veff as rks_get_veff
from pyscf.lib import StreamObject
from openfermion.utils import count_qubits
from openfermion.ops.representations import get_active_space_integrals

from nbed.ham_converter import HamiltonianConverter
from nbed.localisation import localize_molecular_orbs, orb_change_basis_operator
from nbed.utils import parse, setup_logs

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
    """Get projector onto environement MOs

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


def get_molecular_hamiltonian(
    scf_method: StreamObject,
    embedded_energy_constant: Optional[int] = 0,
) -> InteractionOperator:
    """Returns second quantized fermionic molecular Hamiltonian

    Note if an embedded subsystem C matrix (containing MOs as columns) is used then the environment orbitals
    should have been removed (aka columns deleted) before using as an input to this function.

    TODO: if get_active_space_integrals openfermion function is used then core constant is adjustment to constant
    shift term in Hamiltonian from integrating out core orbitals.

    Args:
        scf_method (StreamObject): A pyscf self-consistent method.
        embedded_energy_constant (list[int]): constant shift in Hamiltonian from classical correction to embedded calculation

    Returns:
        molecular_hamiltonian (InteractionOperator): fermionic molecular Hamiltonian
    """

    # C_matrix containing orbitals to be considered
    # if there are any environment orbs that have been projected out... these should NOT be present in the
    # scf_method.mo_coeff array (aka columns should be deleted!)
    c_matrix_active = scf_method.mo_coeff
    n_orbs = c_matrix_active.shape[1]

    one_body_integrals = c_matrix_active.T @ scf_method.get_hcore() @ c_matrix_active

    two_body_compressed = ao2mo.kernel(scf_method.mol, c_matrix_active)

    # get electron repulsion integrals
    eri = ao2mo.restore(1, two_body_compressed, n_orbs)  # no permutation symmetry

    # Openfermion uses pysicist notation whereas pyscf uses chemists
    two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order="C")

    core_constant, one_body_ints_reduced, two_body_ints_reduced = (
        0,
        one_body_integrals,
        two_body_integrals,
    )
    # core_constant, one_body_ints_reduced, two_body_ints_reduced = get_active_space_integrals(
    #                                                                                        one_body_integrals,
    #                                                                                        two_body_integrals,
    #                                                                                        occupied_indices=None,
    #                                                                                        active_indices=active_mo_inds
    #                                                                                         )

    # print(f"core constant: {core_constant}")
    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
        one_body_ints_reduced, two_body_ints_reduced
    )

    # add core constant and embedded energy constant
    constant_term = core_constant + embedded_energy_constant
    molecular_hamiltonian = InteractionOperator(
        constant_term, one_body_coefficients, 0.5 * two_body_coefficients
    )

    return molecular_hamiltonian


def get_qubit_hamiltonian(
    molecular_ham: InteractionOperator,
    transformation: str = 'jordan_wigner'
) -> QubitOperator:
    """Takes in a second quantized fermionic Hamiltonian and returns a qubit hamiltonian under defined fermion
       to qubit transformation.

    Args:
        molecular_ham (InteractionOperator): A pyscf self-consistent method.
        transformation (str): Type of fermion to qubit mapping (jordan_wigner, bravyi_kitaev, bravyi_kitaev_tree)

    Returns:
        Qubit_Hamiltonian (QubitOperator): Qubit hamiltonian of molecular Hamiltonian (under specified fermion mapping)
    """
    if transformation == 'jordan_wigner':
        qubit_Hamiltonian = jordan_wigner(molecular_ham)
    elif transformation == 'bravyi_kitaev':
        qubit_Hamiltonian = bravyi_kitaev(molecular_ham)
    elif transformation == 'bravyi_kitaev_tree':
        qubit_Hamiltonian = bravyi_kitaev_tree(molecular_ham)
    else:
        raise ValueError(f'unknown fermion to qubit transformation: {transformation}')

    return qubit_Hamiltonian


def huzinaga_RHF(
    scf_method: StreamObject,
    dft_potential: np.ndarray,
    enviro_proj_ortho_basis: np.ndarray,
    s_neg_half: np.ndarray,
    s_half: np.ndarray,
    dm_conv_tol: float = 1e-6,
    dm_initial_guess: Optional[np.ndarray] = None,
):
    """Manual RHF calculation that is implemented using the huzinaga operator

    Note this function uses lowdin (symmetric) orthogonalization only! (PySCF sometimes uses meta-lowdin and NAO). Also
    the intial density matrix guess is based on the modified core Hamilotnian (containing projector and DFT potential)
    PySCF has other methods for initial guess that aren't available here. Manual guess can also be given).

    TODO: make a subclass from PySCF RHF object. Can include all this functionality there. Problems in this function
    can occur due to DIIS and other clever PySCF methods not being available.

    Args:
        scf_method (StreamObjecty):PySCF RHF object (containing info about max cycles and convergence tolerence)
        dft_potential (np.ndarray): DFT active and environment two body terms - DFT active environemnt two body term
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
    # Create an initial dm if needed.
    if dm_initial_guess is None:
        fock = scf_method.get_hcore() + dft_potential

        # Create the orthogonal fock operator
        fock_ortho = s_neg_half @ fock @ s_neg_half
        huzinaga_op_ortho = -1 * (
            fock_ortho @ enviro_proj_ortho_basis + enviro_proj_ortho_basis @ fock_ortho
        )
        huzinaga_op_std = s_half @ huzinaga_op_ortho @ s_half
        fock_ortho += huzinaga_op_ortho

        mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
        mo_coeff_std = s_neg_half @ mo_coeff_ortho
        mo_occ = scf_method.get_occ(mo_energy, mo_coeff_std)
        dm_initial_guess = scf_method.make_rdm1(mo_coeff=mo_coeff_std, mo_occ=mo_occ)

    dm_mat = dm_initial_guess
    conv_flag = False
    rhf_energy_prev = 0
    for _ in range(scf_method.max_cycle):
        # build fock matrix
        vhf = scf_method.get_veff(dm=dm_mat)
        fock = scf_method.get_hcore() + dft_potential + vhf

        # else continue alg
        fock_ortho = s_neg_half @ fock @ s_neg_half
        huzinaga_op_ortho = -1 * (
            fock_ortho @ enviro_proj_ortho_basis + enviro_proj_ortho_basis @ fock_ortho
        )
        fock_ortho += huzinaga_op_ortho

        huzinaga_op_std = s_half @ huzinaga_op_ortho @ s_half

        mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
        mo_coeff_std = s_neg_half @ mo_coeff_ortho
        mo_occ = scf_method.get_occ(mo_energy, mo_coeff_std)

        # Create initial values for i+1 run.
        dm_mat_old = dm_mat
        dm_mat = scf_method.make_rdm1(mo_coeff=mo_coeff_std, mo_occ=mo_occ)
        # Find RHF energy
        e_core_dft = np.einsum("ij,ji->", scf_method.get_hcore() + dft_potential, dm_mat)
        e_coul = 0.5 * np.einsum("ij,ji->", vhf, dm_mat)
        e_huz = np.einsum("ij,ji->", huzinaga_op_std, dm_mat)
        rhf_energy = e_core_dft + e_coul + e_huz

        # check convergence
        run_diff = np.abs(rhf_energy - rhf_energy_prev)
        norm_dm_diff = np.linalg.norm(dm_mat - dm_mat_old)
        if (run_diff < scf_method.conv_tol) and (norm_dm_diff < dm_conv_tol):
            conv_flag = True
            break

        rhf_energy_prev = rhf_energy

    if conv_flag is False:
        warnings.warn("SCF has NOT converged.")

    e_total = rhf_energy + scf_method.energy_nuc()

    return conv_flag, e_total, mo_coeff_std, mo_energy, dm_mat, huzinaga_op_std


class nbed_driver(object):
    """Function to return the embedding Qubit Hamiltonian.

    Args:
        geometry (Path): A path to an .xyz file describing moleclar geometry.
        n_active_atoms (int): The number of atoms to include in the active region.
        basis (str): The name of an atomic orbital basis set to use for chemistry calculations.
        xc_functional (str): The name of an Exchange-Correlation functional to be used for DFT.
        output (str): one of "Openfermion" (TODO other options)
        convergence (float): The convergence tolerance for energy calculations.
        charge (int): Charge of molecular species
        localization_method (str): Orbital Localisation method to use. One of 'spade', 'mullikan', 'boys' or 'ibo'.
        run_mu_shift (bool): Whether to run mu shift projector method
        run_huzinaga (bool): Whether to run run huzinaga method
        mu_level_shift (float): Level shift parameter to use for mu-projector.
        run_ccsd_emb (bool): Whether or not to find the CCSD energy of embbeded system for reference.
        run_fci_emb (bool): Whether or not to find the FCI energy of embbeded system for reference.
        perform_global_fci (bool): Whether to find the FCI energy of total molecular system for reference (expensive!).
        max_ram_memory (int): Amount of RAM memory in MB available for PySCF calculation
        pyscf_print_level (int): Amount of information PySCF prints
        get_full_system_H (bool): Whether to get second quantised Hamiltonian for the whole system for reference.

    Attributes:
        global_fci (StreamObject): PySCF fci object of full molecular system (NOT subsystem)
        global_HF (StreamObject): PySCF rhf object of full molecular system (NOT subsystem)
        global_rks_total_energy (float): Full system RKS DFT ground state energy
        e_act (float): Active energy from subsystem DFT calculation
        e_env (float): Environment energy from subsystem DFT calculation
        two_e_cross (flaot): two electron energy from cross terms (includes exchange correlation
                             and Coloumb contribution) of subsystem DFT calculation
        molecular_ham_MU (InteractionOperator): molecular Hamiltonian for active subsystem (projection using mu shift operator)
        classical_energy_MU (float): environment correction energy to obtain total energy (for mu shift method)
        molecular_ham_HUZ (InteractionOperator): molecular Hamiltonian for active subsystem (projection using huzianga operator)
        classical_energy_HUZ (float): environment correction energy to obtain total energy (for huzianga method)
        full_system_molecular_ham (InteractionOperator): molecular Hamiltonian for full molecular system
        emb_rhf_etot_mu_shift (float): Total system energy, with RHF corrected energy of embedded active subsystem (mu shifted)
        ccsd_emb_MU (cc.CCSD): PySCF CCSD object of embedded active subsystem (mu shifted)
        e_wf_ccsd_emb_MU (float): Total system energy, with CCSD corrected energy of embedded active subsystem (mu shifted)
        fci_emb_MU (fci.FCI): PySCF FCI object of embedded active subsystem (mu shifted)
        e_wf_fci_emb_MU (float): Total system energy, with FCI corrected energy of embedded active subsystem (mu shifted)
        ccsd_emb_HUZ (cc.CCSD): PySCF CCSD object of embedded active subsystem (huzinaga operator)
        e_wf_ccsd_emb_HUZ (float): Total system energy, with CCSD corrected energy of embedded active subsystem (huzinaga operator)
        fci_emb_HUZ (fci.FCI): PySCF FCI object of embedded active subsystem (huzinaga operator)
        e_wf_fci_emb_HUZ (float): Total system energy, with FCI corrected energy of embedded active subsystem (huzinaga operator)
        emb_rhf_etot_HUZ (float) (float): Total system energy, with RHF corrected energy of embedded active subsystem (huzinaga operator)
        n_qubits_HUZ (int): number of qubits needed for Huzinaga embedded Hamiltonian
        n_qubits_MU (int): number of qubits needed for mu shifted embedded Hamiltonian
        n_qubits_full_system (int): number of qubits needed to describe full system Hamiltonian
    """
    def __init__(self,
                 geometry: Path,
                 n_active_atoms: int,
                 basis: str,
                 xc_functional: str,
                 output: str,
                 convergence: Optional[float] = 1e-6,
                 charge: Optional[int] = 0,
                 localization_method: Optional[str] = "spade",
                 run_mu_shift: Optional[bool] = True,
                 run_huzinaga: Optional[bool] = True,
                 mu_level_shift: Optional[float] = 1e6,
                 run_ccsd_emb: Optional[bool] = False,
                 run_fci_emb: Optional[bool] = False,
                 perform_global_fci: Optional[bool] = False,
                 max_ram_memory: Optional[int] = 4000,
                 pyscf_print_level: int = 1,
                 get_full_system_H: Optional[bool] = False,
                 ):

        self.geometry = geometry
        self.n_active_atoms = n_active_atoms
        self.basis = basis
        self.xc_functional = xc_functional
        self.output = output
        self.convergence = convergence
        self.charge = charge
        self.localization_method = localization_method
        self.run_mu_shift = run_mu_shift
        self.run_huzinaga = run_huzinaga
        self.mu_level_shift = mu_level_shift
        self.run_ccsd_emb = run_ccsd_emb
        self.run_fci_emb = run_fci_emb
        self.perform_global_fci = perform_global_fci
        self.max_ram_memory = max_ram_memory
        self.pyscf_print_level = pyscf_print_level
        self.get_full_system_H = get_full_system_H

        if int(run_huzinaga)+int(run_mu_shift) == 0:
            raise ValueError('Not running any embedding calculation. Please use huzinaga and/or mu shift approach')

        # Attributes
        self.global_fci = None
        self.global_HF = None
        self.global_rks_total_energy = None
        self.e_act = None
        self.e_env = None
        self.two_e_cross = None
        self.localized_system = None

        self.molecular_ham_MU = None
        self.classical_energy_MU = None
        self.molecular_ham_HUZ = None
        self.classical_energy_HUZ = None
        self.full_system_molecular_ham = None

        self.emb_rhf_etot_mu_shift = None
        self.ccsd_emb_MU = None
        self.e_wf_ccsd_emb_MU = None
        self.fci_emb_MU = None
        self.e_wf_fci_emb_MU = None

        self.emb_rhf_etot_HUZ = None
        self.ccsd_emb_HUZ = None
        self.e_wf_ccsd_emb_HUZ = None
        self.fci_emb_HUZ = None
        self.e_wf_fci_emb_HUZ = None

        self.n_qubits_HUZ = None
        self.n_qubits_MU = None
        self.n_qubits_full_system = None

        if self.perform_global_fci:
            self.run_global_fci()

        if self.get_full_system_H:
            self.Generate_standard_H()

    def build_mol(self) -> gto.mole:
        """Function to build PySCF molecule

        Returns:
            full_mol (gto.mol): built PySCF molecule object
        """
        logger.debug("Construcing molecule.")
        full_mol = gto.Mole(atom=self.geometry, basis=self.basis, charge=self.charge).build()
        return full_mol

    def run_global_fci(self) -> StreamObject:
        """ Function to run full molecule FCI calculation. Note this is very expensive
        """
        mol_full = self.build_mol()
        # run Hartree-Fock
        self.global_HF = scf.RHF(mol_full)
        self.global_HF.conv_tol = self.convergence
        self.global_HF.max_memory = self.max_ram_memory
        self.global_HF.verbose = self.pyscf_print_level
        self.global_HF.kernel()

        # run FCI after HF
        self.global_fci = fci.FCI(self.global_HF)
        self.global_fci.conv_tol = self.convergence
        self.global_fci.verbose = self.pyscf_print_level
        self.global_fci.max_memory = self.max_ram_memory
        self.global_fci.run()
        print(f"global FCI: {self.global_fci.e_tot}")

        return None

    def run_cheap_global_dft(self):
        """ Method to run full cheap molecule RKS DFT calculation.

         Note this is necessary to perform localization procedure.
        """
        mol_full = self.build_mol()

        global_rks = scf.RKS(mol_full)
        global_rks.conv_tol = self.convergence
        global_rks.xc = self.xc_functional
        global_rks.max_memory = self.max_ram_memory
        global_rks.verbose = self.pyscf_print_level
        global_rks.kernel()

        self.global_rks_total_energy = global_rks.e_tot

        return global_rks

    def define_rks_in_new_basis(self, pyscf_scf_rks: gto.Mole, change_basis_matrix):
        """Redefine global RKS pyscf object in new (localized) basis
        """
        # write operators in localised basis
        hcore_std = pyscf_scf_rks.get_hcore()
        pyscf_scf_rks.get_hcore = lambda *args: get_Hcore_new_basis(hcore_std, change_basis_matrix)

        pyscf_scf_rks.get_veff = (lambda mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1:
                                  get_new_RKS_Veff(pyscf_scf_rks,
                                                   change_basis_matrix,
                                                   dm=dm,
                                                   check_result=True))

        # overwrite C matrix with localised orbitals
        pyscf_scf_rks.mo_coeff = self.localized_system.c_loc_occ_and_virt
        dm_loc = pyscf_scf_rks.make_rdm1(mo_coeff=pyscf_scf_rks.mo_coeff,
                                         mo_occ=pyscf_scf_rks.mo_occ)

        # fock_loc_basis = global_rks.get_hcore() + global_rks.get_veff(dm=dm_loc)
        fock_loc_basis = pyscf_scf_rks.get_fock(dm=dm_loc)

        # orbital_energies_std = global_rks.mo_energy
        orbital_energies_loc = np.diag(
            pyscf_scf_rks.mo_coeff.conj().T @ fock_loc_basis @ pyscf_scf_rks.mo_coeff
        )
        pyscf_scf_rks.mo_energy = orbital_energies_loc

        # check electronic energy matches standard global calc
        global_rks_total_energy_loc = pyscf_scf_rks.energy_tot(dm=dm_loc)
        if not np.isclose(self.global_rks_total_energy, global_rks_total_energy_loc):
            raise ValueError("electronic energy of standard calculation not matching localized calculation")

        # check if mo energies match
        # orbital_energies_std = global_rks.mo_energy
        # if not np.allclose(orbital_energies_std, orbital_energies_loc):
        #     raise ValueError('orbital energies of standard calc not matching localized calc')

        return pyscf_scf_rks

    def get_embedded_rhf(self, change_basis_matrix) -> scf.RHF:
        """Function to build embedded restricted Hartree Fock object for active subsystem

        Note this function overwrites the total number of electrons to only include active number

        Returns:
            embedded_RHF (scf.RHF): PySCF RHF object for active embedded subsystem
        """
        embedded_mol = self.build_mol()
        # overwrite total number of electrons to only include active system
        embedded_mol.nelectron = 2 * len(self.localized_system.active_MO_inds)

        logger.debug("Define Hartree-Fock object")
        embedded_RHF = scf.RHF(embedded_mol)
        embedded_RHF.max_memory = self.max_ram_memory
        embedded_RHF.conv_tol = self.convergence
        embedded_RHF.verbose = self.pyscf_print_level

        ##############################################################################################
        logger.debug("Define Hartree-Fock object in localized basis")
        # TODO: need to check if change of basis here is necessary (START)
        H_core_std = embedded_RHF.get_hcore()
        embedded_RHF.get_hcore = lambda *args: get_Hcore_new_basis(H_core_std,
                                                                   unitary_rot=change_basis_matrix)
        embedded_RHF.get_veff = (lambda mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1: get_new_RHF_Veff(
                                                                                       embedded_RHF,
                                                                                       change_basis_matrix,
                                                                                       dm=dm,
                                                                                       hermi=hermi)
                                 )
        return embedded_RHF

    def subsystem_dft(self, local_basis_pyscf_scf_rks: gto.Mole):
        """Function to perform subsystem RKS DFT calculation
        """
        logger.debug("Calculating active and environment subsystem terms.")
        (self.e_act,
         e_xc_act,
         j_act,
         k_act,
         v_xc_act) = calc_RKS_components_from_dm(local_basis_pyscf_scf_rks,
                                                 self.localized_system.dm_active,
                                                 check_E_with_pyscf=True)
        (self.e_env,
         e_xc_env,
         j_env,
         k_env,
         v_xc_env) = calc_RKS_components_from_dm(local_basis_pyscf_scf_rks,
                                                 self.localized_system.dm_enviro,
                                                 check_E_with_pyscf=True)
        # Computing cross subsystem terms
        logger.debug("Calculating two electron cross subsystem energy.")
        self.two_e_cross = get_cross_terms_DFT(local_basis_pyscf_scf_rks,
                                               self.localized_system.dm_active,
                                               self.localized_system.dm_enviro,
                                               j_env,
                                               j_act,
                                               e_xc_act,
                                               e_xc_env)

        energy_DFT_components = self.e_act + self.e_env + self.two_e_cross + local_basis_pyscf_scf_rks.energy_nuc()
        print('act:', self.e_act)
        print('e_env:', self.e_env)
        print('two_e_cross:', self.two_e_cross)
        print('energy_nuc:', local_basis_pyscf_scf_rks.energy_nuc())

        if not np.isclose(energy_DFT_components, self.global_rks_total_energy):
            raise ValueError("DFT energy of localized components not matching supersystem DFT")

        return None

    def run_emb_CCSD(self, emb_pyscf_scf_rhf: scf.RHF, frozen_orb_list: Optional[list] = None) -> Tuple[cc.CCSD, float]:
        """Function run CCSD on embedded restricted Hartree Fock object

        Note emb_pyscf_scf_rhf is RHF object for the active embedded subsystem (defined in localized basis)
        (see get_embedded_rhf method)

        Args:
            emb_pyscf_scf_rhf (scf.RHF): PySCF restricted Hartree Fock object of active embedded subsystem
            frozen_orb_list (List): A path to an .xyz file describing moleclar geometry.
        Returns:
            ccsd (cc.CCSD): PySCF CCSD object
            e_ccsd_corr (float): electron correlation CCSD energy
        """
        ccsd = cc.CCSD(emb_pyscf_scf_rhf)
        ccsd.conv_tol = self.convergence
        ccsd.max_memory = self.max_ram_memory
        ccsd.verbose = self.pyscf_print_level

        # Set which orbitals are to be frozen
        ccsd.frozen = frozen_orb_list
        e_ccsd_corr, t1, t2 = ccsd.kernel()
        return ccsd, e_ccsd_corr

    def run_emb_FCI(self, emb_pyscf_scf_rhf: gto.Mole, frozen_orb_list: Optional[list] = None) -> Tuple[fci.FCI]:
        """Function run FCI on embedded restricted Hartree Fock object

        Note emb_pyscf_scf_rhf is RHF object for the active embedded subsystem (defined in localized basis)
        (see get_embedded_rhf method)

        Args:
            emb_pyscf_scf_rhf (scf.RHF): PySCF restricted Hartree Fock object of active embedded subsystem
            frozen_orb_list (List): A path to an .xyz file describing moleclar geometry.
        Returns:
            fci_scf (fci.FCI): PySCF FCI object
        """
        fci_scf = fci.FCI(emb_pyscf_scf_rhf)
        fci_scf.conv_tol = self.convergence
        fci_scf.verbose = self.pyscf_print_level
        fci_scf.max_memory = self.max_ram_memory

        fci_scf.frozen = frozen_orb_list
        fci_scf.run()
        return fci_scf

    def Generate_embedded_H(self):
        """Generate fermionic embedded Hamiltonian

        Note run_mu_shift (bool) and run_huzinaga (bool) flags define which method to use (can be both)
        This is done when object is initialized
        """
        global_rks = self.run_cheap_global_dft()
        e_nuc = global_rks.mol.energy_nuc()

        self.localized_system = localize_molecular_orbs(global_rks,
                                                        self.n_active_atoms,
                                                        self.localization_method,
                                                        occ_THRESHOLD=0.95,
                                                        virt_THRESHOLD=0.95,
                                                        run_virtual_localization=False,
                                                        )
        self.localized_system.run_localization(sanity_check=True)

        # get canonical to localized change of basis
        change_basis_matrix = orb_change_basis_operator(global_rks,
                                                        self.localized_system.c_loc_occ_and_virt,
                                                        sanity_check=True)

        global_rks = self.define_rks_in_new_basis(global_rks, change_basis_matrix)
        print('orbs:', global_rks.mo_energy)
        self.subsystem_dft(global_rks)

        logger.debug("Get global DFT potential to optimize embedded calc in.")
        g_act_and_env = global_rks.get_veff(dm=(self.localized_system.dm_active + self.localized_system.dm_enviro))
        g_act = global_rks.get_veff(dm=self.localized_system.dm_active)
        dft_potential = g_act_and_env - g_act

        # get system matrices
        s_mat = global_rks.get_ovlp()
        s_half = sp.linalg.fractional_matrix_power(s_mat, 0.5)

        if self.run_mu_shift is True:
            embedded_RHF_MU = self.get_embedded_rhf(change_basis_matrix)
            enviro_projector_MU = get_enivornment_projector(self.localized_system.c_loc_occ_and_virt,
                                                            s_half,
                                                            self.localized_system.active_MO_inds,
                                                            self.localized_system.enviro_MO_inds,
                                                            return_in_ortho_basis=False)
            v_emb_MU = (self.mu_level_shift * enviro_projector_MU) + dft_potential
            hcore_std = embedded_RHF_MU.get_hcore()
            embedded_RHF_MU.get_hcore = lambda *args: hcore_std + v_emb_MU

            logger.debug("Running embedded RHF calculation.")
            embedded_RHF_MU.kernel()
            print(
                f"embedded HF energy MU_SHIFT: {embedded_RHF_MU.e_tot}, converged: {embedded_RHF_MU.converged}"
            )

            dm_active_embedded_MU = embedded_RHF_MU.make_rdm1(mo_coeff=embedded_RHF_MU.mo_coeff,
                                                              mo_occ=embedded_RHF_MU.mo_occ)

            wf_correction_MU = np.einsum("ij,ij", v_emb_MU, self.localized_system.dm_active)
            shift = global_rks.mol.nao - len(self.localized_system.enviro_MO_inds)
            frozen_enviro_orb_inds_MU = [mo_i for mo_i in range(shift, global_rks.mol.nao)]
            active_MO_inds_MU = [mo_i for mo_i in range(embedded_RHF_MU.mo_coeff.shape[1])
                                 if mo_i not in frozen_enviro_orb_inds_MU]

            # delete enviroment orbitals:
            embedded_RHF_MU.mo_coeff = embedded_RHF_MU.mo_coeff[:, active_MO_inds_MU]
            embedded_RHF_MU.mo_energy = embedded_RHF_MU.mo_energy[active_MO_inds_MU]
            embedded_RHF_MU.mo_occ = embedded_RHF_MU.mo_occ[active_MO_inds_MU]

            self.emb_rhf_etot_mu_shift = embedded_RHF_MU.e_tot + self.e_env + self.two_e_cross - wf_correction_MU
            if self.run_ccsd_emb is True:
                self.ccsd_emb_MU, e_ccsd_corr_MU = self.run_emb_CCSD(embedded_RHF_MU, frozen_orb_list=None)

                self.e_wf_ccsd_emb_MU = (self.ccsd_emb_MU.e_hf + e_ccsd_corr_MU) + self.e_env + self.two_e_cross - wf_correction_MU
                print("CCSD Energy MU shift:\n\t%s", self.e_wf_ccsd_emb_MU)

            if self.run_fci_emb is True:
                self.fci_emb_MU = self.run_emb_FCI(embedded_RHF_MU, frozen_orb_list=None)
                self.e_wf_fci_emb_MU = (self.fci_emb_MU.e_tot) + self.e_env + self.two_e_cross - wf_correction_MU
                print("FCI Energy MU shift:\n\t%s", self.e_wf_fci_emb_MU)

            self.classical_energy_MU = self.e_env + self.two_e_cross + e_nuc - wf_correction_MU
            self.molecular_ham_MU = get_molecular_hamiltonian(embedded_RHF_MU,
                                                              embedded_energy_constant=self.classical_energy_MU)
            self.n_qubits_MU = count_qubits(self.molecular_ham_MU)

        if self.run_huzinaga is True:
            embedded_RHF_HUZ = self.get_embedded_rhf(change_basis_matrix)
            enviro_projector_ortho = get_enivornment_projector(self.localized_system.c_loc_occ_and_virt,
                                                               s_half,
                                                               self.localized_system.active_MO_inds,
                                                               self.localized_system.enviro_MO_inds,
                                                               return_in_ortho_basis=True)

            # run manual HF
            s_neg_half = sp.linalg.fractional_matrix_power(s_mat, -0.5)
            (conv_flag,
             rhf_energy_HUZ,
             c_active_embedded,
             mo_embedded_energy,
             dm_active_embedded,
             huzinaga_op_std) = huzinaga_RHF(embedded_RHF_HUZ,
                                             dft_potential,
                                             enviro_projector_ortho,
                                             s_neg_half,
                                             s_half,
                                             dm_conv_tol=1e-6,
                                             dm_initial_guess=None
                                             )  # TODO: use dm_active_embedded_MU (use mu answer to initialize!)

            print(f"embedded HF energy HUZINAGA: {rhf_energy_HUZ}, converged: {conv_flag}")

            # write results to pyscf object
            hcore_std = embedded_RHF_HUZ.get_hcore()
            v_emb_HUZ = huzinaga_op_std + dft_potential
            embedded_RHF_HUZ.get_hcore = lambda *args: hcore_std + v_emb_HUZ
            embedded_RHF_HUZ.mo_coeff = c_active_embedded
            embedded_RHF_HUZ.mo_occ = embedded_RHF_HUZ.get_occ(
                mo_embedded_energy, c_active_embedded
            )
            embedded_RHF_HUZ.mo_energy = mo_embedded_energy

            print('object test :', embedded_RHF_HUZ.energy_tot(dm=embedded_RHF_HUZ.make_rdm1()))
            # print('object test :', embedded_RHF_HUZ.energy_elec(dm=embedded_RHF_HUZ.make_rdm1()))

            wf_correction_HUZ = np.einsum("ij,ij", v_emb_HUZ, self.localized_system.dm_active)

            n_act_mo = len(self.localized_system.active_MO_inds)
            n_env_mo = len(self.localized_system.enviro_MO_inds)
            frozen_enviro_orb_inds_HUZ = [i for i in range(n_act_mo, n_act_mo + n_env_mo)]

            active_MO_inds_HUZ = [mo_i for mo_i in range(embedded_RHF_HUZ.mo_coeff.shape[1])
                                  if mo_i not in frozen_enviro_orb_inds_HUZ]
            embedded_RHF_HUZ.mo_coeff = embedded_RHF_HUZ.mo_coeff[:, active_MO_inds_HUZ]
            embedded_RHF_HUZ.mo_energy = embedded_RHF_HUZ.mo_energy[active_MO_inds_HUZ]
            embedded_RHF_HUZ.mo_occ = embedded_RHF_HUZ.mo_occ[active_MO_inds_HUZ]

            self.emb_rhf_etot_HUZ = rhf_energy_HUZ + self.e_env + self.two_e_cross - wf_correction_HUZ
            if self.run_ccsd_emb is True:
                self.ccsd_emb_HUZ, e_ccsd_corr_HUZ = self.run_emb_CCSD(embedded_RHF_HUZ, frozen_orb_list=None)

                self.e_wf_ccsd_emb_HUZ = (self.ccsd_emb_HUZ.e_hf + e_ccsd_corr_HUZ) + self.e_env + self.two_e_cross - wf_correction_HUZ
                print("CCSD Energy HUZ:\n\t%s", self.e_wf_ccsd_emb_HUZ)

            if self.run_fci_emb is True:
                self.fci_emb_HUZ = self.run_emb_FCI(embedded_RHF_HUZ, frozen_orb_list=None)
                self.e_wf_fci_emb_HUZ = (self.fci_emb_HUZ.e_tot) + self.e_env + self.two_e_cross - wf_correction_HUZ
                print("FCI Energy HUZ:\n\t%s", self.e_wf_fci_emb_HUZ)

            self.classical_energy_HUZ = self.e_env + self.two_e_cross + e_nuc - wf_correction_HUZ
            self.molecular_ham_HUZ = get_molecular_hamiltonian(embedded_RHF_HUZ,
                                                               embedded_energy_constant=self.classical_energy_HUZ)
            self.n_qubits_HUZ = count_qubits(self.molecular_ham_HUZ)

        print(f"number of e- in embedded system: {2 * len(self.localized_system.active_MO_inds)}")
        print(self.localized_system.active_MO_inds)
        print(self.localized_system.enviro_MO_inds)

        return None

    def Generate_standard_H(self):
        """Build full molecular fermionic Hamiltonian (of whole system)

        Idea is to compare the number of terms to embedded Hamiltonian
        """
        if self.global_HF is None:
            mol_full = self.build_mol()
            # run Hartree-Fock
            self.global_HF = scf.RHF(mol_full)
            self.global_HF.conv_tol = self.convergence
            self.global_HF.max_memory = self.max_ram_memory
            self.global_HF.verbose = self.pyscf_print_level
            self.global_HF.kernel()

        self.full_system_molecular_ham = get_molecular_hamiltonian(self.global_HF)
        self.n_qubits_full_system = count_qubits(self.full_system_molecular_ham)

    def print_summary(self):

        if self.localized_system is None:
            raise ValueError('Need to run an embedded calculation first before summary can be made')

        print("".center(80, '*'))
        print("  Summary of Embedded Calculation".center(80))
        print("".center(80, '*'))

        print(f'global (cheap) DFT calculation {self.global_rks_total_energy}')

        if self.run_huzinaga is True:
            print("".center(80, '*'))
            print("  Huzinaga calculation".center(20))
            print(f'Total energy - active system at RHF level: {self.emb_rhf_etot_HUZ}')
            if self.run_ccsd_emb is True:
                print(f'Total energy - active system at CCSD level: {self.e_wf_ccsd_emb_HUZ}')
            if self.run_fci_emb is True:
                print(f'Total energy - active system at FCI level: {self.e_wf_fci_emb_HUZ}')

            print(f'length of huzinaga embedded fermionic Hamiltonian: {len(list(self.molecular_ham_HUZ))}')
            print(f'number of qubits required: {self.n_qubits_HUZ}')

        if self.run_mu_shift is True:
            print("".center(80, '*'))
            print("  Mu shift calculation".center(20))
            print(f'Total energy - active system at RHF level: {self.emb_rhf_etot_mu_shift}')
            if self.run_ccsd_emb is True:
                print(f'Total energy - active system at CCSD level: {self.e_wf_ccsd_emb_MU}')
            if self.run_fci_emb is True:
                print(f'Total energy - active system at FCI level: {self.e_wf_fci_emb_MU}')

            print(f'length of mu embedded fermionic Hamiltonian: {len(list(self.molecular_ham_MU))}')
            print(f'number of qubits required: {self.n_qubits_MU}')

        if int(self.perform_global_fci) + int(self.get_full_system_H)>0:
            print("".center(80, '*'))
            print("  Summary of reference Calculation".center(80))
            print("".center(80, '*'))

            if self.perform_global_fci:
                print(f'global (expensive) full FCI calculation {self.global_fci.e_tot}')
            if self.get_full_system_H:
                print(f'length of full system fermionic Hamiltonian: {len(list(self.full_system_molecular_ham))}')
                print(f'number of qubits required: {self.n_qubits_full_system}')

        return None


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
