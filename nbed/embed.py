"""Main embedding functionality."""

import logging
import warnings
from functools import cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyscf
import scipy as sp
from cached_property import cached_property
from openfermion import QubitOperator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import bravyi_kitaev, bravyi_kitaev_tree, jordan_wigner
from pyscf import ao2mo, cc, fci, gto, lib, scf
from pyscf.dft import numint
from pyscf.dft.rks import get_veff as rks_get_veff
from pyscf.lib import StreamObject

from nbed.exceptions import NbedConfigError
from nbed.localisation import (
    Localizer,
    PySCFLocalizer,
    SpadeLocalizer,
    _local_basis_transform,
)
from nbed.utils import setup_logs

logger = logging.getLogger(__name__)
setup_logs()

def rks_veff(
    pyscf_RKS: StreamObject,
    unitary_rot: np.ndarray,
    dm: np.ndarray = None,
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
    _, _, vxc = numint.nr_vxc(pyscf_RKS.mol, pyscf_RKS.grids, pyscf_RKS.xc, dm)

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

def orthogonal_enviro_projector(
    local_sys: Localizer,
    s_half: np.ndarray,
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
    c_loc_ortho = s_half @ local_sys.c_loc_occ_and_virt

    # 2. Define projector that projects MO orbs of subsystem B onto themselves and system A onto zero state!
    #    (do this in orthongoal basis!)
    #    note we only take MO environment indices!
    ortho_proj = np.einsum(
        "ik,jk->ij",
        c_loc_ortho[:, local_sys.enviro_MO_inds],
        c_loc_ortho[:, local_sys.enviro_MO_inds],
    )

    # # env projected onto itself
    # logger.info(
    #     f"""Are subsystem B (env) projected onto themselves in ORTHO basis: {
    #         np.allclose(ortho_proj @ c_loc_ortho[:, enviro_MO_inds],
    #         c_loc_ortho[:, enviro_MO_inds])}"""
    # )

    # # act projected onto zero vec
    # logger.info(
    #     f"""Is subsystem A traced out  in ORTHO basis?: {
    #         np.allclose(ortho_proj @ c_loc_ortho[:, active_MO_inds],
    #         np.zeros_like(c_loc_ortho[:, active_MO_inds]))}"""
    # )
    return ortho_proj


def get_molecular_hamiltonian(
    scf_method: StreamObject,
) -> InteractionOperator:
    """Returns second quantized fermionic molecular Hamiltonian

    Args:
        scf_method (StreamObject): A pyscf self-consistent method.
        frozen_indices (list[int]): A list of integer indices of frozen moleclar orbitals.

    Returns:
        molecular_hamiltonian (InteractionOperator): fermionic molecular Hamiltonian
    """

    # C_matrix containing orbitals to be considered
    # if there are any environment orbs that have been projected out... these should NOT be present in the
    # scf_method.mo_coeff array (aka columns should be deleted!)
    c_matrix_active = scf_method.mo_coeff
    n_orbs = c_matrix_active.shape[1]

    # one body terms
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

    print(f"core constant: {core_constant}")

    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
        one_body_ints_reduced, two_body_ints_reduced
    )

    molecular_hamiltonian = InteractionOperator(
        core_constant, one_body_coefficients, 0.5 * two_body_coefficients
    )

    return molecular_hamiltonian


def get_qubit_hamiltonian(
    molecular_ham: InteractionOperator, transformation: str = "jordan_wigner"
) -> QubitOperator:
    """Takes in a second quantized fermionic Hamiltonian and returns a qubit hamiltonian under defined fermion
       to qubit transformation.

    Args:
        molecular_ham (InteractionOperator): A pyscf self-consistent method.
        transformation (str): Type of fermion to qubit mapping (jordan_wigner, bravyi_kitaev, bravyi_kitaev_tree)

    Returns:
        Qubit_Hamiltonian (QubitOperator): Qubit hamiltonian of molecular Hamiltonian (under specified fermion mapping)
    """

    transforms = {
        "jordan_wigner": jordan_wigner,
        "bravyi_kitaev": bravyi_kitaev,
        "bravyi_kitaev_tree": bravyi_kitaev_tree,
    }
    try:
        qubit_ham = transforms[transformation](molecular_ham)
    except KeyError:
        raise NbedConfigError(
            "No Qubit Hamiltonian mapping with name %s", transformation
        )

    return qubit_ham


def huzinaga_RHF(
    scf_method: StreamObject,
    dft_potential: np.ndarray,
    enviro_proj_ortho_basis: np.ndarray,
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
    s_mat = s_half @ s_half
    s_neg_half = sp.linalg.fractional_matrix_power(s_mat, -0.5)

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
        e_core_dft = np.einsum(
            "ij,ji->", scf_method.get_hcore() + dft_potential, dm_mat
        )
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


if __name__ == "__main__":
    from .utils import cli

    cli()
