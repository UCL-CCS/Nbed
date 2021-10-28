"""Perform Huzinaga RHF with PySCF"""

import logging
from typing import Optional, Tuple

import numpy as np
import scipy as sp
from pyscf.lib import StreamObject

logger = logging.getLogger(__name__)


def huzinaga_RHF(
    scf_method: StreamObject,
    dft_potential: np.ndarray,
    enviro_proj_ortho_basis: np.ndarray,
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
    s_mat = scf_method.get_ovlp()
    s_half = sp.linalg.fractional_matrix_power(s_mat, 0.5)
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
        logger.warning("SCF has NOT converged.")

    e_total = rhf_energy + scf_method.energy_nuc()

    return mo_coeff_std, mo_energy, dm_mat, huzinaga_op_std

# def huzinaga_RHF(
#     scf_method: StreamObject,
#     dft_potential: np.ndarray,
#     enviro_proj_ortho_basis: np.ndarray,
#     dm_conv_tol: float = 1e-6,
#     dm_initial_guess: Optional[np.ndarray] = None,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """Manual RHF calculation that is implemented using the huzinaga operator

#     Note this function uses lowdin (symmetric) orthogonalization only! (PySCF sometimes uses meta-lowdin and NAO). Also
#     the intial density matrix guess is based on the modified core Hamilotnian (containing projector and DFT potential)
#     PySCF has other methods for initial guess that aren't available here. Manual guess can also be given).

#     TODO: make a subclass from PySCF RHF object. Can include all this functionality there. Problems in this function
#     can occur due to DIIS and other clever PySCF methods not being available.

#     Args:
#         scf_method (StreamObjecty):PySCF RHF object (containing info about max cycles and convergence tolerence)
#         dft_potential (np.ndarray): DFT active and environment two body terms - DFT active environemnt two body term
#         enviro_proj_ortho_basis (np.ndarray): Projector onto environment space (defined in orthogonal basis)
#         s_neg_half (np.ndarray): AO overlap matrix to the power of -1/2
#         s_half (np.ndarray): AO overlap matrix to the power of 1/2
#         dm_conv_tol (float): density matrix convergence tolerance
#         dm_initial_guess (np.ndarray): Optional initial guess density matrix
#     Returns:
#         mo_coeff_std (np.ndarray): Optimized C_matrix (columns are optimized moelcular orbtials)
#         mo_energy (np.ndarray): 1D array of molecular orbital energies
#         dm_mat (np.ndarray): Converged density matrix
#         huzinaga_op_std (np.ndarray): Huzinaga operator in standard basis (same basis as Fock operator).
#     """
#     s_mat = scf_method.get_ovlp()
#     s_half = sp.linalg.fractional_matrix_power(s_mat, 0.5)
#     s_neg_half = sp.linalg.fractional_matrix_power(s_mat, -0.5)

#     # Create an initial dm if needed.
#     if dm_initial_guess is None:
#         fock = scf_method.get_hcore() + dft_potential

#         # Create the orthogonal fock operator
#         fock_ortho = s_neg_half @ fock @ s_neg_half
#         huzinaga_op_ortho = -1 * (
#             fock_ortho @ enviro_proj_ortho_basis + enviro_proj_ortho_basis @ fock_ortho
#         )
#         huzinaga_op_std = s_half @ huzinaga_op_ortho @ s_half
#         fock_ortho += huzinaga_op_ortho

#         mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
#         mo_coeff_std = s_neg_half @ mo_coeff_ortho
#         mo_occ = scf_method.get_occ(mo_energy, mo_coeff_std)
#         dm_initial_guess = scf_method.make_rdm1(mo_coeff=mo_coeff_std, mo_occ=mo_occ)

#     dm_mat = dm_initial_guess
#     conv_flag = False
#     rhf_energy_prev = 0
#     for _ in range(scf_method.max_cycle):
#         # build fock matrix
#         vhf = scf_method.get_veff(dm=dm_mat)
#         fock = scf_method.get_hcore() + dft_potential + vhf

#         # else continue alg
#         fock_ortho = s_neg_half @ fock @ s_neg_half
#         huzinaga_op_ortho = -1 * (
#             fock_ortho @ enviro_proj_ortho_basis + enviro_proj_ortho_basis @ fock_ortho
#         )
#         fock_ortho += huzinaga_op_ortho

#         huzinaga_op_std = s_half @ huzinaga_op_ortho @ s_half

#         mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
#         mo_coeff_std = s_neg_half @ mo_coeff_ortho
#         mo_occ = scf_method.get_occ(mo_energy, mo_coeff_std)

#         # Create initial values for i+1 run.
#         dm_mat_old = dm_mat
#         dm_mat = scf_method.make_rdm1(mo_coeff=mo_coeff_std, mo_occ=mo_occ)
#         # Find RHF energy
#         e_core_dft = np.einsum(
#             "ij,ji->", scf_method.get_hcore() + dft_potential, dm_mat
#         )
#         e_coul = 0.5 * np.einsum("ij,ji->", vhf, dm_mat)
#         e_huz = np.einsum("ij,ji->", huzinaga_op_std, dm_mat)
#         rhf_energy = e_core_dft + e_coul + e_huz

#         # check convergence
#         run_diff = np.abs(rhf_energy - rhf_energy_prev)
#         norm_dm_diff = np.linalg.norm(dm_mat - dm_mat_old)
#         if (run_diff < scf_method.conv_tol) and (norm_dm_diff < dm_conv_tol):
#             conv_flag = True
#             break

#         rhf_energy_prev = rhf_energy

#     if conv_flag is False:
#         logger.warn("SCF has NOT converged.")

#     e_total = rhf_energy + scf_method.energy_nuc()

#     logger.info(f"Huzinaga embedding energy {e_total}.")

#     return mo_coeff_std, mo_energy, dm_mat, huzinaga_op_std
