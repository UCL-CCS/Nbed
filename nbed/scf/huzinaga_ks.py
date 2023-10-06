"""Perform Huzinaga RHF with PySCF."""

import logging
from typing import Optional, Tuple

import numpy as np
import scipy as sp
from pyscf.dft import rks, uks
from pyscf.lib import StreamObject, diis

logger = logging.getLogger(__name__)


def huzinaga_KS(
    scf_method: StreamObject,
    dft_potential: np.ndarray,
    dm_enviroment: np.ndarray,
    dm_conv_tol: float = 1e-6,
    dm_initial_guess: Optional[np.ndarray] = None,
    use_DIIS: Optional[np.ndarray] = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Manual RHF/UHF calculation that is implemented using the huzinaga operator.

    Note this function uses lowdin (symmetric) orthogonalization only! (PySCF sometimes uses meta-lowdin and NAO). Also
    the intial density matrix guess is based on the modified core Hamilotnian (containing projector and DFT potential)
    PySCF has other methods for initial guess that aren't available here. Manual guess can also be given).
    TODO: make a subclass from PySCF RHF object. Can include all this functionality there. Problems in this function
    can occur due to DIIS and other clever PySCF methods not being available.

    Args:
        scf_method (StreamObjecty):PySCF RHF/UHF object (containing info about max cycles and convergence tolerence)
        dft_potential (np.ndarray): DFT active and environment two body terms - DFT active environemnt two body term
        dm_enviroment (np.ndarray): Density matrix of the environment.
        dm_conv_tol (float): density matrix convergence tolerance
        dm_initial_guess (np.ndarray): Optional initial guess density matrix
        use_DIIS (bool): whether to use  Direct Inversion in the Iterative Subspace (DIIS) method
    Returns:
        mo_coeff_std (np.ndarray): Optimized C_matrix (columns are optimized moelcular orbtials)
        mo_energy (np.ndarray): 1D array of molecular orbital energies
        dm_mat (np.ndarray): Converged density matrix
        huzinaga_op_std (np.ndarray): Huzinaga operator in standard basis (same basis as Fock operator).
        converged (bool): Flag to indicate whether SCF has converged or not
    """
    s_mat = scf_method.get_ovlp()
    s_neg_half = sp.linalg.fractional_matrix_power(s_mat, -0.5)

    if type(scf_method) is uks.UKS:
        unrestricted = True
    elif type(scf_method) is rks.RKS:
        unrestricted = False
    else:
        logger.error("SCF method is not RKS or UKS")
        raise TypeError("SCF method is not RKS or UKS")

    if unrestricted:
        dm_env_S = np.array([dm_enviroment[0] @ s_mat, dm_enviroment[1] @ s_mat])
    else:
        dm_env_S = dm_enviroment @ s_mat

    dm_env_S = dm_enviroment @ s_mat
    # Create an initial dm if needed.
    if dm_initial_guess is None:
        fock = scf_method.get_hcore() + dft_potential

        if unrestricted:
            fds_alpha = fock[0] @ dm_env_S[0]
            fds_beta = fock[1] @ dm_env_S[1]
            huzinaga_op_std = np.array(
                [-(fds_alpha + fds_alpha.T), -(fds_beta + fds_beta.T)]
            )
        else:
            fds = fock @ dm_env_S
            huzinaga_op_std = -0.5 * (fds + fds.T)

        fock += huzinaga_op_std
        # Create the orthogonal fock operator
        fock_ortho = s_neg_half @ fock @ s_neg_half

        mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
        mo_coeff_std = s_neg_half @ mo_coeff_ortho
        mo_occ = scf_method.get_occ(mo_energy, mo_coeff_std)
        dm_initial_guess = scf_method.make_rdm1(mo_coeff=mo_coeff_std, mo_occ=mo_occ)

    dm_mat = dm_initial_guess
    converged = False
    rks_energy_prev = 0

    if use_DIIS:
        adiis = diis.DIIS()

    for i in range(scf_method.max_cycle):
        # build fock matrix
        vhf = scf_method.get_veff(dm=dm_mat)
        fock = scf_method.get_hcore() + vhf + dft_potential

        if unrestricted:
            fds_alpha = fock[0] @ dm_env_S[0]
            fds_beta = fock[1] @ dm_env_S[1]
            huzinaga_op_std = np.array(
                [-(fds_alpha + fds_alpha.T), -(fds_beta + fds_beta.T)]
            )
        else:
            fds = fock @ dm_env_S
            huzinaga_op_std = -0.5 * (fds + fds.T)

        fock += huzinaga_op_std

        if use_DIIS and (i > 1):
            # DIIS update of Fock matrix
            fock = adiis.update(fock)

        fock_ortho = s_neg_half @ fock @ s_neg_half

        mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
        mo_coeff_std = s_neg_half @ mo_coeff_ortho
        mo_occ = scf_method.get_occ(mo_energy, mo_coeff_std)

        # Create initial values for i+1 run.
        dm_mat_old = dm_mat
        dm_mat = scf_method.make_rdm1(mo_coeff=mo_coeff_std, mo_occ=mo_occ)
        # Find RKS energy
        #     rks_energy = scf_method.energy_elec(dm=dm_mat)[0]
        vhf_updated = scf_method.get_veff(dm=dm_mat)
        if unrestricted:
            rks_energy = (
                vhf_updated.ecoul
                + vhf_updated.exc
                + np.einsum(
                    "ij, ji->",
                    dm_mat[0],
                    (scf_method.get_hcore()[0] + huzinaga_op_std[0] + dft_potential[0]),
                )
                + np.einsum(
                    "ij, ji->",
                    dm_mat[1],
                    (scf_method.get_hcore()[1] + huzinaga_op_std[1] + dft_potential[1]),
                )
            )
        else:
            rks_energy = (
                vhf_updated.ecoul
                + vhf_updated.exc
                + np.einsum(
                    "ij, ji->",
                    dm_mat,
                    (scf_method.get_hcore() + huzinaga_op_std + dft_potential),
                )
            )

        # check convergence
        run_diff = np.abs(rks_energy - rks_energy_prev)
        norm_dm_diff = np.linalg.norm(dm_mat - dm_mat_old)
        if (run_diff < scf_method.conv_tol) and (norm_dm_diff < dm_conv_tol):
            converged = True
            break

        rks_energy_prev = rks_energy

    if converged is False:
        logger.warning("SCF has NOT converged.")

    return mo_coeff_std, mo_energy, dm_mat, huzinaga_op_std, converged
