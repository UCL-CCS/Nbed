"""Perform Huzinaga RHF with PySCF."""

import logging
from typing import Optional

import numpy as np
import scipy as sp
from pyscf import dft, scf
from pyscf.lib import StreamObject, diis

logger = logging.getLogger(__name__)


def calculate_hf_energy(
    scf_method, dft_potential, density_matrix, vhf, huzinaga_op_std
) -> float:
    """Calculate the Hartree-Fock Energy.

    Args:
        scf_method (StreamObject): PySCF HF method
        dft_potential (np.ndarray): DFT embedding potential
        density_matrix (np.ndarray): Embedded region density matrix (updates each cycle)
        vhf (np.ndarray): Mean field potential
        huzinaga_op_std (np.ndarray): Huzinaga Fock operator

    Returns:
        float: Hartree-fock energy
    """
    # Find RHF energy
    hamiltonian = scf_method.get_hcore() + dft_potential + 0.5 * vhf + huzinaga_op_std
    return np.einsum("...ij,...ji->...", hamiltonian, density_matrix)


def calculate_ks_energy(
    scf_method, dft_potential, density_matrix, huzinaga_op_std
) -> float:
    """Calculate the Hartree-Fock Energy.

    Args:
        scf_method (StreamObject): PySCF Kohn-sham method
        dft_potential (np.ndarray): DFT embedding potential
        density_matrix (np.ndarray): Embedded region density matrix (updates each cycle)
        huzinaga_op_std (np.ndarray): Huzinaga Fock operator

    Returns:
        float: Kohn-sham energy
    """
    vhf_updated = scf_method.get_veff(dm=density_matrix)
    rks_energy = vhf_updated.ecoul + vhf_updated.exc
    rks_energy += np.einsum(
        "...ij, ...ji->...",
        density_matrix,
        (scf_method.get_hcore() + huzinaga_op_std + dft_potential),
    )
    return rks_energy


def huzinaga_scf(
    scf_method: StreamObject,
    dft_potential: np.ndarray,
    dm_environment: np.ndarray,
    dm_conv_tol: float = 1e-6,
    dm_initial_guess: Optional[np.ndarray] = None,
    use_DIIS: Optional[np.ndarray] = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Manual RHF calculation that is implemented using the huzinaga operator.

    Note this function uses lowdin (symmetric) orthogonalization only! (PySCF sometimes uses meta-lowdin and NAO). Also
    the intial density matrix guess is based on the modified core Hamilotnian (containing projector and DFT potential)
    PySCF has other methods for initial guess that aren't available here. Manual guess can also be given).
    TODO: make a subclass from PySCF RHF object. Can include all this functionality there. Problems in this function
    can occur due to DIIS and other clever PySCF methods not being available.

    Args:
        scf_method (StreamObjecty):PySCF RHF object (containing info about max cycles and convergence tolerence)
        dft_potential (np.ndarray): DFT active and environment two body terms - DFT active environemnt two body term
        dm_environment (np.ndarray): Density matrix of the environment.
        dm_conv_tol (float): density matrix convergence tolerance.
        dm_initial_guess (np.ndarray): Optional initial guess density matrix.
        use_DIIS (bool): whether to use  Direct Inversion in the Iterative Subspace (DIIS) method
    Returns:
        mo_coeff_std (np.ndarray): Optimized C_matrix (columns are optimized moelcular orbtials)
        mo_energy (np.ndarray): 1D array of molecular orbital energies
        density_matrix (np.ndarray): Converged density matrix
        huzinaga_op_std (np.ndarray): Huzinaga operator in standard basis (same basis as Fock operator).
        conv_flag (bool): Flag to indicate whether SCF has converged or not
    """
    logger.debug("Initializising Huzinaga HF calculation")
    s_mat = scf_method.get_ovlp()
    s_neg_half = sp.linalg.fractional_matrix_power(s_mat, -0.5)

    adiis = diis.DIIS() if use_DIIS else None

    # there are many more unrestricted types than restricted
    unrestricted = not isinstance(scf_method, (scf.rhf.RHF, dft.rks.RKS))

    if unrestricted and dm_environment.ndim != 3:
        raise ValueError(
            "Unrestricted calculation requires stacked dm_environment shape (2xMxM)."
        )

    if unrestricted:
        dm_env_S = np.array([dm_environment[0] @ s_mat, dm_environment[1] @ s_mat])
    else:
        dm_env_S = dm_environment @ s_mat

    # Create an initial dm if needed.
    if dm_initial_guess is None:
        fock = scf_method.get_hcore() + dft_potential

        if unrestricted:
            fds_alpha = fock[0] @ dm_env_S[0]
            fds_beta = fock[1] @ dm_env_S[1]
            huzinaga_op_std = np.array(
                [
                    -(fds_alpha + fds_alpha.T),
                    -(fds_beta + fds_beta.T),
                ]
            )
        else:
            fds = fock @ dm_env_S
            # Cant use T as restricted with spin has split DFT potential
            huzinaga_op_std = -0.5 * (fds + np.swapaxes(fds, -1, -2))

        fock += huzinaga_op_std
        # Create the orthogonal fock operator
        fock_ortho = s_neg_half @ fock @ s_neg_half
        mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
        mo_coeff_std = s_neg_half @ mo_coeff_ortho
        mo_occ = scf_method.get_occ(mo_energy, mo_coeff_std)
        dm_initial_guess = scf_method.make_rdm1(mo_coeff=mo_coeff_std, mo_occ=mo_occ)

    density_matrix = dm_initial_guess
    conv_flag = False
    scf_energy_prev = 0

    for i in range(scf_method.max_cycle):
        # build fock matrix
        vhf = scf_method.get_veff(dm=density_matrix)
        fock = scf_method.get_hcore() + dft_potential + vhf

        if unrestricted:
            fds_alpha = fock[0] @ dm_env_S[0]
            fds_beta = fock[1] @ dm_env_S[1]
            huzinaga_op_std = np.array(
                [-(fds_alpha + fds_alpha.T), -(fds_beta + fds_beta.T)]
            )
        else:
            fds = fock @ dm_env_S
            huzinaga_op_std = -0.5 * (fds + np.swapaxes(fds, -1, -2))
        fock += huzinaga_op_std

        if use_DIIS and (i > 1):
            # DIIS update of Fock matrix
            fock = adiis.update(fock)

        fock_ortho = s_neg_half @ fock @ s_neg_half

        mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
        mo_coeff_std = s_neg_half @ mo_coeff_ortho
        mo_occ = scf_method.get_occ(mo_energy, mo_coeff_std)

        dm_mat_old = density_matrix

        density_matrix = scf_method.make_rdm1(mo_coeff=mo_coeff_std, mo_occ=mo_occ)

        if isinstance(scf_method, (dft.rks.RKS, dft.uks.UKS)):
            # Find RKS energy
            scf_energy = calculate_ks_energy(
                scf_method, dft_potential, density_matrix, huzinaga_op_std
            )
        elif isinstance(scf_method, (scf.hf.RHF, scf.uhf.UHF)):
            hamiltonian = (
                scf_method.get_hcore() + dft_potential + 0.5 * vhf + huzinaga_op_std
            )
            scf_energy = np.einsum("...ij,...ji->...", hamiltonian, density_matrix)
        else:
            raise TypeError("Cannot run Huzinaga SCF with type %s", type(scf_method))

        # check convergence
        # use max difference so that this works for unrestricted
        run_diff = np.max(np.abs(scf_energy - scf_energy_prev))
        norm_dm_diff = np.max(
            np.linalg.norm(density_matrix - dm_mat_old, axis=(-2, -1))
        )

        if (run_diff < scf_method.conv_tol) and (norm_dm_diff < dm_conv_tol):
            conv_flag = True
            logger.debug("Huzinaga SCF converged in cycle %s", i)
            break

        scf_energy_prev = scf_energy

    if conv_flag is False:
        logger.warning("SCF has NOT converged.")

    return mo_coeff_std, mo_energy, density_matrix, huzinaga_op_std, conv_flag
