"""Perform Huzinaga RHF with PySCF."""

import logging
from typing import Optional, Tuple

import numpy as np
import pytest
import scipy as sp
from pyscf import dft, scf
from pyscf.lib import StreamObject, diis

logger = logging.getLogger(__name__)


def _huzinaga_fock_operator(
    scf_method: StreamObject,
    dft_potential: np.ndarray,
    vhf: np.ndarray,
    dm_enviroment: np.ndarray,
    adiis: Optional[diis.DIIS],
) -> np.ndarray:
    logger.debug("Calculating Huzinaga operator")
    fock = scf_method.get_hcore() + dft_potential + vhf
    logger.debug(f"{fock.shape=}")
    dm_env_S = dm_enviroment @ scf_method.get_ovlp()
    logger.debug(f"{dm_env_S.shape=}")

    fock = fock @ dm_env_S
    logger.debug(f"{fock.shape=}")

    if fock.ndim == 3 and fock.shape[0] == 2:
        logger.debug("Calculatign unrestricted density matrix")
        huzinaga_op_std = np.array(
            [
                -(fock[0] + fock[0].T),
                -(fock[1] + fock[1].T),
            ]
        )
    else:
        logger.debug("Calculating restricted density matrix")
        huzinaga_op_std = -0.5 * (fock + fock.T)

    fock += huzinaga_op_std

    if isinstance(diis, diis.DIIS):
        fock = adiis.update(fock)

    # Create the orthogonal fock operator
    return huzinaga_op_std, fock


def calculate_hf_energy(scf_method, dft_potential, density_matrix, huzinaga_op_std):
    # Find RHF energy
    hcore = scf_method.get_hcore()
    vhf = scf_method.get_veff(dm=density_matrix)

    e_core_dft = np.einsum("...ij,...ji->...", hcore + dft_potential, density_matrix)

    e_coul = 0.5 * np.einsum("...ij,...ji->...", vhf, density_matrix)
    e_huz = np.einsum("...ij,...ji->...", huzinaga_op_std, density_matrix)
    return e_core_dft + e_coul + e_huz


def calculate_ks_energy(scf_method, dft_potential, density_matrix, huzinaga_op_std):
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
    dm_enviroment: np.ndarray,
    dm_conv_tol: float = 1e-6,
    dm_initial_guess: Optional[np.ndarray] = None,
    use_DIIS: Optional[np.ndarray] = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Manual RHF calculation that is implemented using the huzinaga operator.

    Note this function uses lowdin (symmetric) orthogonalization only! (PySCF sometimes uses meta-lowdin and NAO). Also
    the intial density matrix guess is based on the modified core Hamilotnian (containing projector and DFT potential)
    PySCF has other methods for initial guess that aren't available here. Manual guess can also be given).
    TODO: make a subclass from PySCF RHF object. Can include all this functionality there. Problems in this function
    can occur due to DIIS and other clever PySCF methods not being available.

    Args:
        scf_method (StreamObjecty):PySCF RHF object (containing info about max cycles and convergence tolerence)
        dft_potential (np.ndarray): DFT active and environment two body terms - DFT active environemnt two body term
        dm_enviroment (np.ndarray): Density matrix of the environment.
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
    if unrestricted and dm_enviroment.shape[0] != 2:
        raise ValueError(
            "Unrestricted calculation requires stacked dm_environment shape (2xMxM)."
        )

    # Create an initial dm if needed.
    if dm_initial_guess is None:
        fock = scf_method.get_hcore() + dft_potential
        dm_env_S = dm_enviroment @ s_mat
        fds = fock @ dm_env_S

        huzinaga_op_std, fock = _huzinaga_fock_operator(
            scf_method, dft_potential, 0, dm_enviroment, None
        )
        fock_ortho = s_neg_half @ fock @ s_neg_half
        mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
        mo_coeff_std = s_neg_half @ mo_coeff_ortho
        mo_occ = scf_method.get_occ(mo_energy, mo_coeff_std)
        dm_initial_guess = scf_method.make_rdm1(mo_coeff=mo_coeff_std, mo_occ=mo_occ)

    density_matrix = dm_initial_guess
    conv_flag = False
    rhf_energy_prev = 0

    for i in range(scf_method.max_cycle):
        # build fock matrix
        vhf = scf_method.get_veff(dm=density_matrix)
        dm_mat_old = density_matrix

        if i == 0:
            huzinaga_op_std, fock = _huzinaga_fock_operator(
                scf_method, dft_potential, vhf, dm_enviroment, None
            )
        else:
            # DIIS update of Fock matrix
            huzinaga_op_std, fock = _huzinaga_fock_operator(
                scf_method, dft_potential, vhf, dm_enviroment, adiis
            )

        fock_ortho = s_neg_half @ fock @ s_neg_half

        mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
        mo_coeff_std = s_neg_half @ mo_coeff_ortho
        mo_occ = scf_method.get_occ(mo_energy, mo_coeff_std)

        density_matrix = scf_method.make_rdm1(mo_coeff=mo_coeff_std, mo_occ=mo_occ)

        if isinstance(scf_method, (dft.rks.RKS, dft.uks.UKS)):
            # Find RKS energy
            scf_energy = calculate_ks_energy(
                scf_method, dft_potential, density_matrix, huzinaga_op_std
            )
        elif isinstance(scf_method, (scf.rhf.RHF, scf.uhf.UHF)):
            # Find RHF energy
            scf_energy = calculate_hf_energy(
                scf_method,
                dft_potential,
                density_matrix,
                huzinaga_op_std,
            )

        # check convergence
        # use max difference so that this works for unrestricted
        run_diff = np.max(np.abs(scf_energy - rhf_energy_prev))
        norm_dm_diff = np.max(
            np.linalg.norm(density_matrix - dm_mat_old, axis=(-2, -1))
        )

        if (run_diff < scf_method.conv_tol) and (norm_dm_diff < dm_conv_tol):
            conv_flag = True
            break

        rhf_energy_prev = scf_energy

    if conv_flag is False:
        logger.warning("SCF has NOT converged.")

    return mo_coeff_std, mo_energy, density_matrix, huzinaga_op_std, conv_flag
