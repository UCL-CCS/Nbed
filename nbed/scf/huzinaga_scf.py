"""Perform Huzinaga RHF with PySCF."""

import logging
from typing import Optional

import numpy as np
from pyscf import dft, scf  # type:ignore
from pyscf.lib import diis  # type:ignore
from scipy import linalg  # type:ignore

logger = logging.getLogger(__name__)


def calculate_hf_energy(
    scf_method, embedding_potential, density_matrix, vhf, huzinaga_op_occ
) -> float:
    """Calculate the Hartree-Fock Energy.

    Args:
        scf_method (scf.hf.SCF): PySCF HF method
        embedding_potential (np.ndarray): DFT embedding potential
        density_matrix (np.ndarray): Embedded region density matrix (updates each cycle)
        vhf (np.ndarray): Mean field potential
        huzinaga_op_occ (np.ndarray): Huzinaga Fock operator

    Returns:
        float: Hartree-fock energy
    """
    # Find RHF energy
    hamiltonian = (
        scf_method.get_hcore() + embedding_potential + 0.5 * vhf + huzinaga_op_occ
    )
    return np.einsum("...ij,...ji->...", hamiltonian, density_matrix)


def calculate_ks_energy(
    scf_method, embedding_potential, density_matrix, huzinaga_op_occ
) -> float:
    """Calculate the Hartree-Fock Energy.

    Args:
        scf_method (scf.hf.SCF): PySCF Kohn-sham method
        embedding_potential (np.ndarray): DFT embedding potential
        density_matrix (np.ndarray): Embedded region density matrix (updates each cycle)
        huzinaga_op_occ (np.ndarray): Huzinaga Fock operator

    Returns:
        float: Kohn-sham energy
    """
    logger.debug("Calculating Huzinaga KS energy")
    logger.debug(f"{embedding_potential.shape=}")
    logger.debug(f"{density_matrix.shape=}")
    logger.debug(f"{huzinaga_op_occ.shape=}")

    vhf_updated = scf_method.get_veff(dm=density_matrix)
    rks_energy = vhf_updated.ecoul + vhf_updated.exc
    rks_energy += np.einsum(
        "...ij, ...ji->...",
        density_matrix,
        (scf_method.get_hcore() + huzinaga_op_occ + embedding_potential),
    )
    return rks_energy


def get_huzinaga_operator(
    fock: np.ndarray, dm_occ_S: np.ndarray, dm_virt_S: np.ndarray
) -> np.ndarray:
    """Return the huzinaga operator.

    occupied :$-(S P_{occ} F + F P_{occ} S)$
    virtuall :$-(S P_{virt} F+F P_{virt} S) + 2 S P_{virt} F P_{virt} S$

    Args:
        fock (np.ndarray): The Fock operator.
        dm_occ_S (np.ndarray): The density matrix (projector onto) the occupied environment orbitals.
        dm_virt_S (np.ndarray): The density matrix (projector onto) the virtual environment orbitals.
    """
    logger.debug("Creating Huzinaga operator")
    logger.debug(f"{fock.shape=}")
    logger.debug(f"{dm_occ_S.shape=}")
    logger.debug(f"{dm_virt_S.shape=}")
    fds_occ = np.einsum("...ij,...jk->...ik", fock, dm_occ_S)
    huzinaga_op_occ = fds_occ + np.swapaxes(fds_occ, -1, -2)
    huzinaga_op_occ *= (-0.5) if fds_occ.ndim == 2 else (-1.0)

    fds_virt = np.einsum("...ij,...jk->...ik", fock, dm_virt_S)
    huzinaga_op_virt = (
        fds_virt
        + np.swapaxes(fds_virt, -1, -2)
        - 2 * np.einsum("...ij,...jk->...ik", np.swapaxes(dm_virt_S, -1, -2), fds_virt)
    )
    huzinaga_op_virt *= (-0.5) if fds_virt.ndim == 2 else (-1.0)

    logger.debug(f"{huzinaga_op_occ.shape=}")
    logger.debug(f"{huzinaga_op_virt.shape=}")
    return huzinaga_op_occ + huzinaga_op_virt


def huzinaga_scf(
    scf_method: scf.hf.SCF,
    embedding_potential: np.ndarray,
    dm_environment_occupied: np.ndarray,
    dm_environment_virtual: np.ndarray | None = None,
    dm_conv_tol: float = 1e-6,
    dm_initial_guess: Optional[np.ndarray] = None,
    use_DIIS: Optional[bool] = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Manual RHF calculation that is implemented using the huzinaga operator.

    Note this function uses lowdin (symmetric) orthogonalization only! (PySCF sometimes uses meta-lowdin and NAO). Also
    the intial density matrix guess is based on the modified core Hamilotnian (containing projector and DFT potential)
    PySCF has other methods for initial guess that aren't available here. Manual guess can also be given).
    TODO: make a subclass from PySCF RHF object. Can include all this functionality there. Problems in this function
    can occur due to DIIS and other clever PySCF methods not being available.

    Args:
        scf_method (scf.hf.SCFy):PySCF RHF object (containing info about max cycles and convergence tolerence)
        embedding_potential (np.ndarray): DFT active and environment two body terms - DFT active environemnt two body term
        dm_environment_occupied (np.ndarray): Density matrix of the environment occupied orbitals.
        dm_environment_virtual (np.ndarray | None): Density matrix of the environment virtual orbitals.
        dm_conv_tol (float): density matrix convergence tolerance.
        dm_initial_guess (np.ndarray): Optional initial guess density matrix.
        use_DIIS (bool): whether to use  Direct Inversion in the Iterative Subspace (DIIS) method
    Returns:
        mo_coeff_std (np.ndarray): Optimized C_matrix (columns are optimized moelcular orbtials)
        mo_energy (np.ndarray): 1D array of molecular orbital energies
        density_matrix (np.ndarray): Converged density matrix
        huzinaga_op_occ (np.ndarray): Huzinaga operator in standard basis (same basis as Fock operator).
        conv_flag (bool): Flag to indicate whether SCF has converged or not
    """
    logger.info("Running Huzinaga HF calculation...")
    s_mat = scf_method.get_ovlp()
    match scf_method:
        case scf.rhf.RHF() | dft.rks.RKS():
            mo_coeff_std = np.empty(s_mat.shape)
            mo_energy = np.empty(s_mat.shape[-1])
        case scf.uhf.UHF() | dft.uks.UKS():
            mo_coeff_std = np.empty((2, *s_mat.shape))
            mo_energy = np.empty((2, s_mat.shape[-1]))
        case _:
            raise ValueError("SCF method is not Restricted or Unrestricted.")

    logger.debug(f"{s_mat.shape=}")
    s_neg_half = linalg.fractional_matrix_power(s_mat, -0.5)

    adiis = diis.DIIS() if use_DIIS else None
    dm_occ_S = np.einsum("...ij,jk->...ik", dm_environment_occupied, s_mat)
    if dm_environment_virtual is not None:
        dm_virt_S = np.einsum("...ij,jk->...ik", dm_environment_virtual, s_mat)
    else:
        dm_virt_S = np.zeros(dm_occ_S.shape)

    # Create an initial dm if needed.
    if dm_initial_guess is None:
        fock = scf_method.get_hcore() + embedding_potential
        fock += get_huzinaga_operator(fock, dm_occ_S, dm_virt_S)
        logger.debug(f"{fock=}")
        # Create the orthogonal fock operator
        fock_ortho = s_neg_half @ fock @ s_neg_half
        mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
        mo_coeff_std = s_neg_half @ mo_coeff_ortho
        logger.debug(f"{mo_energy.shape=}")
        logger.debug(f"{mo_coeff_std.shape=}")
        mo_occ = scf_method.get_occ(mo_energy, mo_coeff_std)
        dm_initial_guess = scf_method.make_rdm1(mo_coeff=mo_coeff_std, mo_occ=mo_occ)  # type: ignore

    density_matrix = dm_initial_guess
    conv_flag = False
    scf_energy_prev = 0.0

    for i in range(scf_method.max_cycle):
        # build fock matrix
        vhf = scf_method.get_veff(dm=density_matrix)
        fock = scf_method.get_hcore() + embedding_potential + vhf

        huzinaga_op = get_huzinaga_operator(fock, dm_occ_S, dm_virt_S)
        fock += huzinaga_op

        if isinstance(adiis, diis.DIIS) and (i > 1):
            # DIIS update of Fock matrix
            fock = adiis.update(fock)

        fock_ortho = s_neg_half @ fock @ s_neg_half

        mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
        mo_coeff_std = s_neg_half @ mo_coeff_ortho
        mo_occ = scf_method.get_occ(mo_energy, mo_coeff_std)

        dm_mat_old = density_matrix

        density_matrix = scf_method.make_rdm1(mo_coeff=mo_coeff_std, mo_occ=mo_occ)  # type: ignore
        scf_energy: float
        if isinstance(scf_method, (dft.rks.RKS, dft.uks.UKS)):
            # Find RKS energy
            scf_energy = calculate_ks_energy(
                scf_method, embedding_potential, density_matrix, huzinaga_op
            )
        elif isinstance(scf_method, (scf.rhf.RHF, scf.uhf.UHF)):
            hamiltonian = (
                scf_method.get_hcore() + embedding_potential + 0.5 * vhf + huzinaga_op
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
        logger.warning("Huzinaga SCF has NOT converged.")

    return mo_coeff_std, mo_energy, density_matrix, huzinaga_op, conv_flag
