"""Perform Huzinaga UKS with PySCF."""

import logging
from typing import Optional, Tuple

import numpy as np
import scipy as sp
from pyscf.lib import StreamObject, diis

logger = logging.getLogger(__name__)


def huzinaga_UKS(
    scf_method: StreamObject,
    dft_potential: np.ndarray,
    dm_enviroment: np.ndarray,
    dm_conv_tol: float = 1e-6,
    dm_initial_guess: Optional[np.ndarray, np.ndarray] = None,
    use_DIIS: Optional[np.ndarray] = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Manual RHF calculation that is implemented using the huzinaga operator.

    Note this function uses lowdin (symmetric) orthogonalization only! (PySCF sometimes uses meta-lowdin and NAO). Also
    the intial density matrix guess is based on the modified core Hamilotnian (containing projector and DFT potential)
    PySCF has other methods for initial guess that aren't available here. Manual guess can also be given).
    TODO: make a subclass from PySCF RHF object. Can include all this functionality there. Problems in this function
    can occur due to DIIS and other clever PySCF methods not being available.

    Args:
        scf_method (StreamObjecty):PySCF UKS object (containing info about max cycles and convergence tolerence)
        dft_potential (np.ndarray): DFT active and environment two body terms - DFT active environemnt two body term
        dm_enviroment (np.ndarray): Density matrix of the environment.
        dm_conv_tol (float): density matrix convergence tolerance
        dm_initial_guess (np.ndarray, np.ndarray): Optional initial guess density matrix
        use_DIIS (bool): whether to use  Direct Inversion in the Iterative Subspace (DIIS) method
    Returns:
        mo_coeff_std_alpha (np.ndarray): Optimized C_matrix for spin up electrons (columns are optimized moelcular orbtials)
        mo_coeff_std_beta (np.ndarray): Optimized C_matrix for spin down electrons (columns are optimized moelcular orbtials)
        mo_energy_alpha (np.ndarray): 1D array of molecular orbital energies for spin up electrons
        mo_energy_beta (np.ndarray): 1D array of molecular orbital energies for spin down electrons
        dm_mat_alpha (np.ndarray): Converged density matrix for spin up electrons
        dm_mat_beta (np.ndarray): Converged density matrix for spin down electrons
        huzinaga_op_std (np.ndarray): Huzinaga operator in standard basis (same basis as Fock operator).
        converged (bool): Flag to indicate whether SCF has converged or not
    """
    s_mat = scf_method.get_ovlp()
    s_neg_half = sp.linalg.fractional_matrix_power(s_mat, -0.5)

    dm_env_S = dm_enviroment @ s_mat
    # Create an initial dm if needed.
    if dm_initial_guess is None:
        fock = scf_method.get_hcore() + dft_potential

        fds = fock @ dm_env_S
        huzinaga_op_std = -0.5 * (fds + fds.T)

        fock += huzinaga_op_std
        # Create the orthogonal fock operator
        fock_ortho = s_neg_half @ fock @ s_neg_half
        
        mo_energy, mo_coeff_ortho = np.linalg.eigh(fock_ortho)
        ind = np.argsort(mo_energy)
        mo_coeff_ortho = mo_coeff_ortho[ind]
        mo_coeff_std = s_neg_half @ mo_coeff_ortho

        mo_coeff_alpha_occ = mo_coeff_std[:,:n_alpha]
        mo_coeff_beta_occ = mo_coeff_std[:,:n_beta]

        #Generate initial density matrix for spin up and down e
        dm_initial_guess_alpha = mo_coeff_alpha_occ @ mo_coeff_alpha_occ.T
        dm_initial_guess_beta = mo_coeff_beta_occ @ mo_coeff_beta_occ.T

    dm_mat_alpha = dm_initial_guess_alpha
    dm_mat_beta = dm_initial_guess_beta
    converged = False
    rks_energy_prev = 0

    if use_DIIS:
        adiis = diis.DIIS()

    for i in range(scf_method.max_cycle):

        # build fock matrix
        vhf_alpha, vhf_beta = scf_method.get_veff(dm=[dm_mat_alpha, dm_mat_beta])
        fock_alpha = 0.5 * scf_method.get_hcore() + vhf_alpha + dft_potential
        fock_beta = 0.5 * scf_method.get_hcore() + vhf_beta + dft_potential

        # projector
        fds_alpha = fock_alpha @ dm_env_S
        huzinaga_op_std_alpha = -0.5 * (fds_alpha + fds_alpha.T)
        fds_beta = fock_beta @ dm_env_S
        huzinaga_op_std_beta = -0.5 * (fds_beta + fds_beta.T)

        fock_alpha += huzinaga_op_std_alpha
        fock_beta += huzinaga_op_std_beta

        if use_DIIS and (i > 1):
            # DIIS update of Fock matrix
            fock_alpha = adiis.update(fock_alpha)
            fock_beta = adiis.update(fock_beta)

        fock_ortho_alpha = s_neg_half @ fock_alpha @ s_neg_half
        fock_ortho_beta = s_neg_half @ fock_beta @ s_neg_half

        # Create the orthogonal fock operator
        mo_energy_alpha, mo_coeff_ortho_alpha = np.linalg.eigh(fock_ortho_alpha)
        ind_alpha = np.argsort(mo_energy_alpha)
        mo_coeff_ortho_alpha = mo_coeff_ortho_alpha[ind_alpha]
        mo_coeff_std_alpha = s_neg_half @ mo_coeff_ortho_alpha

        mo_energy_beta, mo_coeff_ortho_beta = np.linalg.eigh(fock_ortho_beta)
        ind_beta = np.argsort(mo_energy_beta)
        mo_coeff_ortho_beta = mo_coeff_ortho_beta[ind_beta]
        mo_coeff_std_beta = s_neg_half @ mo_coeff_ortho_beta

        mo_coeff_alpha_occ = mo_coeff_std_alpha[:,:n_alpha]
        mo_coeff_beta_occ = mo_coeff_std_beta[:,:n_beta]

        # Create initial values for i+1 run.
        dm_mat_alpha_old = dm_mat_alpha.copy()
        dm_mat_beta_old = dm_mat_beta.copy()
        dm_mat_alpha = mo_coeff_alpha_occ @ C_alpmo_coeff_alpha_occha_occ.T
        dm_mat_beta = mo_coeff_beta_occ @ mo_coeff_beta_occ.T

        # Find UKS energy
        uks_energy = scf_method.energy_elec(dm=[dm_mat_alpha, dm_mat_beta])[0]

        # check convergence
        run_diff = np.abs(rks_energy - rks_energy_prev)
        norm_dm_diff = np.linalg.norm(dm_mat_alpha - dm_mat_alpha_old)
        if (run_diff < scf_method.conv_tol) and (norm_dm_diff < dm_conv_tol):
            converged = True
            break

        rks_energy_prev = rks_energy

    if converged is False:
        logger.warning("SCF has NOT converged.")

    return mo_coeff_std_alpha, mo_coeff_std_beta, mo_energy_alpha, mo_energy_beta, dm_mat_alpha, dm_mat_beta, huzinaga_op_std_alpha, huzinaga_op_std_beta, converged
