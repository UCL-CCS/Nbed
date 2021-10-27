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


if __name__ == "__main__":
    from .utils import cli

    cli()
