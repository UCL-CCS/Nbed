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

from .driver import NbedDriver
from .exceptions import NbedConfigError
from .utils import parse, setup_logs

logger = logging.getLogger(__name__)


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


def cli() -> None:
    """CLI Interface."""
    setup_logs()
    args = parse()
    driver = NbedDriver(
        geometry=args["geometry"],
        n_active_atoms=args["active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        output=args["output"],
        localisation=args["localisation"],
        convergence=args["convergence"],
        run_ccsd=args["ccsd"],
        qubits=args["qubits"],
        savefile=args["savefile"],
    )
    embedded_rhf = driver.embed()

    print("Qubit Hamiltonian:")
    print(qham)
    print(f"Classical Energy (Ha): {e_classical}")


if __name__ == "__main__":
    cli()
