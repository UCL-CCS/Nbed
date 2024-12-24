"""
Tests for the HamiltonianBuilder class.
"""

from logging import getLogger
from pathlib import Path
import pytest

import numpy as np
import scipy as sp
from openfermion import count_qubits, get_sparse_operator
from pyscf.fci import FCI
from pyscf.gto import Mole
from pyscf.scf import RHF, UHF

from nbed.ham_builder import HamiltonianBuilder

logger = getLogger(__name__)

@pytest.fixture
def uncharged_mol(water_filepath) -> dict:
    mol_args = {
    "atom": str(water_filepath),
    "n_active_atoms": 1,
    "basis": "STO-3G",
    "unit": "angstrom",
    }
    return Mole(**mol_args, charge=0, spin=0).build()

@pytest.fixture
def restricted_scf(uncharged_mol):
    rhf = RHF(uncharged_mol)
    rhf.kernel()
    return rhf

@pytest.fixture
def unrestricted_scf(uncharged_mol):
    uhf = UHF(uncharged_mol)
    uhf.kernel()
    return uhf

@pytest.fixture
def rbuilder(restricted_scf):
    return HamiltonianBuilder(restricted_scf, 0, "jordan_wigner")

@pytest.fixture
def ubuilder(unrestricted_scf):
    return HamiltonianBuilder(unrestricted_scf, 0, "jordan_wigner")

def test_restricted(restricted_scf, rbuilder) -> None:
    """
    Use the full system to check that output hamiltonian diagonalises to fci value for a restricted calculation.
    """

    e_fci = FCI(restricted_scf).kernel()[0] - restricted_scf.energy_nuc()

    logger.info(f"FCI energy of unrestricted driver test: {e_fci}")

    ham = rbuilder.build(taper=False)
    diag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(ham), k=1, which="SA")
    logger.info(f"Ground state via diagonalisation (without tapering): {diag}")
    assert np.isclose(e_fci, diag)

    tapered_ham = rbuilder.build(taper=True)
    tdiag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(tapered_ham), k=1, which="SA")
    logger.info(f"Ground state via diagonalisation (with tapering): {tdiag}")
    assert np.isclose(e_fci, tdiag)


def test_qubit_number_match(rbuilder, ubuilder) -> None:
    """
    Check that the qubit hamiltonian is working as expected.
    """

    # We're still constructing qubit hamiltonians that double the size for restricted systems!

    rham = rbuilder.build(taper=False)
    assert count_qubits(rham) == 14
    uham = ubuilder.build(taper=False)
    assert count_qubits(uham) == 14


def test_taper(rbuilder, ubuilder) -> None:

    rham = rbuilder.build(taper=True)
    assert count_qubits(rham) == 10
    uham = ubuilder.build(taper=True)
    assert count_qubits(uham) == 10


# def test_qubit_reduction() -> None:

#     rham = rbuilder.build(n_qubits=-1, taper=False)
#     assert count_qubits(rham) == 12
#     uham = ubuilder.build(n_qubits=-1, taper=False)
#     assert count_qubits(uham) == 12


# def test_qubit_specification() -> None:
#     rham = rbuilder.build(n_qubits=8)
#     assert count_qubits(rham) == 8
#     uham = ubuilder.build(n_qubits=8, taper=False)
#     assert count_qubits(uham) == 8


# def test_contextual_subspace() -> None:
#     rham = rbuilder.build(n_qubits=8, taper=False, contextual_space=True)
#     assert count_qubits(rham) == 8


# def test_active_space_reduction() -> None:
#     rham = rbuilder.build(
#         core_indices=[], active_indices=[0, 1, 2, 3, 4, 5], taper=False
#     )
#     assert count_qubits(rham) == 12
#     uham = ubuilder.build(
#         core_indices=[], active_indices=[0, 1, 2, 3, 4, 5], taper=False
#     )
#     assert count_qubits(uham) == 12


# def test_frozen_core_validation() -> None:
#     """
#     Test the the appropriate error is raised for invalid frozen core indices.
#     """
#     rbuilder = HamiltonianBuilder(restricted_scf, 0, "jordan_wigner")

#     with raises(HamiltonianBuilderError, match="Core indices must be 1D array."):
#         rbuilder.build(core_indices=[[0, 1], [0, 1]], active_indices=[0, 1, 2, 3, 4, 5])

#     with raises(HamiltonianBuilderError, match="Active indices must be 1D array."):
#         rbuilder.build(core_indices=[0], active_indices=[[1, 2], [1, 2]])

#     with raises(
#         HamiltonianBuilderError, match="Core and active indices must not overlap."
#     ):
#         rbuilder.build(
#             core_indices=[0, 1, 2, 3, 4, 5], active_indices=[0, 1, 2, 3, 4, 5]
#         )

#     with raises(
#         HamiltonianBuilderError,
#         match="Number of core and active indices must not exceed number of orbitals.",
#     ):
#         rbuilder.build(
#             core_indices=[0, 1, 2, 3, 4, 5], active_indices=[6, 7, 8, 9, 10, 11]
#         )


@pytest.fixture
def charged_mol(water_filepath) -> Mole:
    mol_args = {
    "atom": str(water_filepath),
    "n_active_atoms": 1,
    "basis": "STO-3G",
    "unit": "angstrom",
    }
    return Mole(**mol_args, charge=1, spin=1).build()

@pytest.fixture
def charged_scf(charged_mol):
    rhf = UHF(charged_mol)
    rhf.kernel()
    return rhf

def test_unrestricted(charged_scf) -> None:
    """
    Check the output hamiltonian diagonalises to fci value for an unrestricted calculation with spin and charge.
    """
    e_fci = FCI(charged_scf).kernel()[0] - charged_scf.energy_nuc()

    logger.info(f"FCI energy of unrestricted driver test: {e_fci}")

    builder = HamiltonianBuilder(charged_scf, 0, "jordan_wigner")
    ham = builder.build(taper=False)
    diag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(ham), k=2, which="SA")

    print(diag)
    # Ground state for this charge is 2nd eigenstate
    diag = diag[1]

    logger.info(f"Ground state via diagonalisation: {diag}")
    assert np.isclose(e_fci, diag)
