"""
Tests for the HamiltonianBuilder class.
"""

from logging import getLogger
from pathlib import Path

import numpy as np
import pytest
import scipy as sp
from openfermion import count_qubits, get_sparse_operator
from pyscf.fci import FCI
from pyscf.gto import Mole
from pyscf.scf import RHF, UHF

from nbed.driver import NbedDriver
from nbed.ham_builder import HamiltonianBuilder

logger = getLogger(__name__)


@pytest.fixture
def restricted_scf(water_mol):
    rhf = RHF(water_mol)
    rhf.kernel()
    return rhf


@pytest.fixture
def unrestricted_scf(water_mol):
    uhf = UHF(water_mol)
    uhf.kernel()
    return uhf


@pytest.fixture
def rbuilder(restricted_scf):
    return HamiltonianBuilder(restricted_scf, 0, "jordan_wigner")


@pytest.fixture
def ubuilder(unrestricted_scf):
    return HamiltonianBuilder(unrestricted_scf, 0, "jordan_wigner")


def test_restricted_energy(restricted_scf, rbuilder) -> None:
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


def test_unrestricted_energy(charged_scf) -> None:
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
