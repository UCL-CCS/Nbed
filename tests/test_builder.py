"""Tests for the HamiltonianBuilder class."""

from logging import getLogger

import numpy as np
import pytest
import scipy as sp
from openfermion import count_qubits, get_sparse_operator
from openfermion.ops import InteractionOperator
from openfermion.transforms.opconversions import jordan_wigner
from pyscf.fci import FCI
from pyscf.gto import Mole
from pyscf.scf import ROHF, UHF

from nbed.ham_builder import HamiltonianBuilder, reduce_virtuals

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
    rhf = ROHF(uncharged_mol)
    rhf.kernel()
    return rhf


@pytest.fixture
def unrestricted_scf(uncharged_mol):
    uhf = UHF(uncharged_mol)
    uhf.kernel()
    return uhf


@pytest.fixture
def rbuilder(restricted_scf):
    return HamiltonianBuilder(restricted_scf, 0).build()


@pytest.fixture
def ubuilder(unrestricted_scf):
    return HamiltonianBuilder(unrestricted_scf, 0).build()


def test_restricted_groundstate(restricted_scf, rbuilder) -> None:
    """Use the full system to check that output hamiltonian diagonalises to fci value for a restricted calculation."""
    e_fci = FCI(restricted_scf).kernel()[0] - restricted_scf.energy_nuc()

    logger.info(f"FCI energy of unrestricted driver test: {e_fci}")

    const, ones, twos = rbuilder
    intop = InteractionOperator(const, ones, twos)
    qham = jordan_wigner(intop)

    assert count_qubits(qham) == 14
    diag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(qham), k=1, which="SA")
    logger.info(f"Ground state via diagonalisation: {diag}")
    assert np.isclose(e_fci, diag)

def test_unrestricted_groundstate(unrestricted_scf, ubuilder) -> None:
    """Use the full system to check that output hamiltonian diagonalises to fci value for a restricted calculation."""
    e_fci = FCI(unrestricted_scf).kernel()[0] - unrestricted_scf.energy_nuc()

    logger.info(f"FCI energy of unrestricted driver test: {e_fci}")

    const, ones, twos = ubuilder
    intop = InteractionOperator(const, ones, twos)
    qham = jordan_wigner(intop)

    assert count_qubits(qham) == 14
    diag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(qham), k=1, which="SA")
    logger.info(f"Ground state via diagonalisation: {diag}")
    assert np.isclose(e_fci, diag)


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


def test_unrestricted_charged_groundstate(charged_scf) -> None:
    """Check the output hamiltonian diagonalises to fci value for an unrestricted calculation with spin and charge."""
    e_fci = FCI(charged_scf).kernel()[0] - charged_scf.energy_nuc()

    logger.info(f"FCI energy of unrestricted driver test: {e_fci}")

    builder = HamiltonianBuilder(charged_scf)
    const, ones, twos = builder.build()
    intop = InteractionOperator(const, ones, twos)
    qham = jordan_wigner(intop)

    diag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(qham), k=2, which="SA")

    logger.info(f"Ground state via diagonalisation: {diag}")
    # Ground state for this charge is 2nd eigenstate
    assert np.isclose(e_fci, diag[1])

def test_reduce_virtuals(restricted_scf, unrestricted_scf):
    reduced_restricted = reduce_virtuals(restricted_scf, 1)
    reduced_unrestricted = reduce_virtuals(unrestricted_scf, 1)

    assert  reduced_restricted.mo_coeff.shape[-1] == reduced_unrestricted.mo_coeff.shape[-1] == 6
    assert  np.all(reduced_restricted.mo_occ == np.sum(reduced_unrestricted.mo_occ, axis=0))

    with pytest.raises(ValueError) as excinfo:
        reduce_virtuals(restricted_scf, 7)

    assert "more than exist" in str(excinfo)

    assert np.all(restricted_scf.mo_coeff == reduce_virtuals(restricted_scf, 0).mo_coeff)
    assert np.all(restricted_scf.mo_coeff == reduce_virtuals(restricted_scf, 0).mo_coeff)
