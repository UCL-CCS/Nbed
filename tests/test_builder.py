"""
Tests for the HamiltonianBuilder class.
"""

from logging import getLogger
from pathlib import Path

import numpy as np
import scipy as sp
from openfermion import QubitOperator, count_qubits, get_sparse_operator

from nbed.driver import NbedDriver
from nbed.ham_builder import HamiltonianBuilder

logger = getLogger(__name__)

mol_filepath = Path("tests/molecules/water.xyz").absolute()

args = {
    "geometry": str(mol_filepath),
    "n_active_atoms": 1,
    "basis": "STO-3G",
    # "xc_functional": "b3lyp",
    # "projector": "mu",
    # "localization": "spade",
    "convergence": 1e-8,
    # "run_ccsd_emb": False,
    # "run_fci_emb": False,
}

mol_args = {
    "atom": str(mol_filepath),
    "n_active_atoms": 1,
    "basis": "STO-3G",
    "charge": 0,
    "spin":0,
    "unit": "angstrom",
}

scf_args = {
    "conv_tol": 1e-8,
    "max_memory": 8_000,
    "verbose": 1,
    "max_cycle": 50,
}

from pyscf.gto import Mole
from pyscf.scf import RHF,UHF
from pyscf.fci import FCI

mol = Mole(**mol_args).build()
restricted_scf = RHF(mol)
restricted_scf.kernel()

unrestricted_scf = UHF(mol)
unrestricted_scf.kernel()

def test_restricted() -> None:
    """
    Use the full system to check that output hamiltonian diagonalises to fci value for a restricted calculation.
    """

    e_fci = FCI(restricted_scf).kernel()[0] - restricted_scf.energy_nuc()

    logger.info(f"FCI energy of unrestricted driver test: {e_fci}")

    builder = HamiltonianBuilder(restricted_scf, 0, "jordan_wigner")
    ham = builder.build()
    diag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(ham), k=1, which="SA")
    logger.info(f"Ground state via diagonalisation: {diag}")
    assert np.isclose(e_fci, diag)

def test_qubit_reduction() -> None:
    """
    Check that the qubit reduction is working as expected.
    """

    builder = HamiltonianBuilder(restric_driver.embedded_scf, 0, "jordan_wigner")
    ham = builder.build()
    assert count_qubits(ham) == 8

    builder = HamiltonianBuilder(restric_driver.embedded_scf, 0, "jordan_wigner")
    ham = builder.build(n_qubits=-1)
    assert count_qubits(ham) == 4


def test_unrestricted() -> None:
    """
    Check the output hamiltonian diagonalises to fci value for an unrestricted calculation with spin and charge.
    """
    e_fci = FCI(unrestricted_scf).kernel()[0] - unrestricted_scf.energy_nuc()

    logger.info(f"FCI energy of unrestricted driver test: {e_fci}")

    builder = HamiltonianBuilder(unrestricted_scf, 0, "jordan_wigner")
    ham = builder.build()
    diag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(ham), k=1, which="SA")
    logger.info(f"Ground state via diagonalisation: {diag}")
    assert np.isclose(e_fci, diag)

    # Ground state for this charge is 2nd eigenstate
    diag = diag[1]

    logger.info(f"Ground state via diagonalisation: {diag}")
    assert np.isclose(fci, diag)
