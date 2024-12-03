"""
Tests for the HamiltonianBuilder class.
"""

from logging import getLogger
from pathlib import Path

import numpy as np
import scipy as sp
from openfermion import count_qubits, get_sparse_operator
from pyscf.fci import FCI
from pyscf.gto import Mole
from pyscf.scf import RHF, UHF

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
    "unit": "angstrom",
}

mol = Mole(**mol_args, charge=0, spin=0).build()
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
    ham = builder.build(taper=False)
    diag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(ham), k=1, which="SA")
    logger.info(f"Ground state via diagonalisation (without tapering): {diag}")
    assert np.isclose(e_fci, diag)

    builder = HamiltonianBuilder(restricted_scf, 0, "jordan_wigner")
    tapered_ham = builder.build(taper=True)
    tdiag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(tapered_ham), k=1, which="SA")
    logger.info(f"Ground state via diagonalisation (with tapering): {tdiag}")
    assert np.isclose(e_fci, tdiag)


rbuilder = HamiltonianBuilder(restricted_scf, 0, "jordan_wigner")
ubuilder = HamiltonianBuilder(unrestricted_scf, 0, "jordan_wigner")


def test_qubit_number_match() -> None:
    """
    Check that the qubit hamiltonian is working as expected.
    """

    # We're still constructing qubit hamiltonians that double the size for restricted systems!

    rham = rbuilder.build(taper=False)
    assert count_qubits(rham) == 14
    uham = ubuilder.build(taper=False)
    assert count_qubits(uham) == 14


def test_taper() -> None:

    rham = rbuilder.build(taper=True)
    assert count_qubits(rham) == 10
    # Unrestricted tapering not implemente
    # uham = ubuilder.build(taper=True)
    # assert count_qubits(uham) == 10


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


mol = Mole(**mol_args, charge=1, spin=1).build()
unrestricted_scf = UHF(mol)
unrestricted_scf.kernel()


def test_unrestricted() -> None:
    """
    Check the output hamiltonian diagonalises to fci value for an unrestricted calculation with spin and charge.
    """
    e_fci = FCI(unrestricted_scf).kernel()[0] - unrestricted_scf.energy_nuc()

    logger.info(f"FCI energy of unrestricted driver test: {e_fci}")

    builder = HamiltonianBuilder(unrestricted_scf, 0, "jordan_wigner")
    ham = builder.build(taper=False)
    diag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(ham), k=2, which="SA")

    print(diag)
    # Ground state for this charge is 2nd eigenstate
    diag = diag[1]

    logger.info(f"Ground state via diagonalisation: {diag}")
    assert np.isclose(e_fci, diag)
