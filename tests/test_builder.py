"""
Tests for the HamiltonianBuilder class.
"""

from asyncio.log import logger
from nbed.driver import NbedDriver
from nbed.ham_builder import HamiltonianBuilder
from openfermion import QubitOperator, get_sparse_operator, count_qubits
import scipy as sp
from pathlib import Path
from logging import getLogger
import numpy as np

logger = getLogger(__name__)

mol_filepath = Path("tests/molecules/water.xyz").absolute()

args = {
    "geometry": str(mol_filepath),
    "n_active_atoms": 1,
    "basis": "STO-3G",
    "xc_functional": "b3lyp",
    "projector": "mu",
    "localization": "spade",
    "convergence": 1e-8,
    "savefile": None,
    "run_ccsd_emb": False,
    "run_fci_emb": False,
}

def test_restricted_result() -> None:
    """
    Use the full system to check that output hamiltonian diagonalises to fci value for a restricted calculation.
    """
    unrestric_driver = NbedDriver(
        geometry=args["geometry"],
        n_active_atoms=args["n_active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        projector=args["projector"],
        localization=args["localization"],
        convergence=args["convergence"],
        savefile=args["savefile"],
        run_ccsd_emb=args["run_ccsd_emb"],
        run_fci_emb=args["run_fci_emb"],
        force_unrestricted=False,
    )

    fci = unrestric_driver._global_fci.e_tot - unrestric_driver._global_ks.energy_nuc()
    logger.info(f"FCI energy of unrestricted driver test: {fci}")

    builder = HamiltonianBuilder(unrestric_driver._global_ks, 0, "jordan_wigner")
    ham = builder.build()
    diag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(ham), k=1, which="SA")
    logger.info(f"Ground state via diagonalisation: {diag}")
    assert(np.isclose(fci, diag))

def test_unrestricted_result() -> None:
    """
    Use the full system to check that output hamiltonian diagonalises to fci value for an unrestricted calculation.
    """
    unrestric_driver = NbedDriver(
        geometry=args["geometry"],
        charge=1,
        spin=1,
        n_active_atoms=args["n_active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        projector=args["projector"],
        localization=args["localization"],
        convergence=args["convergence"],
        savefile=args["savefile"],
        run_ccsd_emb=args["run_ccsd_emb"],
        run_fci_emb=args["run_fci_emb"],
        force_unrestricted=True,
    )

    fci = unrestric_driver._global_fci.e_tot - unrestric_driver._global_ks.energy_nuc()
    logger.info(f"FCI energy of unrestricted driver test: {fci}")

    # We need to double up the size of the hcore
    old_hcore = unrestric_driver._global_ks.get_hcore()
    unrestric_driver._global_ks.get_hcore = lambda *args: np.array([old_hcore, old_hcore])

    builder = HamiltonianBuilder(unrestric_driver._global_ks, 0, "jordan_wigner")
    ham = builder.build()
    diag, _ = sp.sparse.linalg.eigsh(get_sparse_operator(ham), k=1, which="SA")
    logger.info(f"Ground state via diagonalisation: {diag}")
    assert(np.isclose(fci, diag))