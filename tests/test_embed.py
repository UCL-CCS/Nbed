"""
File to contain tests of the embed.py script.
"""

from pathlib import Path
import pytest

from openfermion import QubitOperator

from nbed.embed import nbed



@pytest.fixture
def args(water_filepath) -> dict:
    return {
        "geometry": str(water_filepath),
        "n_active_atoms": 1,
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "transform": "jordan_wigner",
        "output": "openfermion",
        "convergence": 1e-6,
        "savefile": "save_tests/",
        "run_ccsd_emb": True,
        "run_fci_emb": True,
        "qubits": None,
    }

def test_nbed_openfermion(args) -> None:
    """test nbed"""

    assert isinstance(nbed(**args), QubitOperator)


if __name__ == "__main__":
    pass
