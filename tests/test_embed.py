"""
File to contain tests of the embed.py script.
"""
from pathlib import Path

from openfermion import QubitOperator, count_qubits

from nbed.embed import cli, nbed

water_filepath = Path("tests/molecules/water.xyz").absolute()


def test_nbed_openfermion() -> None:
    """test nbed"""
    args = {
        "geometry": str(water_filepath),
        "n_active_atoms": 1,
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "transform": "jordan_wigner",
        "output": "openfermion",
        "convergence": 1e-6,
        "savefile": None,
        "run_ccsd_emb": True,
        "run_fci_emb": True,
        "qubits": None,
    }

    qham = nbed(**args)

    assert isinstance(qham, QubitOperator)
    return None


"""
This test is useful once tapering can force a qubit count.
"""
# def test_nbed_6_qubits() -> None:
#     """test nbed"""
#     args = {
#         "geometry": str(water_filepath),
#         "n_active_atoms": 1,
#         "basis": "STO-3G",
#         "xc_functional": "b3lyp",
#         "projector": "mu",
#         "localization": "spade",
#         "transform": "jordan_wigner",
#         "output": "openfermion",
#         "convergence": 1e-6,
#         "savefile": None,
#         "run_ccsd_emb": True,
#         "run_fci_emb": True,
#         "qubits": 4,
#     }


def test_nbed_6_qubits() -> None:
    """test nbed"""
    args = {
        "geometry": str(water_filepath),
        "n_active_atoms": 1,
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "transform": "jordan_wigner",
        "output": "openfermion",
        "convergence": 1e-6,
        "savefile": None,
        "run_ccsd_emb": True,
        "run_fci_emb": True,
        "qubits": 6,
    }

    qham = nbed(**args)

    assert count_qubits(qham) == 6
    return None


if __name__ == "__main__":
    pass
