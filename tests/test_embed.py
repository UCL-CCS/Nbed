"""
File to contain tests of the embed.py script.
"""
from pathlib import Path

from openfermion import QubitOperator, count_qubits

from nbed.embed import nbed, cli

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

    qham = nbed(
        geometry=args["geometry"],
        n_active_atoms=args["n_active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        projector=args["projector"],
        localization=args["localization"],
        transform=args["transform"],
        output=args["output"],
        convergence=args["convergence"],
        savefile=args["savefile"],
        run_ccsd_emb=args["run_ccsd_emb"],
        run_fci_emb=args["run_fci_emb"],
        qubits=args["qubits"],
    )

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

    qham = nbed(
        geometry=args["geometry"],
        n_active_atoms=args["n_active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        projector=args["projector"],
        localization=args["localization"],
        transform=args["transform"],
        output=args["output"],
        convergence=args["convergence"],
        savefile=args["savefile"],
        run_ccsd_emb=args["run_ccsd_emb"],
        run_fci_emb=args["run_fci_emb"],
        qubits=args["qubits"],
    )

    assert count_qubits(qham) == 6
    return None

if __name__ == "__main__":
    pass
