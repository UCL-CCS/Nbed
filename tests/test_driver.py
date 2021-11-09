"""
File to contain tests of the driver.py script.
"""
from pathlib import Path
from nbed.driver import NbedDriver
import pytest
from openfermion.ops.representations import InteractionOperator

water_filepath = Path("tests/molecules/water.xyz").absolute()

# def test_incorrect_geometry_path() -> None:
#     """test to make sure that FileNotFoundError is thrown if invalid path to xyz geometry file is given"""
#
#     molecule = 'THIS IS NOT A PATH TO AN XYZ FILE'
#
#     args ={
#         'geometry': molecule,
#         "n_active_atoms": 1,
#         "basis": 'STO-3G',
#         "xc_functional": 'b3lyp',
#         "projector": 'mu',
#         "localization": 'spade',
#         "convergence": 1e-6,
#         "savefile": None,
#         "run_ccsd_emb": True,
#         "run_fci_emb":True
#     }
#
#     regex_match_any_string = r"[\s\S]*"
#     with pytest.raises(FileNotFoundError, match=regex_match_any_string):
#         # match will match with any printed error message
#         driver = NbedDriver(
#             geometry=args["geometry"],
#             n_active_atoms=args["n_active_atoms"],
#             basis=args["basis"],
#             xc_functional=args["xc_functional"],
#             projector=args["projector"],
#             localization=args["localization"],
#             convergence=args["convergence"],
#             savefile=args["savefile"],
#             run_ccsd_emb=args["run_ccsd_emb"],
#             run_fci_emb=args["run_fci_emb"],
#         )
#
#     return None


def test_driver_standard_xyz_file_input() -> None:
    """test to check driver works... path to xyz file given"""

    args = {
        "geometry": str(water_filepath),
        "n_active_atoms": 1,
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "convergence": 1e-6,
        "savefile": None,
        "run_ccsd_emb": True,
        "run_fci_emb": True,
    }

    driver = NbedDriver(
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
    )
    sec_quant_h = driver.molecular_ham
    assert isinstance(sec_quant_h, InteractionOperator)

    return None


def test_driver_standard_xyz_string_input() -> None:
    """test to check driver works... raw xyz string given"""
    water_xyz_raw = (
        "3\n \nH\t0.2774\t0.8929\t0.2544\nO\t0\t0\t0\nH\t0.6068\t-0.2383\t-0.7169"
    )
    args = {
        "geometry": water_xyz_raw,
        "n_active_atoms": 2,
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "convergence": 1e-6,
        "savefile": None,
        "run_ccsd_emb": False,
        "run_fci_emb": False,
    }

    driver = NbedDriver(
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
    )
    sec_quant_h = driver.molecular_ham
    assert isinstance(sec_quant_h, InteractionOperator)

    return None


if __name__ == "__main__":
    pass
