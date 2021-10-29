"""
File to contain tests of the driver.py script.
"""
from pathlib import Path
from nbed.driver import NbedDriver
import pytest

water_filepath = Path("tests/molecules/water.xyz").absolute()

def test_incorrect_molecule_name() -> None:
    """test to make sure that ValueError is thrown if pubchem cannot find molecule"""

    molecule = 'THIS IS NOT A MOLECULE'

    args ={
        'molecule': molecule,
        "n_active_atoms": 1,
        "basis": 'STO-3G',
        "xc_functional": 'b3lyp',
        "projector": 'mu',
        "localization": 'spade',
        "convergence": 1e-6,
        "savefile": None,
        "run_ccsd_emb": True,
        "run_fci_emb":True
    }

    regex_match_any_string = r"[\s\S]*"
    with pytest.raises(ValueError, match=regex_match_any_string):
        qham = NbedDriver(
            molecule=args["molecule"],
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

    return None

if __name__ == "__main__":
    pass
