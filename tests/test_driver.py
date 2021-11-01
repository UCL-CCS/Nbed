"""
File to contain tests of the driver.py script.
"""
from pathlib import Path
from nbed.driver import NbedDriver
import pytest
from nbed.exceptions import NbedConfigError
from openfermion.ops.representations import InteractionOperator

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
    with pytest.raises(NbedConfigError, match=regex_match_any_string):
        # match will match with any printed error message
        driver = NbedDriver(
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

def test_driver_standard_xyz_file_input() -> None:
    """test to check driver works... path to xyz file given"""

    args ={
        'molecule': str(water_filepath),
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

    driver = NbedDriver(
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
    sec_quant_h = driver.molecular_ham
    assert isinstance(sec_quant_h, InteractionOperator)

    return None

def test_driver_standard_mol_name_input() -> None:
    """test to check driver works... molecular name given and structure found on pubchem"""

    args ={
        'molecule': 'H2O',
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

    driver = NbedDriver(
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
    sec_quant_h = driver.molecular_ham
    assert isinstance(sec_quant_h, InteractionOperator)

    return None

if __name__ == "__main__":
    pass
