"""
File to contain tests of the embed.py script.
"""
from pathlib import Path

import numpy as np
from nbed.embed import nbed
from openfermion import QubitOperator
from nbed.localizers import Localizer

water_filepath = Path("tests/molecules/water.xyz").absolute()

def test_nbed_openfermion() -> None:
    """test nbed"""
    args ={
        'molecule': str(water_filepath),
        "n_active_atoms": 1,
        "basis": 'STO-3G',
        "xc_functional": 'b3lyp',
        "projector": 'mu',
        "localization": 'spade',
        "transform": 'jordan_wigner',
        "output": 'openfermion',
        "convergence": 1e-6,
        "savefile": None,
        "run_ccsd_emb": True,
        "run_fci_emb":True
    }

    qham = nbed(
        molecule=args["molecule"],
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
    )

    assert isinstance(qham, QubitOperator)
    return None

if __name__ == "__main__":
    pass
