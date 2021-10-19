"""
File to contain tests of the embed.py script.
"""
from pathlib import Path

import numpy as np
import openfermion

from vqe_in_dft import nbed

water_filepath = Path("tests/molecules/water.xyz").absolute()


def test_openfermion_output() -> None:
    q_ham, e_classical = nbed(
        geometry=str(water_filepath),
        active_atoms=2,
        basis="sto-3g",
        xc_functional="b3lyp",
        output="openfermion",
        convergence=1e-8,
    )
    assert type(q_ham) is openfermion.QubitOperator
    assert len(q_ham.terms) == 1079
    assert np.isclose(q_ham.constant, -45.42234047466274)
    assert np.isclose(e_classical, -3.5605837557207654)

if __name__ == "__main__":
    pass
