"""
File to contain tests of the embed.py script.
"""
from pathlib import Path

from vqe_in_dft import nbed

water_filepath = Path("molecules/water.xyz").absolute()

# def test_nbed() -> None:
#     q_ham, e_classical = nbed(
#         geometry=str(water_filepath),
#         active_atoms=2,
#         basis="sto-3g",
#         xc_functional="b3lyp",
#         output="openfermion",
#         convergence=1e-8,
#     )
#     assert e_classical == -3.5605837557207654
#     assert len(q_ham.terms) == 193
