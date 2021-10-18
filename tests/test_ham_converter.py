"""Tests for the hamiltonian converter."""

from vqe_in_dft import nbed
from pathlib import Path
from openfermion import QubitOperator
from vqe_in_dft.ham_converter import HamiltonianConverter

water_filepath = Path("tests/molecules/water.xyz").absolute()

hamiltonian = (
    0.5 * QubitOperator("") + 0.25 * QubitOperator("X2") + 0.2 * QubitOperator("Y3")
)
# TODO need to sort the qubit number issue.

def test_of_to_int() -> None:
    converted_ham = HamiltonianConverter(hamiltonian).convert("qiskit")
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    test_of_to_int()
