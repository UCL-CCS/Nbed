"""Tests for the hamiltonian converter."""

from pathlib import Path

import numpy as np
from openfermion import QubitOperator
from qiskit_nature.operators.second_quantization.spin_op import SpinOp

from vqe_in_dft.ham_converter import HamiltonianConverter

water_filepath = Path("tests/molecules/water.xyz").absolute()

hamiltonian = 0.5 * QubitOperator("")
hamiltonian += 0.25 * QubitOperator("X2")
hamiltonian += 0.2 * QubitOperator("Y3")

qiskit_hamiltonian = SpinOp([("IIII", 0.5), ("IIXI", 0.25), ("IIIY", 0.2)])


def test_of_to_int() -> None:
    converted_ham = HamiltonianConverter(hamiltonian).convert("qiskit")
    assert type(converted_ham) is SpinOp
    assert converted_ham.register_length == 4
    assert np.all(converted_ham.to_matrix() - qiskit_hamiltonian.to_matrix() == 0)


if __name__ == "__main__":
    pass
