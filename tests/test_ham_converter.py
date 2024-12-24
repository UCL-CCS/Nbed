"""Tests for the hamiltonian converter."""

from pathlib import Path

import numpy as np
import pennylane as qml
import pytest
from openfermion import QubitOperator
from pytest import raises
from qiskit.quantum_info import SparsePauliOp

from nbed.exceptions import HamiltonianConverterError
from nbed.ham_converter import HamiltonianConverter


@pytest.fixture
def intermediate_hamiltonian() -> dict:
    return {"IIII": 0.5, "IIXI": 0.25, "IIIY": 0.2}


@pytest.fixture
def hamiltonian() -> QubitOperator:
    return (
        0.5 * QubitOperator("") + 0.25 * QubitOperator("X2") + 0.2 * QubitOperator("Y3")
    )


@pytest.fixture
def qiskit_hamiltonian() -> SparsePauliOp:
    return SparsePauliOp.from_list([("IIII", 0.5), ("IIXI", 0.25), ("IIIY", 0.2)])


@pytest.fixture
def pennylane_hamiltonian() -> qml.Hamiltonian:
    return qml.Hamiltonian(
        [0.5, 0.25, 0.2],
        [
            qml.Identity(0) @ qml.Identity(1) @ qml.Identity(2) @ qml.Identity(3),
            qml.Identity(0) @ qml.Identity(1) @ qml.PauliX(2) @ qml.Identity(3),
            qml.Identity(0) @ qml.Identity(1) @ qml.Identity(2) @ qml.PauliY(3),
        ],
    )


def test_intermediate_input(intermediate_hamiltonian) -> None:
    converted_ham = HamiltonianConverter(intermediate_hamiltonian)._intermediate
    assert converted_ham == intermediate_hamiltonian

    with raises(
        HamiltonianConverterError, match=".*Input dict keys must only contain I,X,Y,Z.*"
    ):
        HamiltonianConverter({"A": 1, "1": 2})

    with raises(
        HamiltonianConverterError, match=".*All operator keys must be of equal length.*"
    ):
        HamiltonianConverter({"x": 1, "YIZ": 2})

    with raises(
        HamiltonianConverterError, match=".*All operator weights must be numbers.*"
    ):
        HamiltonianConverter({"I": "1"})


def test_file_input(intermediate_hamiltonian) -> None:
    assert (
        HamiltonianConverter("tests/test.qham")._intermediate
        == intermediate_hamiltonian
    )


def test_bad_input_type() -> None:
    error_message = (
        "Input Hamiltonian must be an openfermion.QubitOperator, dict or filepath."
    )
    with raises(TypeError, match=error_message):
        HamiltonianConverter([0, 1, 2, 3])
    with raises(TypeError, match=error_message):
        HamiltonianConverter({"a", 1, 0.1})


def test_qiskit(hamiltonian, qiskit_hamiltonian) -> None:
    converted_ham = HamiltonianConverter(hamiltonian).convert("qiskit")
    assert type(converted_ham) is SparsePauliOp
    assert np.all(converted_ham.to_matrix() - qiskit_hamiltonian.to_matrix() == 0)


def test_pennylane(hamiltonian, pennylane_hamiltonian) -> None:
    converted_ham = HamiltonianConverter(hamiltonian).convert("pennylane")
    assert type(converted_ham) is qml.Hamiltonian
    assert pennylane_hamiltonian.compare(converted_ham)


if __name__ == "__main__":
    pass
