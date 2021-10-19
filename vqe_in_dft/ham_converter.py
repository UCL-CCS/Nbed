"""File to contain the qubit hamiltonian format."""

import logging
from functools import cached_property
from typing import Dict, List

import numpy as np
import openfermion
import pennylane as qml
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import jordan_wigner
from pennylane import Identity, PauliX, PauliY, PauliZ
from pyscf import ao2mo
from pyscf.lib import StreamObject
from qiskit.opflow import I, X, Y, Z
from qiskit_nature.operators.second_quantization import SpinOp

logger = logging.getLogger(__name__)


class HamiltonianConverter:
    """Class to create and output qubit hamiltonians."""

    def __init__(self, input_hamiltonian: openfermion.QubitOperator) -> None:
        """Initialise class and return output.

        Args:
            input_hamiltonian (object): The input hamiltonian object.
            output_format (str): The name of the desired output format.

        Returns:
            None
        """
        self.openfermion = input_hamiltonian
        self.intermediate = self._of_to_int()

    def convert(self, output_format: str) -> object:
        """Return the required qubit hamiltonian format.

        Returns:
            object: The qubit hamiltonian object of the selected backend.
        """
        output = getattr(self, output_format, None)

        if output is None:
            raise NotImplementedError(
                f"{output_format} is not a valid hamiltonian output format."
            )
        return output

    def _of_to_int(self) -> Dict[str, float]:
        """Construct intermediate representation of qubit hamiltonian from openfermion representation.

        Returns:
            Dict[str, float]: Generic representation of a qubit hamiltonian.
        """
        qh_terms = self.openfermion.terms
        n_qubits: int = 0
        for term in qh_terms.keys():
            for pauli in term:
                position = pauli[0]
                if position > n_qubits:
                    n_qubits = position
        n_qubits += 1

        self.n_qubits = n_qubits

        logger.debug(f"{n_qubits} qubits found in hamiltonian.")

        intermediate: Dict[str, float] = {}
        for term, value in qh_terms.items():
            # Assume I for each qubit unless explicity stated
            op_string = ["I"] * n_qubits
            for pauli in term:
                position = pauli[0]
                operator = pauli[1]

                op_string[position] = operator

            intermediate["".join(op_string)] = value

        return intermediate

    @cached_property
    def pennylane(self) -> qml.Hamiltonian:
        """Convert the intermediate representation to pennlyane Hamiltonian.

        Returns:
            qml.Hamiltonian: Hamiltonian pennylane object.
        """
        # Pennylane

        opdict = {"I": Identity, "X": PauliX, "Y": PauliY, "Z": PauliZ}

        # Initialise the operator with the identity contribution
        values = [v for v in self.intermediate.values()]
        operators = [Identity(self.n_qubits)]

        for op in self.intermediate.keys():

            if op == "I" * self.n_qubits:
                continue

            paulis = [opdict[pauli] for pauli in op]

            pauli_product = paulis[0]
            for p in paulis[1:]:
                pauli_product = pauli_product @ p

            operators += pauli_product

        hamiltonian = qml.Hamiltonian(values, operators)

        return hamiltonian

    @cached_property
    def qiskit(self) -> SpinOp:
        """Return Qiskit spin operator.

        Args:
            intermediate (dict[str, float]): Intermediate representation of a qubit hamiltonian.

        Returns:
            qiskit_nature.operators.second_quantization.SpinOp
        """
        # opdict = {"I": I, "X": X, "Y": Y, "Z": Z}

        # # Initialise the operator with the identity contribution
        # qiskit_op = intermediate["I" * self.n_qubits] * I.tensorpower(self.n_qubits)

        # for op, value in intermediate.items():

        #     if op == "I" * self.n_qubits:
        #         continue

        #     paulis = [opdict[pauli] for pauli in op]

        #     pauli_product = paulis[0]
        #     for p in paulis[1:]:
        #         pauli_product = pauli_product.tensor(p)

        #     qiskit_op += value * pauli_product

        # print(qiskit_op)

        input_list = [(key, value) for key, value in self.intermediate.items()]

        hamiltonian = SpinOp(input_list)
        return hamiltonian
