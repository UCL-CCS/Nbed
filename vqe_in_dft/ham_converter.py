"""File to contain the qubit hamiltonian format."""

from abc import abstractmethod
import json
import logging
from pathlib import Path
from typing import Dict

import openfermion
from openfermion.ops.operators.qubit_operator import QubitOperator
import pennylane as qml
from cached_property import cached_property
from pennylane import Identity, PauliX, PauliY, PauliZ
from qiskit_nature.operators.second_quantization import SpinOp

logger = logging.getLogger(__name__)

class HamiltonianConverterError(Exception):
    """Base Exception class."""
    pass

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
        if type(input_hamiltonian) is openfermion.QubitOperator:
            self.openfermion = input_hamiltonian
            self.intermediate = self._of_to_int()
        elif type(input_hamiltonian) is Path:
            self.intermediate = self._read_file(input_hamiltonian)
            self.openfermion = self._int_to_of()

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

    def _save(self, filepath: Path) -> None:
        """Save the intermediate representation to file.

        Dump the IR using JSON so that it can be picked up again later.

        Args:
            filepath (Path): Path to the save file location.
        """
        json_ir = json.dumps(self.intermediate)

        with open(filepath, "w") as file:
            file.write(json_ir)

    def _read_file(filepath) -> Dict[str, float]:
        """Read the Intermediate Representation from a file.
        
        Args:
            filepath (Path): Path to a .json file containing the IR.
        """
        with open(filepath, 'r') as file:
            intermediate = json.load(file)

        # Validate input
        error_string = ""
        keys = [key for key in intermediate.keys()]
        if not all([type(key) is str for key in keys]):
            error_string += "JSON file must use strings as operator keys.\n"

        elif not all(len(key) == len(keys[0]) for key in keys):
            error_string += "All operator keys must be of equal length.\n"
        
        elif not all([type(value) is int or type(value) is float for value in intermediate.values()]):
            error_string += "All operator weights must be ints or floats.\n"

        if error_string:
            raise HamiltonianConverterError(error_string)

        return intermediate

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

    def _int_to_of(self) -> openfermion.QubitOperator:
        """Convert from IR to openfermion.
        
        This is needed for reading from file.
        
        Returns:
            openfermion.QubitOperator: Qubit Hamiltonian in openfermion form.
        """
        keys = list(self.intermediate.keys())
        self.n_qubits = len(keys[0])

        operator = self.intermediate["I"*self.n_qubits] * QubitOperator("")
        for key, value in self.intermediate.items():
            term = ""

            if key == "I" * self.n_qubits:
                continue

            for pos, op in enumerate(key):
                if op in ["X", "Y", "Z"]:
                    term += op + str(pos) + " "
            operator += value * QubitOperator(term)

        return operator

    @cached_property
    def pennylane(self) -> qml.Hamiltonian:
        """Convert the intermediate representation to pennlyane Hamiltonian.

        Returns:
            qml.Hamiltonian: Hamiltonian pennylane object.
        """
        opdict = {"I": Identity, "X": PauliX, "Y": PauliY, "Z": PauliZ}

        # Initialise the operator with the identity contribution
        values = [v for v in self.intermediate.values()]
        operators = []

        for op in self.intermediate.keys():

            # Construct a list like [PauliX(0), PauliY(1), Identity(3)]
            paulis = [opdict[pauli](pos) for pos, pauli in enumerate(op)]

            pauli_product = paulis[0]
            for p in paulis[1:]:
                pauli_product = pauli_product @ p

            operators.append(pauli_product)

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
