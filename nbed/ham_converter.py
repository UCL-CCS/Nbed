"""File to contain the qubit hamiltonian format."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import openfermion as of
import pennylane as qml
from cached_property import cached_property
from openfermion.ops.operators.qubit_operator import QubitOperator
from openfermion.utils import count_qubits
from pennylane import Identity, PauliX, PauliY, PauliZ
from qiskit_nature.operators.second_quantization import SpinOp

from .exceptions import HamiltonianConverterError

logger = logging.getLogger(__name__)


class HamiltonianConverter:
    """Class to create and output qubit hamiltonians."""

    def __init__(
        self,
        input_hamiltonian: Union[of.InteractionOperator, of.QubitOperator, str, Path],
        transform: Optional[str] = None,
    ) -> None:
        """Initialise class and return output.

        Args:
            input_hamiltonian (object): The input hamiltonian object. InteractionOperator, QubitOperator, or a path to a save file.
            transform (str): Transform used to convert to qubit hamiltonian.
            output_format (str): The name of the desired output format.
        """
        if type(input_hamiltonian) is of.InteractionOperator:
            self._second_quantized = input_hamiltonian
            self.openfermion = self.transform(transform.lower())
            self.n_qubits = count_qubits(input_hamiltonian)
            self._intermediate = self._of_to_int()

        elif type(input_hamiltonian) is of.QubitOperator:
            self.openfermion = input_hamiltonian
            self.n_qubits = count_qubits(input_hamiltonian)
            self._intermediate = self._of_to_int()

        elif type(input_hamiltonian) in [Path, str]:
            self._intermediate = self._read_file(input_hamiltonian)
            self.openfermion = self._int_to_of()
        else:
            raise TypeError(
                "Input Hamiltonian must be an openfermion.InteractionOperator or path."
            )

    def transform(self, transform):
        """Transform second quantised hamiltonain to qubit hamiltonian."""
        if transform is None or hasattr(of.transforms, transform) is False:
            raise HamiltonianConverterError(
                "Invalid transform. Please use a transform from `openfermion.transforms`."
            )

        transform = getattr(of.transforms, transform)

        try:
            qubit_hamiltonain: QubitOperator = transform(self._second_quantized)
        except TypeError as e:
            logger.error(
                "Transform selected is not a valid InteractionOperator transform."
            )
            raise HamiltonianConverterError(
                "Transform selected is not a valid InteractionOperator transform."
            )

        if type(qubit_hamiltonain) is not QubitOperator:
            raise HamiltonianConverterError(
                "Transform selected must output a QubitOperator"
            )

        return qubit_hamiltonain

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

    def save(self, filepath: Path) -> None:
        """Save the intermediate representation to file.

        Dump the IR using JSON so that it can be picked up again later.

        Args:
            filepath (Path): Path to the save file location.
        """
        data_to_save = {"qubits": self.n_qubits, "hamiltonian": self._intermediate}
        json_ir = json.dumps(data_to_save)

        with open(filepath, "w") as file:
            file.write(json_ir)

    def _read_file(self, filepath) -> Dict[str, float]:
        """Read the Intermediate Representation from a file.

        Args:
            filepath (Path): Path to a .json file containing the IR.
        """
        with open(filepath, "r") as file:
            file_data = json.load(file)

        self.n_qubits = file_data["qubits"]
        intermediate = file_data["hamiltonian"]

        # Validate input
        error_string = ""
        keys = [key for key in intermediate.keys()]
        if not all([type(key) is str for key in keys]):
            error_string += "JSON file must use strings as operator keys.\n"

        elif not all(len(key) == len(keys[0]) for key in keys):
            error_string += "All operator keys must be of equal length.\n"

        elif not all(
            [
                type(value) is int or type(value) is float
                for value in intermediate.values()
            ]
        ):
            error_string += "All operator weights must be ints or floats.\n"

        if error_string:
            raise HamiltonianConverterError(error_string)

        return intermediate

    def _of_to_int(self) -> Dict[str, float]:
        """Construct intermediate representation of qubit hamiltonian from openfermion representation.

        Returns:
            Dict[str, float]: Generic representation of a qubit hamiltonian.
        """
        intermediate: Dict[str, float] = {}
        for term, value in self.openfermion.terms.items():
            # Assume I for each qubit unless explicity stated
            op_string = ["I"] * self.n_qubits
            for pauli in term:
                position = pauli[0]
                operator = pauli[1]

                op_string[position] = operator

            intermediate["".join(op_string)] = value

        return intermediate

    def _int_to_of(self) -> of.QubitOperator:
        """Convert from IR to openfermion.

        This is needed for reading from file.

        Returns:
            openfermion.QubitOperator: Qubit Hamiltonian in openfermion form.
        """
        operator = self._intermediate["I" * self.n_qubits] * QubitOperator("")
        for key, value in self._intermediate.items():
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
        values = [v for v in self._intermediate.values()]
        operators = []

        for op in self._intermediate.keys():

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
        input_list = [(key, value) for key, value in self._intermediate.items()]

        hamiltonian = SpinOp(input_list)
        return hamiltonian
