"""File to contain the qubit hamiltonian format."""

import json
import logging
import re
from numbers import Number
from pathlib import Path
from typing import Dict, Optional, Union

import pennylane as qml
from cached_property import cached_property
from openfermion.ops.operators.qubit_operator import QubitOperator
from openfermion.utils import count_qubits
from pennylane import Identity, PauliX, PauliY, PauliZ
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from symmer.operators import PauliwordOp

from .exceptions import HamiltonianConverterError

logger = logging.getLogger(__name__)


class HamiltonianConverter:
    """Class to create and output qubit hamiltonians."""

    def __init__(
        self,
        input_hamiltonian: Union[QubitOperator, str, Path],
    ) -> None:
        """Initialise class and return output.

        Args:
            input_hamiltonian (object): The input hamiltonian object. InteractionOperator, QubitOperator, or a path to a save file.
            transform (str): Transform used to convert to qubit hamiltonian.
            output_format (str): The name of the desired output format.
        """
        logger.debug("Initialising HamiltonianConverter.")
        self._input = None

        if type(input_hamiltonian) is QubitOperator:
            logger.debug("Openfermion Hamiltonian input")
            self._input = input_hamiltonian
            self.n_qubits = count_qubits(input_hamiltonian)
            self._intermediate = self._of_to_int()
        elif type(input_hamiltonian) is dict:
            logger.debug("Dictionary input.")
            # Turn keys into an iterator and get the length of the first item.
            self._validate(input_hamiltonian)
            self.n_qubits = len(next(iter(input_hamiltonian.keys())))
            self._intermediate = input_hamiltonian
        elif type(input_hamiltonian) in [Path, str]:
            logger.debug("Filepath input")
            self._intermediate = self._read_file(input_hamiltonian)
        else:
            logger.error(
                "Input Hamiltonian must be an openfermion.QubitOperator, dict or filepath."
            )
            raise TypeError(
                "Input Hamiltonian must be an openfermion.QubitOperator, dict or filepath."
            )

    def convert(self, output_format: str) -> object:
        """Return the required qubit hamiltonian format.

        Returns:
            object: The qubit hamiltonian object of the selected backend.
        """
        output = getattr(self, output_format, None)

        if output is None:
            logger.debug(f"Output format {output_format} not supported.")
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
        logger.info(f"Saving Hamiltonian to {filepath}.")
        data_to_save = {"qubits": self.n_qubits, "hamiltonian": self._intermediate}
        json_ir = json.dumps(data_to_save)

        with open(filepath, "w") as file:
            file.write(json_ir)
        logger.debug("File saved.")

    def _read_file(self, filepath) -> Dict[str, float]:
        """Read the Intermediate Representation from a file.

        Args:
            filepath (Path): Path to a .json file containing the IR.
        """
        logger.debug("Reading IR from file.")
        with open(filepath, "r") as file:
            file_data = json.load(file)

        self.n_qubits = file_data["qubits"]
        intermediate = file_data["hamiltonian"]

        self._validate(intermediate)

        return intermediate

    def _validate(self, dict_input: Dict[str, Number]) -> None:
        # Validate input
        logger.debug("Validating input file.")
        error_string = ""
        keys = [key for key in dict_input]

        if not all([re.match("[IXYZixyz]+", key) for key in keys]):
            error_string += "Input dict keys must only contain I,X,Y,Z\n"

        elif not all(len(key) == len(keys[0]) for key in keys):
            error_string += "All operator keys must be of equal length.\n"

        elif not all([isinstance(value, Number) for value in dict_input.values()]):
            error_string += "All operator weights must be numbers.\n"

        if error_string:
            logger.error(f"{error_string}")
            raise HamiltonianConverterError(error_string)
        logger.debug("Input dict valid.")

    def _of_to_int(self) -> Dict[str, float]:
        """Construct intermediate representation of qubit hamiltonian from openfermion representation.

        Returns:
            Dict[str, float]: Generic representation of a qubit hamiltonian.
        """
        logger.debug("Creating IR from openfermion representation.")
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

    @cached_property
    def openfermion(self) -> QubitOperator:
        """Convert from IR to openfermion.

        This is needed for reading from file.

        Returns:
            openfermion.QubitOperator: Qubit Hamiltonian in openfermion form.
        """
        logger.debug("Converting to openfermion QubitOperator.")
        if self._input is not None:
            return self._input

        operator = self._intermediate.get("I" * self.n_qubits, 0) * QubitOperator("")
        for key, value in self._intermediate.items():
            term = ""

            if key == "I" * self.n_qubits:
                continue

            for pos, op in enumerate(key):
                if op in ["X", "Y", "Z"]:
                    term += op + str(pos) + " "
            operator += value * QubitOperator(term)

        logger.debug("Openfermion QubitOperator created.")
        return operator

    @cached_property
    def pennylane(self) -> qml.Hamiltonian:
        """Convert the intermediate representation to pennlyane Hamiltonian.

        Returns:
            qml.Hamiltonian: Hamiltonian pennylane object.
        """
        logger.debug("Converting to pennylane Hamiltonian.")
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

        logger.debug("Pennylane Hamiltonian created.")
        return hamiltonian

    @cached_property
    def qiskit(self) -> PauliSumOp:
        """Return Qiskit spin operator.

        Args:
            intermediate (dict[str, float]): Intermediate representation of a qubit hamiltonian.

        Returns:
            qiskit_nature.opflow.PauliSumOp
        """
        logger.debug("Converting to qiskit PauliSumOp.")
        from qiskit.opflow.primitive_ops import PauliSumOp

        input_list = [(key, value) for key, value in self._intermediate.items()]

        logger.debug("Qiskit PauliSumOp created.")
        return PauliSumOp.from_list(input_list)

    @cached_property
    def symmer(self) -> PauliwordOp:
        """Convert from intermediate representation to symmer PauliwordOp.

        Args:
            intermediate (dict[str, float]): Intermediate representation of a qubit hamiltonian.

        Returns:
            symmer.operators.PauliwordOp
        """
        logger.debug("Converting to symmer PauliwordOp")
        from symmer.operators import PauliwordOp

        input_dict = self._intermediate

        logger.debug("symmer PauliwordOp created.")
        return PauliwordOp.from_dictionary(input_dict)


def load_hamiltonian(
    filepath: Path, output: str
) -> Union[QubitOperator, qml.Hamiltonian, PauliSumOp, PauliwordOp]:
    """Create a Hamiltonian from a file.

    Reads the input file and converts to the desired output format.

    Args:
        filepath (Path): Path to a saved Qubit Hamiltonian file.
        output (str): One of 'openfermion', 'pennylane', 'qiskit', 'symmer'

    Returns:
        object: A qubit Hamiltonian.
    """
    return HamiltonianConverter(filepath).convert(output)
