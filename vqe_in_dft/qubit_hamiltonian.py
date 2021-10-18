"""File to contain the qubit hamiltonian format"""

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


class HamiltonianBuilder:
    """Class to create and output qubit hamiltonians."""

    def __init__(
        self, scf_method: StreamObject, active_indices: List[int], output_format: str
    ) -> object:
        self.openfermion = self._get_openfermion(scf_method, active_indices)
        self.intermediate = self._of_to_int()
        return self.qubit_hamiltonian(output_format)

    def qubit_hamiltonian(self, output_format) -> object:
        """Return the required qubit hamiltonian format.

        Args:
            output_format (str): Name of the quantum backend to output a hamiltonian for.

        Returns:
            object: The qubit hamiltonian object of the selected backend.
        """

        options = {
            "openfermion": self.openfermion,
            "qiskit": self.build_qiskit(),
            "pennylane": self.build_pennylane(),
        }

        output = options.get(output_format, None)

        if not output:
            raise NotImplementedError(
                "The entered Hamiltonian output format is not valid"
            )
        return output

    def _get_openfermion(
        self, scf_method: StreamObject, active_indices: List[int]
    ) -> openfermion.ops.QubitOperator:
        """Construct the openfermion operator.

        Args:
            scf_method (StreamObject): A pyscf self-consistent method.
            active_indices (list[int]): A list of integer indices of active moleclar orbitals.

        Returns:
            object: A qubit hamiltonian.
        """
        n_orbs = len(active_indices)

        mo_coeff = scf_method.mo_coeff[:, active_indices]

        one_body_integrals = mo_coeff.T @ scf_method.get_hcore() @ mo_coeff

        # temp_scf.get_hcore = lambda *args, **kwargs : initial_h_core
        scf_method.mol.incore_anyway is True

        # Get two electron integrals in compressed format.
        two_body_compressed = ao2mo.kernel(scf_method.mol, mo_coeff)

        two_body_integrals = ao2mo.restore(
            1, two_body_compressed, n_orbs  # no permutation symmetry
        )

        # Openfermion uses pysicist notation whereas pyscf uses chemists
        two_body_integrals = np.asarray(
            two_body_integrals.transpose(0, 2, 3, 1), order="C"
        )

        one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
            one_body_integrals, two_body_integrals
        )

        molecular_hamiltonian = InteractionOperator(
            0, one_body_coefficients, 0.5 * two_body_coefficients
        )

        Qubit_Hamiltonian = jordan_wigner(molecular_hamiltonian)

        self.n_qubits = Qubit_Hamiltonian.many_body_order()

        self.openfermion = Qubit_Hamiltonian

        return self.openfermion

    def _of_to_int(self) -> Dict[str, float]:
        """Construct intermediate representation of qubit hamiltonian from openfermion representation.

        Returns:
            Dict[str, float]: Generic representation of a qubit hamiltonian.
        """
        qh_terms = self.openfermion.terms
        n_qubits = self.openfermion.many_body_order()

        intermediate = {}
        for term, value in qh_terms.items():
            # Assume I for each qubit unless explicity stated
            op_string = ["I"] * n_qubits
            for pauli in term:
                position = pauli[0]
                operator = pauli[1]

                op_string[position] = operator

            intermediate["".join(op_string)] = value

        return intermediate

    def build_pennylane(self) -> qml.Hamiltonian:
        """Convert the intermediate representation to pennlyane Hamiltonian.

        Returns:
            qml.Hamiltonian: Hamiltonian pennylane object.
        """
        # Pennylane

        opdict = {"I": Identity, "X": PauliX, "Y": PauliY, "Z": PauliZ}

        # Initialise the operator with the identity contribution
        values = [v for v in self.intermediate.values()]
        operators = [Identity(self.n_qubits)]

        for op, value in self.intermediate.items():

            if op == "I" * self.n_qubits:
                continue

            paulis = [opdict[pauli] for pauli in op]

            pauli_product = paulis[0]
            for p in paulis[1:]:
                pauli_product = pauli_product @ p

            operators += pauli_product

        self.pennylane = qml.Hamiltonian(values, operators)

        return self.pennylane

    def build_qiskit(self) -> SpinOp:
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

        input_list = [(key, value) for key, value in intermediate.items()]

        self.qiskit = SpinOp(input_list)

        return self.qiskit
