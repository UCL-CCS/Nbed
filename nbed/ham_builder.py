"""Class to build qubit Hamiltonians from scf object."""
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import openfermion.transforms as of_transforms
from cached_property import cached_property
from openfermion import InteractionOperator, QubitOperator, count_qubits
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import get_active_space_integrals
from openfermion.transforms import taper_off_qubits
from pyscf import ao2mo
from pyscf.lib import StreamObject
from pyscf.lib.numpy_helper import SYMMETRIC
from qiskit.opflow import Z2Symmetries
from typing_extensions import final

from .exceptions import HamiltonianBuilderError
from .ham_converter import HamiltonianConverter

logger = logging.getLogger(__name__)


class HamiltonianBuilder:
    """Class to build molecular hamiltonians."""

    def __init__(
        self,
        scf_method: StreamObject,
        constant_e_shift: Optional[float] = 0,
        transform: Optional[str] = "jordan_wigner",
    ) -> None:
        """Initialise the HamiltonianBuilder.

        Args:
            scf_method: Pyscf scf object.
            constant_e_shift: Constant energy shift to apply to the Hamiltonian.
            transform: Transformation to apply to the Hamiltonian.
        """
        logger.debug("Initialising HamiltonianBuilder.")
        self.scf_method = scf_method
        self.constant_e_shift = constant_e_shift
        self.transform = transform

    @property
    def _one_body_integrals(self) -> np.ndarray:
        """Get the one electron integrals."""
        logger.debug("Calculating one body integrals.")
        c_matrix_active = self.scf_method.mo_coeff

        # one body terms
        one_body_integrals = (
            c_matrix_active.T @ self.scf_method.get_hcore() @ c_matrix_active
        )
        logger.debug("One body integrals found.")
        return one_body_integrals

    @property
    def _two_body_integrals(self) -> np.ndarray:
        """Get the two electron integrals."""
        logger.debug("Calculating two body integrals.")
        c_matrix_active = self.scf_method.mo_coeff
        n_orbs = c_matrix_active.shape[1]

        two_body_compressed = ao2mo.kernel(self.scf_method.mol, c_matrix_active)

        # get electron repulsion integrals
        eri = ao2mo.restore(1, two_body_compressed, n_orbs)  # no permutation symmetry

        # Openfermion uses physicist notation whereas pyscf uses chemists
        two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order="C")
        logger.debug("Two body integrals found.")
        return two_body_integrals

    def _reduce_active_space(self, qubit_reduction: int) -> None:
        """Reduce the active space to accommodate a certain number of qubits."""
        logger.debug("Reducing the active space.")

        if type(qubit_reduction) is not int:
            logger.error("Invalid qubit_reduction of type %s.", type(qubit_reduction))
            raise HamiltonianBuilderError("qubit_reduction must be an Intger")
        if qubit_reduction == 0:
            logger.debug("No active space reduction required.")
            return 0, self._one_body_integrals, self._two_body_integrals

        # find where the last occupied level is
        scf = self.scf_method
        occupied = np.where(scf.mo_occ > 0)[0]
        unoccupied = np.where(scf.mo_occ == 0)[0]

        # +1 because each MO is 2 qubits for closed shell.
        n_orbitals = (qubit_reduction + 1) // 2
        logger.debug(f"Reducing to {n_orbitals}.")
        # Again +1 because we want to use odd numbers to reduce
        # occupied orbitals
        occupied_reduction = (n_orbitals + 1) // 2
        unoccupied_reduction = qubit_reduction - occupied_reduction

        # We want the MOs nearest the fermi level
        # unoccupied orbitals go from 0->N and occupied from N->M
        self._active_space_indices = np.append(
            occupied[occupied_reduction:], unoccupied[:unoccupied_reduction]
        )

        occupied_indices = np.where(self.scf_method.mo_occ > 0)[0]
        logger.debug(f"Active indices {self._active_space_indices}.")

        (
            core_constant,
            one_body_integrals,
            two_body_integrals,
        ) = get_active_space_integrals(
            self._one_body_integrals,
            self._two_body_integrals,
            occupied_indices=occupied_indices,
            active_indices=self._active_space_indices,
        )

        logger.debug("Active space reduced.")
        return core_constant, one_body_integrals, two_body_integrals

    def _qubit_transform(
        self, transform: str, intop: InteractionOperator
    ) -> QubitOperator:
        """Transform second quantised hamiltonain to qubit Hamiltonian.

        Args:
            transform: Transformation to apply to the Hamiltonian.
            intop: InteractionOperator to transform.

        Returns:
            QubitOperator: Transformed qubit Hamiltonian.
        """
        logger.debug(f"Transforming to qubit Hamiltonian using {transform} transform.")
        if transform is None or hasattr(of_transforms, transform) is False:
            raise HamiltonianBuilderError(
                "Invalid transform. Please use a transform from `openfermion.transforms`."
            )

        transform = getattr(of_transforms, transform)

        try:
            qubit_hamiltonain: QubitOperator = transform(intop)
        except TypeError:
            logger.error(
                "Transform selected is not a valid InteractionOperator transform."
            )
            raise HamiltonianBuilderError(
                "Transform selected is not a valid InteractionOperator transform."
            )

        if type(qubit_hamiltonain) is not QubitOperator:
            raise HamiltonianBuilderError(
                "Transform selected must output a QubitOperator."
            )

        logger.debug("Qubit Hamiltonian constructed.")
        return qubit_hamiltonain

    def _taper(self, qham: QubitOperator) -> QubitOperator:
        """Taper a hamiltonian.

        Args:
            qham: QubitOperator to taper.

        Returns:
            QubitOperator: Tapered QubitOperator.
        """
        logger.error("Tapering not implemented.")
        raise ValueError("tapering currently NOT working properly!")
        logger.debug("Beginning qubit tapering.")
        converter = HamiltonianConverter(qham)
        symmetries = Z2Symmetries.find_Z2_symmetries(converter.qiskit)
        symm_strings = [symm.to_label() for symm in symmetries.sq_paulis]

        logger.debug(f"Found {len(symm_strings)} Z2Symmetries")

        stabilizers = []
        for string in symm_strings:
            term = [
                f"{pauli}{index}" for index, pauli in enumerate(string) if pauli != "I"
            ]
            term = " ".join(term)
            stabilizers.append(QubitOperator(term=term))

        logger.debug("Tapering complete.")
        return taper_off_qubits(qham, stabilizers)

    def build(
        self, n_qubits: Optional[int] = None, taper: Optional[bool] = False
    ) -> QubitOperator:
        """Returns second quantized fermionic molecular Hamiltonian.

        constant_e_shift is a constant energy addition... in this code this will be the classical embedding energy
        that corrects for the full system.

        The active_indices and occupied indices are an active space approximation... where occupied and virtual orbitals
        can be frozen. This is different to removing the environment orbitals, as core_constant terms must be added to
        make this approximation.

        Args:
            scf_method (StreamObject): A pyscf self-consistent method.
            constant_e_shift (float): constant energy term to add to Hamiltonian
            active_indices (list): A list of spatial orbital indices indicating which orbitals should be
                                considered active.
            occupied_indices (list):  A list of spatial orbital indices indicating which orbitals should be
                                    considered doubly occupied.

        Returns:
            molecular_hamiltonian (InteractionOperator): fermionic molecular Hamiltonian
        """
        logger.debug("Building for %s qubits.", n_qubits)
        qubit_reduction = 0
        while True:
            (
                core_constant,
                one_body_integrals,
                two_body_integrals,
            ) = self._reduce_active_space(qubit_reduction)

            one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
                one_body_integrals, two_body_integrals
            )

            molecular_hamiltonian = InteractionOperator(
                (self.constant_e_shift + core_constant),
                one_body_coefficients,
                0.5 * two_body_coefficients,
            )

            qham = self._qubit_transform(self.transform, molecular_hamiltonian)

            # Don't like this option sitting with the recursive
            # call beneath it - just a little too complicated.
            # ...but it works for now.
            if taper is True:
                qham = self._taper(qham)
            if n_qubits is None:
                logger.debug("Unreduced Hamiltonain found.")
                return qham

            # Wanted to do a recursive thing to get the correct number
            # from tapering but it takes ages.
            final_n_qubits = count_qubits(qham)

            if final_n_qubits <= n_qubits:
                logger.debug("Hamiltonian reduced to %s qubits.", final_n_qubits)
                return qham

            # Check that we have the right number of qubits.
            qubit_reduction += final_n_qubits - n_qubits
