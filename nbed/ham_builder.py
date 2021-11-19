import logging
from typing import List, Optional, Union

import numpy as np
import openfermion.transforms as of_transforms
from openfermion import InteractionOperator, QubitOperator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import get_active_space_integrals
from pyscf import ao2mo
from pyscf.lib import StreamObject

from .exceptions import HamiltonianBuilderError

logger = logging.getLogger(__name__)


class HamiltonianBuilder:
    """Class to build molecular hamiltonians."""

    def __init__(
        self,
        scf_method: StreamObject,
        constant_e_shift: Optional[float] = 0,
        num_qubits: Optional[int] = None,
        transform: Optional[str] = "jordan_wigner",
    ) -> None:
        self.scf_method = scf_method
        self.constant_e_shift = constant_e_shift
        self.transform = transform
        self.num_qubits = num_qubits
        self._core_constant = 0

    @property
    def _one_body_integrals(self) -> np.ndarray:
        """Get the one electron integrals."""
        c_matrix_active = self.scf_method.mo_coeff

        # one body terms
        one_body_integrals = (
            c_matrix_active.T @ self.scf_method.get_hcore() @ c_matrix_active
        )
        return one_body_integrals

    @property
    def _two_body_integrals(self) -> np.ndarray:
        """Get the two electron integrals."""
        c_matrix_active = self.scf_method.mo_coeff
        n_orbs = c_matrix_active.shape[1]

        two_body_compressed = ao2mo.kernel(self.scf_method.mol, c_matrix_active)

        # get electron repulsion integrals
        eri = ao2mo.restore(1, two_body_compressed, n_orbs)  # no permutation symmetry

        # Openfermion uses physicist notation whereas pyscf uses chemists
        two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order="C")
        return two_body_integrals

    def __taper(self) -> InteractionOperator:
        """Taper a hamiltonian."""
        raise NotImplementedError("Tapering not yet implemented")

    def _reduce_active_space(
        self,
        active_indices: Union[np.ndarray, List],
        frozen_indices: Union[np.ndarray, List],
    ) -> None:
        """Reduce the active space to accommodate a certain number of qubits."""
        if self._active_indices or self._frozen_indices:
            (
                core_constant,
                one_body_integrals,
                two_body_integrals,
            ) = get_active_space_integrals(
                self._one_body_integrals,
                self._two_body_integrals,
                occupied_indices=self._frozen_indices,
                active_indices=self._active_indices,
            )

        return core_constant, one_body_integrals, two_body_integrals

    def _qubit_transform(transform, intop: InteractionOperator) -> QubitOperator:
        """Transform second quantised hamiltonain to qubit hamiltonian."""
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
                "Transform selected must output a QubitOperator"
            )

        return qubit_hamiltonain

    def build(self) -> InteractionOperator:
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
        (
            core_constant,
            one_body_integrals,
            two_body_integrals,
        ) = self._reduce_active_space()

        one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
            one_body_integrals, two_body_integrals
        )

        molecular_hamiltonian = InteractionOperator(
            (self.constant_e_shift + core_constant),
            one_body_coefficients,
            0.5 * two_body_coefficients,
        )

        qham = self._qubit_transform(self.transform, molecular_hamiltonian)

        # TODO add tapering here
        from qiskit.opflow import Z2Symmetries

        from .ham_converter import HamiltonianConverter

        converter = HamiltonianConverter(qham)
        symmetries = Z2Symmetries().find_Z2_symmetries(converter.qiskit)
        tapered_qham = symmetries.taper(converter.qiskit)
        import pdb;pdb.set_trace()

        # tapered_hamiltonian = self.taper(molecular_hamiltonian)

        return qham
