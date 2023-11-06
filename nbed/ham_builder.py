"""Class to build qubit Hamiltonians from scf object."""
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import openfermion.transforms as of_transforms
from cached_property import cached_property
from openfermion import InteractionOperator, QubitOperator, count_qubits
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.config import EQ_TOLERANCE
from openfermion.ops.representations import get_active_space_integrals
from openfermion.transforms import taper_off_qubits
from pyscf import ao2mo, dft, scf
from pyscf.lib import StreamObject
from pyscf.lib.numpy_helper import SYMMETRIC
from qiskit.opflow import Z2Symmetries
from typing_extensions import final

from nbed.exceptions import HamiltonianBuilderError
from nbed.ham_converter import HamiltonianConverter

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
        logger.debug(type(scf_method))
        self.scf_method = scf_method
        self.constant_e_shift = constant_e_shift
        self.transform = transform

    @property
    def _one_body_integrals(self) -> np.ndarray:
        """Get the one electron integrals."""
        logger.debug("Calculating one body integrals.")
        c_matrix_active = self.scf_method.mo_coeff
        logger.debug(f"{c_matrix_active.shape=}")
        logger.debug(f"{c_matrix_active[0].shape=}")

        logger.debug(f"{self.scf_method.get_hcore().shape=}")

        # Embedding procedure creates two different hcores
        # Using different v_eff
        if self.scf_method.get_hcore().ndim == 2:
            # Driver has not been used.
            hcore = [self.scf_method.get_hcore()] * 2
        elif self.scf_method.get_hcore().ndim == 3:
            # Driver has been used.
            hcore = self.scf_method.get_hcore()

        # one body terms
        if isinstance(self.scf_method, (scf.uhf.UHF, dft.uks.UKS)):
            logger.info("Calculating unrestricted one body intergrals.")
            one_body_integrals_alpha = (
                c_matrix_active[0].T @ hcore[0] @ c_matrix_active[0]
            )
            one_body_integrals_beta = (
                c_matrix_active[1].T @ hcore[0] @ c_matrix_active[1]
            )

            one_body_integrals = np.array(
                [one_body_integrals_alpha, one_body_integrals_beta]
            )

        else:
            logger.info("Calculating restricted one body integrals.")
            # We double these up so that we have the same number as
            # the unrestricted case.
            one_body_integrals = np.array(
                [c_matrix_active.T @ self.scf_method.get_hcore() @ c_matrix_active] * 2
            )
        logger.debug("One body integrals found.")
        logger.debug(f"{one_body_integrals.shape}")

        return one_body_integrals

    @property
    def _two_body_integrals(self) -> np.ndarray:
        """Get the two electron integrals."""
        logger.debug("Calculating two body integrals.")
        c_matrix_active = self.scf_method.mo_coeff

        if isinstance(self.scf_method, (scf.uhf.UHF, dft.uks.UKS)):
            n_orbs_alpha = c_matrix_active[0].shape[1]
            n_orbs_beta = c_matrix_active[1].shape[1]

            # Could make this more flexible later.
            if n_orbs_alpha != n_orbs_beta:
                raise HamiltonianBuilderError(
                    "Must localize the same number of alpha and beta orbitals."
                )

            c_alpha = c_matrix_active[0]
            c_beta = c_matrix_active[1]

            # Pyscf is in chemist notation
            # later we transpose to physicist notation for openfermion
            spin_options = {
                "aaaa": (c_alpha, c_alpha, c_alpha, c_alpha),
                "bbbb": (c_beta, c_beta, c_beta, c_beta),
                "aabb": (c_alpha, c_alpha, c_beta, c_beta),
                "bbaa": (c_beta, c_beta, c_alpha, c_alpha),
            }

            two_body_integrals = []
            for spin in spin_options:
                two_body_compressed = ao2mo.kernel(
                    self.scf_method.mol, spin_options[spin]
                )
                eri = ao2mo.restore(1, two_body_compressed, n_orbs_alpha)
                two_body_integrals.append(
                    np.asarray(eri.transpose(0, 2, 3, 1), order="C")
                )

        else:
            n_orbs = c_matrix_active.shape[1]

            two_body_compressed = ao2mo.kernel(self.scf_method.mol, c_matrix_active)

            # get electron repulsion integrals
            eri = ao2mo.restore(
                1, two_body_compressed, n_orbs
            )  # no permutation symmetry

            # Copy this 4 times so that we have the same number as
            # the unrestricted case
            # Openfermion uses physicist notation whereas pyscf uses chemists
            two_body_integrals = [np.asarray(eri.transpose(0, 2, 3, 1), order="C")] * 4

        two_body_integrals = np.array(two_body_integrals)

        logger.debug("Two body integrals found.")
        logger.debug(f"{two_body_integrals.shape}")
        return two_body_integrals

    def _reduce_active_space(
        self,
        qubit_reduction: int,
        one_body_integrals: np.ndarray,
        two_body_integrals: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Reduce the active space to accommodate a certain number of qubits.

        Args:
            qubit_reduction (int): Number of qubits to reduce by.
            one_body_integrals (np.ndarray): One-electron integrals in physicist notation.
            two_body_integrals (np.ndarray): Two-electron integrals in physicist notation.
        """
        logger.debug("Reducing the active space.")

        if type(qubit_reduction) is not int:
            logger.error("Invalid qubit_reduction of type %s.", type(qubit_reduction))
            raise HamiltonianBuilderError("qubit_reduction must be an Intger")
        if qubit_reduction == 0:
            logger.debug("No active space reduction required.")
            return 0, one_body_integrals, two_body_integrals

        # find where the last occupied level is
        occupied = np.where(self.scf_method.mo_occ > 0)[0]
        unoccupied = np.where(self.scf_method.mo_occ == 0)[0]

        # +1 because each MO is 2 qubits for closed shell.
        n_orbitals = (qubit_reduction + 1) // 2
        logger.debug(f"Reducing to {n_orbitals}.")
        # Again +1 because we want to use odd numbers to reduce
        # occupied orbitals
        occupied_reduction = (n_orbitals + 1) // 2
        unoccupied_reduction = qubit_reduction - occupied_reduction

        # We want the MOs nearest the fermi level
        # unoccupied orbitals go from 0->N and occupied from N->M
        active_indices = np.append(
            occupied[occupied_reduction:], unoccupied[:unoccupied_reduction]
        )
        self._active_space_indices = active_indices
        logger.debug(f"Active indices {self._active_space_indices}.")

        occupied_indices = np.where(self.scf_method.mo_occ > 0)[0]

        # Determine core constant
        core_constant = 0.0
        for i in occupied_indices:
            core_constant += one_body_integrals[0, i, i]
            core_constant += one_body_integrals[1, i, i]

            for j in occupied_indices:
                core_constant += (
                    two_body_integrals[0, i, j, j, i]
                    - two_body_integrals[0, i, j, i, j]
                )
                core_constant += (
                    two_body_integrals[1, i, j, j, i]
                    - two_body_integrals[1, i, j, i, j]
                )

        # Modified one electron integrals
        one_body_integrals_new = np.copy(one_body_integrals)
        for u in active_indices:
            for v in active_indices:
                for i in occupied_indices:
                    one_body_integrals_new[0, u, v] += (
                        two_body_integrals[0, i, u, v, i]
                        - two_body_integrals[0, i, u, i, v]
                    )
                    one_body_integrals_new[1, u, v] += (
                        two_body_integrals[1, i, u, v, i]
                        - two_body_integrals[1, i, u, i, v]
                    )

        one_body_integrals_new = one_body_integrals_new[
            np.ix_([0, 1], active_indices, active_indices)
        ]
        two_body_integrals_new = two_body_integrals[
            np.ix_(
                [0, 1, 2, 3],
                active_indices,
                active_indices,
                active_indices,
                active_indices,
            )
        ]

        # Restrict integral ranges and change M appropriately
        logger.debug("Active space reduced.")
        logger.debug(f"{one_body_integrals_new.shape}")
        logger.debug(f"{two_body_integrals_new.shape}")
        return core_constant, one_body_integrals_new, two_body_integrals_new

    def _spinorb_from_spatial(
        self, one_body_integrals: np.ndarray, two_body_integrals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert spatial integrals to spin-orbital integrals.

        Args:
            one_body_integrals (np.ndarray): One-electron integrals in physicist notation.
            two_body_integrals (np.ndarray): Two-electron integrals in physicist notation.

        Returns:
            one_body_coefficients (np.ndarray): One-electron coefficients in spinorb form.
            two_body_coefficients (np.ndarray): Two-electron coefficients in spinorb form.

        """
        logger.debug("Converting to spin-orbital coefficients.")
        n_qubits = one_body_integrals[0].shape[0] + one_body_integrals[1].shape[0]

        # Initialize Hamiltonian coefficients.
        one_body_coefficients = np.zeros((n_qubits, n_qubits))
        two_body_coefficients = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

        # Loop through integrals.
        for p in range(n_qubits // 2):
            for q in range(n_qubits // 2):

                # Populate 1-body coefficients. Require p and q have same spin.
                one_body_coefficients[2 * p, 2 * q] = one_body_integrals[0, p, q]
                one_body_coefficients[2 * p + 1, 2 * q + 1] = one_body_integrals[
                    1, p, q
                ]

                # Continue looping to prepare 2-body coefficients.
                # Assumes 2e ints are ordered as aaaa,bbbb,aabb,bbaa.
                for r in range(n_qubits // 2):
                    for s in range(n_qubits // 2):

                        # Same spin
                        two_body_coefficients[
                            2 * p, 2 * q, 2 * r, 2 * s
                        ] = two_body_integrals[0, p, q, r, s]
                        two_body_coefficients[
                            2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1
                        ] = two_body_integrals[1, p, q, r, s]

                        # Mixed spin in physicist
                        two_body_coefficients[
                            2 * p, 2 * q + 1, 2 * r + 1, 2 * s
                        ] = two_body_integrals[2, p, q, r, s]
                        two_body_coefficients[
                            2 * p + 1, 2 * q, 2 * r, 2 * s + 1
                        ] = two_body_integrals[3, p, q, r, s]

        # Truncate.
        one_body_coefficients[np.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.0
        two_body_coefficients[np.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.0

        return one_body_coefficients, two_body_coefficients

    @staticmethod
    def _qubit_transform(transform: str, intop: InteractionOperator) -> QubitOperator:
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
            n_qubits (int): Either total number of qubits to use (positive value) or number of qubits to reduce size by (negative value).
            taper (bool): Whether to taper the Hamiltonian.
        Returns:
            molecular_hamiltonian (QubitOperator): Qubit Hamiltonian for molecular system.
        """
        if n_qubits == 0:
            logger.error("n_qubits input as 0.")
            message = "n_qubits input as 0.\n"
            + "Positive integers can be used to define total qubits used.\n"
            + "Negative integers can be used to define a reduction."
            raise HamiltonianBuilderError(message)
        elif n_qubits is None:
            logger.debug("No qubit reduction requested.")
            n_qubits = 0
        elif n_qubits < 0:
            logger.debug("Interpreting negative n_qubits as reduction.")
            n_qubits = (self._one_body_integrals.shape[-1] * 2) + n_qubits

        logger.debug("Building for %s qubits.", n_qubits)
        qubit_reduction = 0

        while True:
            one_body_integrals = self._one_body_integrals
            two_body_integrals = self._two_body_integrals

            (
                core_constant,
                one_body_integrals,
                two_body_integrals,
            ) = self._reduce_active_space(
                qubit_reduction, one_body_integrals, two_body_integrals
            )

            one_body_coefficients, two_body_coefficients = self._spinorb_from_spatial(
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
            if n_qubits == 0:
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
