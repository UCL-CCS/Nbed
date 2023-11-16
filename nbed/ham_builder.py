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
from symmer.operators import PauliwordOp
from symmer.projection import QubitSubspaceManager, QubitTapering
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
        self._restricted = isinstance(scf_method, (scf.rhf.RHF, dft.rks.RKS))
        self.occupancy = (
            self.scf_method.mo_occ
        )  # if self._restricted else self.scf_method.mo_occ.sum(axis=1)

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
        if not self._restricted:
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

        if not self._restricted:
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

    def _reduced_orbitals(
        self,
        qubit_reduction: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find the orbitals which correspond to the active space and core.

        Args:
            qubit_reduction (int): Number of qubits to reduce by.

        Returns:
            active_indices (np.ndarray): Indices of active orbitals.
            core_indices (np.ndarray): Indices of core orbitals.
        """
        logger.debug("Finding active space.")
        logger.debug(f"Reducing by {qubit_reduction} qubits.")

        if type(qubit_reduction) is not int:
            logger.error("Invalid qubit_reduction of type %s.", type(qubit_reduction))
            raise HamiltonianBuilderError("qubit_reduction must be an Intger")
        if qubit_reduction == 0:
            logger.debug("No active space reduction required.")
            if self._restricted:
                return np.array([]), np.where(self.scf_method.mo_occ >= 0)[0]
            else:
                return (
                    np.array([]),
                    np.where(self.scf_method.mo_occ.sum(axis=0) >= 0)[0],
                )

        # +1 because each MO is 2 qubits for closed shell
        orbital_reduction = (qubit_reduction + 1) // 2

        occupation = self.scf_method.mo_occ
        if not self._restricted:
            occupation = occupation.sum(axis=0)
        occupied = np.where(occupation > 0)[0]
        virtual = np.where(occupation == 0)[0]

        # find where the last occupied level is
        logger.debug("Occupied orbitals %s.", occupied)
        logger.debug("virtual orbitals %s.", virtual)

        occupied_reduction = (
            orbital_reduction * len(occupied)
        ) // self._one_body_integrals.shape[-1]
        virtual_reduction = orbital_reduction - occupied_reduction
        logger.debug(f"Reducing occupied by {occupied_reduction} spatial orbitals.")
        logger.debug(f"Reducing virtual by {virtual_reduction} spatial orbitals.")

        core_indices = np.array([])
        removed_virtual = np.array([])

        if occupied_reduction > 0:
            core_indices = occupied[:occupied_reduction]
            occupied = occupied[occupied_reduction:]

        # We want the MOs nearest the fermi level
        if virtual_reduction > 0:
            removed_virtual = virtual[-virtual_reduction:]
            virtual = virtual[:-virtual_reduction]

        active_indices = np.append(occupied, virtual)
        logger.debug(f"Core indices {core_indices}.")
        logger.debug(f"Active indices {active_indices}.")
        logger.debug(f"Removed virtual indices {removed_virtual}.")
        return core_indices, active_indices

    def _reduce_active_space(
        self,
        one_body_integrals: np.ndarray,
        two_body_integrals: np.ndarray,
        core_indices: np.ndarray,
        active_indices: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Reduce the active space to accommodate a certain number of qubits.

        Args:
            one_body_integrals (np.ndarray): One-electron integrals in physicist notation.
            two_body_integrals (np.ndarray): Two-electron integrals in physicist notation.
            core_indices (np.ndarray): Indices of core orbitals.
            active_indices (np.ndarray): Indices of active orbitals.
        """
        logger.debug("Reducing the active space.")
        logger.debug(f"{core_indices=}")
        logger.debug(f"{active_indices=}")

        core_indices = np.array(core_indices)
        active_indices = np.array(active_indices)

        # Determine core constant
        core_constant = 0.0
        if core_indices.ndim != 1:
            logger.error("Core indices given as dimension %s array.", core_indices.ndim)
            raise HamiltonianBuilderError("Core indices must be 1D array.")
        if active_indices.ndim != 1:
            logger.error(
                "Active indices given as dimension %s array.", active_indices.ndim
            )
            raise HamiltonianBuilderError("Active indices must be 1D array.")
        if set(core_indices).intersection(set(active_indices)) != set():
            logger.error("Core and active indices overlap.")
            raise HamiltonianBuilderError("Core and active indices must not overlap.")
        if len(core_indices) + len(active_indices) > self._one_body_integrals.shape[-1]:
            logger.error("Too many indices given.")
            raise HamiltonianBuilderError(
                "Number of core and active indices must not exceed number of orbitals."
            )

        for i in core_indices:
            core_constant += one_body_integrals[0, i, i]
            core_constant += one_body_integrals[1, i, i]

            for j in core_indices:
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
        for i in core_indices:
            for u in active_indices:
                for v in active_indices:
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

        self.occupancy = self.scf_method.mo_occ[..., active_indices]

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

    def build(
        self,
        n_qubits: Optional[int] = None,
        taper: Optional[bool] = True,
        contextual_space: Optional[bool] = False,
        core_indices: Optional[List[int]] = None,
        active_indices: Optional[List[int]] = None,
    ) -> QubitOperator:
        """Returns second quantized fermionic molecular Hamiltonian.

        constant_e_shift is a constant energy addition... in this code this will be the classical embedding energy
        that corrects for the full system.

        The active_indices and occupied indices are an active space approximation... where occupied and virtual orbitals
        can be frozen. This is different to removing the environment orbitals, as core_constant terms must be added to
        make this approximation.

        Args:
            n_qubits (int): Either total number of qubits to use (positive value) or
                number of qubits to reduce size by (negative value).
            taper (bool): Whether to taper the Hamiltonian.
            contextual_space (bool): Whether to project onto the contextual subspace.
            core_indices (List[int]): Indices of core orbitals.
            active_indices (List[int]): Indices of active orbitals.

        Returns:
            molecular_hamiltonian (QubitOperator): Qubit Hamiltonian for molecular system.
        """
        qubit_reduction = 0
        indices_not_set = (core_indices is None) or (active_indices is None)
        if indices_not_set is False:
            core_indices = np.array(core_indices)
            active_indices = np.array(active_indices)

        if taper is True and self._restricted is False:
            raise HamiltonianBuilderError("Unrestricted tapering not implemented.")

        if n_qubits == 0:
            logger.error("n_qubits input as 0.")
            message = "n_qubits input as 0.\n"
            +"Positive integers can be used to define total qubits used.\n"
            +"Negative integers can be used to define a reduction."
            raise HamiltonianBuilderError(message)
        elif n_qubits is None:
            logger.debug("No qubit reduction requested.")
            if contextual_space is True:
                logger.error("Contextual subspace requires a specifc qubit reduction.")
                raise HamiltonianBuilderError(
                    "Contextual subspace requires a specifc qubit reduction."
                )
        elif n_qubits < 0:
            logger.debug("Interpreting negative n_qubits as reduction.")
            n_qubits = (self._one_body_integrals.shape[-1] * 2) + n_qubits

        logger.info("Building Hamiltonian for %s qubits.", n_qubits)

        if indices_not_set:
            logger.debug("No active space indices given.")
            core_indices, active_indices = np.array([]), np.arange(
                self._one_body_integrals.shape[-1]
            )
            logger.debug(f"{core_indices=}")
            logger.debug(f"{active_indices=}")

        max_cycles = 5
        for i in range(1, max_cycles + 1):
            one_body_integrals = self._one_body_integrals
            two_body_integrals = self._two_body_integrals

            (
                core_constant,
                one_body_integrals,
                two_body_integrals,
            ) = self._reduce_active_space(
                one_body_integrals, two_body_integrals, core_indices, active_indices
            )

            one_body_coefficients, two_body_coefficients = self._spinorb_from_spatial(
                one_body_integrals, two_body_integrals
            )
            logger.debug(f"{one_body_coefficients.shape=}")
            logger.debug(f"{two_body_coefficients.shape=}")

            logger.debug("Building interaction operator.")
            molecular_hamiltonian = InteractionOperator(
                (self.constant_e_shift + core_constant),
                one_body_coefficients,
                0.5 * two_body_coefficients,
            )
            logger.debug(
                f"{count_qubits(molecular_hamiltonian)} qubits in Hamiltonian."
            )

            qham = self._qubit_transform(self.transform, molecular_hamiltonian)

            logger.debug("Converting to Symmer PauliWordOp")
            pwop = PauliwordOp.from_openfermion(qham)

            logger.debug("Creating reference state.")
            electrons = self.occupancy.sum()
            states = (2 * self.occupancy.shape[-1]) - self.occupancy.sum()
            logger.debug(f"{electrons=} {states=}")
            hf_state = np.hstack((np.ones(int(electrons)), np.zeros(int(states))))
            logger.debug(f"{hf_state.shape=}")

            # We have to do these separately because QubitSubspaceManager requires n_qubits
            if taper is True:
                logger.debug("Running QubitTapering")
                pwop = QubitTapering(pwop).taper_it(ref_state=hf_state)
            if contextual_space is True and n_qubits is not None:
                logger.debug("Creating QubitSubspaceManager.")
                qsm = QubitSubspaceManager(
                    pwop,
                    ref_state=hf_state,
                    run_qubit_tapering=False,
                    run_contextual_subspace=contextual_space,
                )
                pwop = qsm.get_reduced_hamiltonian(n_qubits=n_qubits)
            qham = pwop.to_openfermion
            logger.debug("Symmer functions complete.")

            if n_qubits is None:
                logger.debug("Unreduced Hamiltonain found.")
                return qham

            # Wanted to do a recursive thing to get the correct number
            # from tapering but it takes ages.
            final_n_qubits = count_qubits(qham)
            logger.debug(f"{final_n_qubits} qubits used in cycle {i} Hamiltonian.")
            if final_n_qubits <= n_qubits:
                logger.debug("Hamiltonian reduced to %s qubits.", final_n_qubits)
                return qham
            if i == max_cycles:
                logger.info("Maximum number of cycles reached.")
                return qham

            # Check that we have the right number of qubits.
            qubit_reduction += final_n_qubits - n_qubits

            if indices_not_set:
                logger.debug("No active space indices given.")
                core_indices, active_indices = self._reduced_orbitals(qubit_reduction)
