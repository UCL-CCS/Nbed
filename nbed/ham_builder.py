"""Class to build qubit Hamiltonians from scf object."""

import logging
import warnings
from numbers import Number
from typing import Optional, Tuple

import numpy as np
import openfermion.transforms as of_transforms
from cached_property import cached_property
from openfermion import (
    FermionOperator,
    InteractionOperator,
    QubitOperator,
    count_qubits,
)
from openfermion.config import EQ_TOLERANCE
from openfermion.transforms import jordan_wigner
from pyscf import ao2mo, dft, scf
from pyscf.lib import StreamObject
from symmer.operators import IndependentOp, PauliwordOp, QuantumState
from symmer.projection import QubitTapering, S3Projection

from nbed.exceptions import HamiltonianBuilderError

logger = logging.getLogger(__name__)


class HamiltonianBuilder:
    """Class to build molecular hamiltonians."""

    def __init__(
        self,
        scf_method: StreamObject,
        constant_e_shift: Optional[float] = 0,
        transform: Optional[str] = "jordan_wigner",
        auto_freeze_core: bool = False,
        n_frozen_core: int = 0,
        n_frozen_virt: int = 0,
    ) -> None:
        """Initialise the HamiltonianBuilder.

        Args:
            scf_method: Pyscf scf object.
            constant_e_shift: Constant energy shift to apply to the Hamiltonian.
            transform: Transformation to apply to the Hamiltonian.
            auto_freeze_core: Automatically freeze core orbitals.
            n_frozen_core: Number of core orbitals to freeze.
            n_frozen_virt: Number of virtual orbitals to freeze.
        """
        logger.debug("Initialising HamiltonianBuilder.")
        logger.debug(type(scf_method))
        self.scf_method = scf_method
        self.constant_e_shift = constant_e_shift
        self.transform = transform
        self.auto_freeze_core = auto_freeze_core
        self.n_frozen_core = n_frozen_core
        self.n_frozen_virt = n_frozen_virt
        self._restricted = isinstance(scf_method, (scf.rhf.RHF, dft.rks.RKS))
        # self.occupancy = (
        #     self.scf_method.mo_occ
        # )  # if self._restricted else self.scf_method.mo_occ.sum(axis=1)
        if isinstance(self.scf_method.mo_occ[0], Number):
            self.occupancy = self.scf_method.mo_occ
        elif isinstance(self.scf_method.mo_occ[0], np.ndarray):
            self.occupancy = np.vstack(
                (self.scf_method.mo_occ[0], self.scf_method.mo_occ[1])
            )
        else:
            raise HamiltonianBuilderError("occupancy dimension error")

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
                c_matrix_active[1].T @ hcore[1] @ c_matrix_active[1]
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
                        two_body_coefficients[2 * p, 2 * q, 2 * r, 2 * s] = (
                            two_body_integrals[0, p, q, r, s]
                        )
                        two_body_coefficients[
                            2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1
                        ] = two_body_integrals[1, p, q, r, s]

                        # Mixed spin in physicist
                        two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = (
                            two_body_integrals[2, p, q, r, s]
                        )
                        two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = (
                            two_body_integrals[3, p, q, r, s]
                        )

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

    def get_hartree_fock_state(self):
        """Returns the Hartree-Fock state |1,...,1,0,...,0>."""
        logger.debug(f"{self.occupancy=}")
        electrons = self.occupancy.sum()
        virtuals = (2 * self.occupancy.shape[-1]) - self.occupancy.sum()
        logger.debug(f"{electrons=} {virtuals=}")
        hf_state = np.hstack((np.ones(int(electrons)), np.zeros(int(virtuals)))).astype(
            int
        )
        logger.debug(f"{hf_state=}")
        return QuantumState(hf_state)

    def build(
        self,
        taper: bool = False,
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
        if taper is not None:
            logger.warning("Tapering is deprecated. Use the qubit_reduction_driver.")

        logger.info("Building Hamiltonian")
        one_body_integrals = self._one_body_integrals
        two_body_integrals = self._two_body_integrals
        one_body_coefficients, two_body_coefficients = self._spinorb_from_spatial(
            one_body_integrals, two_body_integrals
        )
        logger.debug(f"{one_body_coefficients.shape=}")
        logger.debug(f"{two_body_coefficients.shape=}")
        logger.debug("Building interaction operator.")
        molecular_hamiltonian = InteractionOperator(
            self.constant_e_shift,
            one_body_coefficients,
            0.5 * two_body_coefficients,
        )
        logger.debug(f"{count_qubits(molecular_hamiltonian)} qubits in Hamiltonian.")
        qham = self._qubit_transform(self.transform, molecular_hamiltonian)
        if taper:
            # for legacy compatibility (recommended usage is via the qubit_reduction_driver)
            logger.debug("Converting to Symmer PauliWordOp")
            pwop = PauliwordOp.from_openfermion(qham)
            logger.debug("Creating reference state.")
            hf_state = self.get_hartree_fock_state()
            logger.debug("Running QubitTapering")
            pwop = QubitTapering(pwop).taper_it(ref_state=hf_state)
            qham = to_openfermion(pwop)
            logger.debug("Symmer functions complete.")
        return qham

    @cached_property
    def H(self) -> PauliwordOp:
        """The full, untapered, unfrozen Hamiltonian."""
        return PauliwordOp.from_openfermion(self.build())

    @cached_property
    def qubit_reduction_driver(self) -> S3Projection:
        """Qubit tapering and frozen core reduction."""
        if isinstance(self.scf_method.mo_occ[0], Number):
            mo_energy = self.scf_method.mo_energy
        elif isinstance(self.scf_method.mo_occ[0], np.ndarray):
            mo_energy = self.scf_method.mo_energy[0]
        else:
            raise ValueError("occupancy dimension error")
        occ_energy = mo_energy[: self.scf_method.mol.nelec[0]]
        nao = mo_energy.shape[0]
        hf_state = self.get_hartree_fock_state()
        if self.auto_freeze_core:
            if self.n_frozen_core != 0:
                warnings.warn(
                    f"Auto freezing core: will overwrite n_frozen_core={self.n_frozen_core}"
                )
            self.n_frozen_core = np.count_nonzero(
                occ_energy < np.mean(occ_energy) - np.std(occ_energy)
            )
        # index the frozen orbitals positions:
        frozen_spatial_orbital_indices = np.append(
            np.argsort(self.scf_method.mo_energy[0])[: self.n_frozen_core],
            np.argsort(self.scf_method.mo_energy[0])[nao - self.n_frozen_virt:],
        )
        frozen_spin_orbital_indices = np.sort(
            np.append(
                2 * frozen_spatial_orbital_indices,
                2 * frozen_spatial_orbital_indices + 1,
            )
        )
        frozen_Z_block = np.eye(nao * 2, dtype=bool)[frozen_spin_orbital_indices]
        # build the symplectic matrix:
        frozen_symp = np.hstack(
            [np.zeros_like(frozen_Z_block, dtype=bool), frozen_Z_block]
        )
        stab_symp = np.vstack(
            [frozen_symp, IndependentOp.symmetry_generators(self.H).symp_matrix]
        )
        # Stabilizer SubSpace (S3) projection object contains the stabilizers for the frozen core AND tapering:
        s3_proj = S3Projection(IndependentOp(stab_symp))
        s3_proj.stabilizers.update_sector(hf_state)
        return s3_proj

    def reduce(self, operator: PauliwordOp = None) -> PauliwordOp:
        """Perform qubit reduction over the input operator.

        If None set, then will take the full molecular Hamiltonian.
        """
        if operator is None:
            operator = self.H
        if isinstance(operator, PauliwordOp):
            return self.qubit_reduction_driver.perform_projection(operator)
        elif isinstance(operator, QuantumState):
            return self.qubit_reduction_driver._project_state(operator)
        else:
            raise ValueError("Unrecognised input, must be PauliwordOp or QuantumState.")


def array_to_dict_nonzero_indices(arr, tol=1e-10):
    """Convert an array to a dict for the non-zero indices."""
    where_nonzero = np.where(~np.isclose(arr, 0, atol=tol))
    nonzero_indices = list(zip(*where_nonzero))
    return dict(zip(nonzero_indices, arr[where_nonzero]))


def fermion_to_qubit_operator(
    fermionic_operator: FermionOperator, n_qubits: int = None
):
    """Function to convert from fermion operators to qubit operators.

    Note: see `openfermion.transforms` for different fermion to qubit mappings

    Args:
        Fermionic_operator(FermionOperator): any fermionic operator (openfermion)
        n_qubits (int): number of qubits (or spin orbitals)

    Returns:
        qubit_operator (PauliwordOp): qubit operator of fermonic operator (under certain mapping)
    """
    mapping = jordan_wigner
    qubit_operator = mapping(fermionic_operator)
    return PauliwordOp.from_openfermion(qubit_operator, n_qubits)


def to_openfermion(pwop: PauliwordOp) -> QubitOperator:
    """Convert to OpenFermion Pauli operator representation.

    Returns:
        open_f (QubitOperator): The QubitOperator representation of the PauliwordOp.
    """

    def symplectic_to_of(symp_vec, coeff) -> str:
        """Returns string form of symplectic vector defined as (X | Z).

        Args:
            symp_vec (array): symplectic Pauliword array

        Returns:
            Pword_string (str): String version of symplectic array
        """
        n_qubits = len(symp_vec) // 2

        X_block = symp_vec[:n_qubits]
        Z_block = symp_vec[n_qubits:]

        Y_loc = np.logical_and(X_block, Z_block)
        X_loc = np.logical_xor(Y_loc, X_block)
        Z_loc = np.logical_xor(Y_loc, Z_block)

        char_aray = np.array(list("I" * n_qubits), dtype=str)

        char_aray[Y_loc] = "Y"
        char_aray[X_loc] = "X"
        char_aray[Z_loc] = "Z"

        indices = np.array(range(n_qubits), dtype=str)
        char_aray = np.char.add(char_aray, indices)[np.where(char_aray != "I")[0]]

        Pword_string = " ".join(char_aray)

        return QubitOperator(Pword_string, coeff)

    open_f = QubitOperator()
    ops = [
        symplectic_to_of(P_sym, coeff)
        for P_sym, coeff in zip(pwop.symp_matrix, pwop.coeff_vec)
    ]
    for op in ops:
        open_f += op
    return open_f
