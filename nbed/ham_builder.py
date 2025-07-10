"""Class to build qubit Hamiltonians from scf object."""

import logging
from numbers import Number

import numpy as np
from numpy.typing import NDArray
from openfermion.config import EQ_TOLERANCE
from pyscf import ao2mo, dft, lib, scf
from pyscf.lib import StreamObject

from nbed.exceptions import HamiltonianBuilderError

logger = logging.getLogger(__name__)


class HamiltonianBuilder:
    """Class to build molecular hamiltonians."""

    def __init__(
        self,
        scf_method: StreamObject,
        constant_e_shift: float = 0,
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
        self.n_frozen_core = n_frozen_core
        self.n_frozen_virt = n_frozen_virt
        self._restricted = isinstance(scf_method, (scf.rhf.RHF, dft.rks.RKS))
        if isinstance(self.scf_method.mo_occ[0], Number):
            self.occupancy = self.scf_method.mo_occ
        elif isinstance(self.scf_method.mo_occ[0], np.ndarray):
            self.occupancy = np.vstack(
                (self.scf_method.mo_occ[0], self.scf_method.mo_occ[1])
            )
        else:
            raise HamiltonianBuilderError("occupancy dimension error")

    @property
    def _one_body_integrals(self) -> NDArray:
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
    def _two_body_integrals(self) -> NDArray:
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

            ints_list = []
            for spin in spin_options:
                two_body_compressed = ao2mo.kernel(
                    self.scf_method.mol, spin_options[spin]
                )
                eri = ao2mo.restore(1, two_body_compressed, n_orbs_alpha)
                ints_list.append(np.asarray(eri.transpose(0, 2, 3, 1), order="C"))
            two_body_integrals = np.stack(ints_list, axis=0)

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
            two_body_integrals = np.stack(
                [np.asarray(eri.transpose(0, 2, 3, 1), order="C")] * 4, axis=0
            )

        two_body_integrals = np.array(two_body_integrals)

        logger.debug("Two body integrals found.")
        logger.debug(f"{two_body_integrals.shape}")
        return two_body_integrals

    def _spinorb_from_spatial(
        self, one_body_integrals: NDArray, two_body_integrals: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Convert spatial integrals to spin-orbital integrals.

        Args:
            one_body_integrals (NDArray): One-electron integrals in physicist notation.
            two_body_integrals (NDArray): Two-electron integrals in physicist notation.

        Returns:
            one_body_coefficients (NDArray): One-electron coefficients in spinorb form.
            two_body_coefficients (NDArray): Two-electron coefficients in spinorb form.

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
                        # aaaa
                        two_body_coefficients[2 * p, 2 * q, 2 * r, 2 * s] = (
                            two_body_integrals[0, p, q, r, s]
                        )
                        # bbbb
                        two_body_coefficients[
                            2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1
                        ] = two_body_integrals[1, p, q, r, s]

                        # Mixed spin in physicist
                        # abba
                        two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = (
                            two_body_integrals[2, p, q, r, s]
                        )
                        # baab
                        two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = (
                            two_body_integrals[3, p, q, r, s]
                        )

        # Truncate.
        one_body_coefficients[np.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.0
        two_body_coefficients[np.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.0

        return one_body_coefficients, two_body_coefficients

    def build(
        self,
    ) -> tuple[float, NDArray, NDArray]:
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
            (float, npt.NDArray, npt.NDArray): The one and two body spinorb coefficients
        """
        if self.n_frozen_virt != 0:
            self.scf_method = reduce_virtuals(self.scf_method, self.n_frozen_virt)

        logger.info("Building Hamiltonian")
        one_body_integrals = self._one_body_integrals
        two_body_integrals = self._two_body_integrals
        one_body_coefficients, two_body_coefficients = self._spinorb_from_spatial(
            one_body_integrals, two_body_integrals
        )

        logger.debug(f"{one_body_coefficients.shape=}")
        logger.debug(f"{two_body_coefficients.shape=}")

        return self.constant_e_shift, one_body_coefficients, 0.5 * two_body_coefficients


def reduce_virtuals(scf_method, n_frozen_virt: int) -> lib.StreamObject:
    """Reduce the number of virtual orbitals.

    Args:
        scf_method (StreamObject): A PySCF scf object.
        n_frozen_virt (int):  Number of virtual orbitals to freeze.

    Return:
        StreamObject: A new scf object with fewer virtual orbitals.
    """
    reduced_scf_method = scf_method.copy()
    if n_frozen_virt <= 0:
        logger.debug("No virtual orbital reduction.")
        return reduced_scf_method
    elif n_frozen_virt >= np.count_nonzero(reduced_scf_method.mo_occ):
        logger.error("Attempting to reduce the virtual space by more than exist.")
        raise ValueError("Atempting to reduce virtual space by more than exist.")

    logger.debug(f"Reducing virtuals by {n_frozen_virt}.")

    if isinstance(reduced_scf_method, (scf.uhf.UHF)):
        reduced_scf_method.mo_coeff = reduced_scf_method.mo_coeff[:, :, :-n_frozen_virt]
        reduced_scf_method.mo_occ = reduced_scf_method.mo_occ[:, :-n_frozen_virt]

    elif isinstance(reduced_scf_method, (scf.hf.RHF, scf.rohf.ROHF)):
        reduced_scf_method.mo_coeff = reduced_scf_method.mo_coeff[:, :-n_frozen_virt]
        reduced_scf_method.mo_occ = reduced_scf_method.mo_occ[:-n_frozen_virt]

    return reduced_scf_method
