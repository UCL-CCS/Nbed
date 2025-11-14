"""Class to build qubit Hamiltonians from scf object."""

import logging

import numpy as np
from numpy.typing import NDArray
from openfermion.config import EQ_TOLERANCE
from pyscf import ao2mo, dft, scf  # type:ignore

from nbed.exceptions import HamiltonianBuilderError

logger = logging.getLogger(__name__)


class HamiltonianBuilder:
    """Class to build molecular hamiltonians."""

    def __init__(
        self,
        scf_method: scf.hf.SCF,
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
        self.occupancy = self.scf_method.mo_occ

    @property
    def _one_body_integrals(self) -> NDArray:
        """Get the one electron integrals."""
        logger.debug("Calculating one body integrals.")
        c_matrix_active: NDArray
        c_matrix_active = self.scf_method.mo_coeff  # type: ignore
        logger.debug(f"{c_matrix_active.shape=}")
        logger.debug(f"{c_matrix_active[0].shape=}")

        logger.debug(f"{self.scf_method.get_hcore().shape=}")

        # Embedding procedure creates two different hcores
        # Using different v_eff
        # We need to cast the unchanged version to 2*M*M
        hcore = self.scf_method.get_hcore()
        if self.scf_method.get_hcore().ndim == 2:
            # Driver has not been used.
            hcore = np.array([self.scf_method.get_hcore()] * 2)

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
        c_matrix_active: NDArray
        c_matrix_active = self.scf_method.mo_coeff  # type: ignore

        eri: NDArray
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
                eri = ao2mo.restore(1, two_body_compressed, n_orbs_alpha)  # type: ignore
                ints_list.append(np.asarray(eri.transpose(0, 2, 3, 1), order="C"))
            two_body_integrals = np.stack(ints_list, axis=0)

        else:
            n_orbs = c_matrix_active.shape[1]

            two_body_compressed = ao2mo.kernel(self.scf_method.mol, c_matrix_active)

            # get electron repulsion integrals
            eri = ao2mo.restore(
                1, two_body_compressed, n_orbs
            )  # no permutation symmetry # type:ignore

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
        logger.info("Building Hamiltonian")
        one_body_integrals = self._one_body_integrals
        two_body_integrals = self._two_body_integrals

        one_body_coefficients, two_body_coefficients = self._spinorb_from_spatial(
            one_body_integrals, two_body_integrals
        )

        logger.debug(f"{self.n_frozen_core=}")
        logger.debug(f"{self.n_frozen_virt=}")
        if self.n_frozen_core != 0:
            occupied_indices = [*range(0, self.n_frozen_core)]
        else:
            occupied_indices = []

        active_indices = [*range(one_body_integrals.shape[1])]
        if self.n_frozen_virt != 0:
            active_indices = active_indices[: -self.n_frozen_virt]
        logger.debug(f"{occupied_indices=}")
        logger.debug(f"{active_indices=}")

        core_const, one_body_coefficients, two_body_coefficients = (
            get_active_space_integrals(
                one_body_coefficients,
                two_body_coefficients,
                occupied_indices=occupied_indices,
                active_indices=active_indices,
            )
        )
        self.constant_e_shift += core_const

        logger.debug(f"{one_body_coefficients.shape=}")
        logger.debug(f"{two_body_coefficients.shape=}")

        return self.constant_e_shift, one_body_coefficients, 0.5 * two_body_coefficients


def get_active_space_integrals(
    one_body_integrals: np.ndarray,
    two_body_integrals: np.ndarray,
    occupied_indices: list,
    active_indices: list,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Restricts a molecule at a spatial orbital level to an active space.

    This function is from Openfermion, see additonal documentation there.

    This active space may be defined by a list of active indices and
        doubly occupied indices. Note that one_body_integrals and
        two_body_integrals must be defined
        n an orthonormal basis set.

    Args:
        one_body_integrals: One-body integrals of the target Hamiltonian
        two_body_integrals: Two-body integrals of the target Hamiltonian
        occupied_indices: A list of spatial orbital indices
            indicating which orbitals should be considered doubly occupied.
        active_indices: A list of spatial orbital indices indicating
            which orbitals should be considered active.

    Returns:
        tuple: Tuple with the following entries:

        **core_constant**: Adjustment to constant shift in Hamiltonian
        from integrating out core orbitals

        **one_body_integrals_new**: one-electron integrals over active
        space.

        **two_body_integrals_new**: two-electron integrals over active
        space.
    """
    # Fix data type for a few edge cases
    occupied_indices = (
        [] if occupied_indices is None else [2 * i for i in occupied_indices]
    )
    occupied_indices += [i + 1 for i in occupied_indices]
    occupied_indices.sort()
    occupied_indices = np.array(occupied_indices)

    if len(active_indices) < 1:
        raise ValueError("Some active indices required for reduction.")

    active_indices = [2 * a for a in active_indices]
    active_indices += [a + 1 for a in active_indices]
    active_indices.sort()
    active_indices = np.array(active_indices)
    logger.debug(f"{occupied_indices=}")
    logger.debug(f"{active_indices=}")

    # Determine core constant
    core_constant = 0.0
    for i in occupied_indices:
        core_constant += 2 * one_body_integrals[i, i]
        for j in occupied_indices:
            core_constant += (
                2 * two_body_integrals[i, j, j, i] - two_body_integrals[i, j, i, j]
            )  # type: ignore

    # Modified one electron integrals
    one_body_integrals_new = np.copy(one_body_integrals)
    for u in active_indices:
        for v in active_indices:
            for i in occupied_indices:
                one_body_integrals_new[u, v] += (
                    2 * two_body_integrals[i, u, v, i] - two_body_integrals[i, u, i, v]
                )
    logger.debug(f"{np.ix_(active_indices, active_indices)}")
    # Restrict integral ranges and change M appropriately
    return (
        core_constant,
        one_body_integrals_new[np.ix_(active_indices, active_indices)],
        two_body_integrals[
            np.ix_(active_indices, active_indices, active_indices, active_indices)
        ],
    )  # type: ignore
