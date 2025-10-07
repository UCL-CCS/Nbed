"""Base Localizer Class."""

import logging
from abc import ABC, abstractmethod

import numpy as np
from pyscf.lib import StreamObject

from ..system import LocalizedSystem

logger = logging.getLogger(__name__)


class OccupiedLocalizer(ABC):
    """Object used to localise molecular orbitals (MOs) using different localization schemes.

    Running localization returns active and environment systems.

    Note:
    The major improvement of IBOs over PM orbitals is that they are based on IAO charges instead of the erratic
    Mulliken charges. As a result, IBOs are always well-defined.  (Ref: J. Chem. Theory Comput. 2013, 9, 4834−4843)

    Args:
        global_scf (gto.Mole): PySCF molecule object
        n_active_atoms (int): Number of active atoms

    Methods:
        run: Main function to run localization.
    """

    def __init__(
        self,
        global_scf: StreamObject,
        n_active_atoms: int,
        n_mo_overwrite: tuple[int | None, int | None] | None = None,
    ):
        """Initialise class."""
        logger.debug("Initialising OccupiedLocalizerTypes.")
        if global_scf.mo_coeff is None:
            logger.debug("SCF method not initialised, running now...")
            global_scf.run()
            logger.debug("SCF method initialised.")

        self.n_mo_overwrite = (None, None) if n_mo_overwrite is None else n_mo_overwrite
        self._global_scf = global_scf
        self._n_active_atoms = n_active_atoms
        if global_scf.mo_coeff.ndim == 2:
            self.spinless = True
        else:
            self.spinless = False
        logger.debug(f"Global scf: {type(global_scf)}")

    def localize(
        self,
    ) -> LocalizedSystem:
        """Localise orbitals using SPADE.

        Returns:
            LocalizedSystem: A dataclass describing the localization.
        """
        if self.spinless:
            logger.debug("Running SPADE for only one spin.")
            localized_system = self._localize_spin(
                self._global_scf.mo_coeff,
                self._global_scf.mo_occ,
                self.n_mo_overwrite[0],
            )

        else:
            alpha = self._localize_spin(
                self._global_scf.mo_coeff[0],
                self._global_scf.mo_occ[0],
                self.n_mo_overwrite[0],
            )
            beta = self._localize_spin(
                self._global_scf.mo_coeff[1],
                self._global_scf.mo_occ[1],
                self.n_mo_overwrite[1],
            )
            localized_system = LocalizedSystem.unrestricted_from_spin_components(
                alpha, beta
            )
            # to ensure the same number of alpha and beta orbitals are included
            # use the sum of occupancies
            if set(alpha.active_occ_inds) != set(beta.active_occ_inds) or set(
                alpha.enviro_occ_inds
            ) != set(beta.enviro_occ_inds):
                logger.debug(
                    "Recalculating occupied embedded C matrices to enforce equal number between spins."
                )
                mo_occ_sum = np.sum(self._global_scf.mo_occ, axis=0)
                alpha_consistent = self._localize_spin(
                    self._global_scf.mo_coeff[0],
                    mo_occ_sum,
                    self.n_mo_overwrite[0],
                )
                beta_consistent = self._localize_spin(
                    self._global_scf.mo_coeff[1],
                    mo_occ_sum,
                    self.n_mo_overwrite[1],
                )
                localized_system = LocalizedSystem.unrestricted_from_spin_components(
                    alpha_consistent, beta_consistent
                )

        logger.debug("Localization complete.")
        return localized_system

    @abstractmethod
    def _localize_spin(
        self,
        c_matrix: np.ndarray,
        occupancy: np.ndarray,
        n_mo_overwrite: int | None = None,
    ) -> LocalizedSystem:
        """Localize orbitals of one spin.

        Args:
            c_matrix (np.ndarray): Unlocalized C matrix of occupied orbitals.
            occupancy (np.ndarray): Occupancy of orbitals.
            n_mo_overwrite (int | None): Overwrite the number of active molecular orbitals.

        Returns:
            np.ndarray: Localized C matrix of occupied orbitals.
        """
        pass
