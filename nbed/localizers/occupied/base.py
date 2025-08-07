"""Base Localizer Class."""

import logging
from abc import ABC, abstractmethod

import numpy as np
from pyscf.lib import StreamObject

from ...exceptions import NbedLocalizerError
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

    Attributes:
        c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
        c_enviro (np.array): C matrix of localized occupied ennironment MOs
        c_loc_occ_and_virt (np.array): Full localized C_matrix (occpuied and virtual)
        dm_active (np.array): active system density matrix
        dm_enviro (np.array): environment system density matrix
        active_mo_inds (np.array): 1D array of active occupied MO indices
        enviro_mo_inds (np.array): 1D array of environment occupied MO indices
        c_loc_occ (np.array): C matrix of localized occupied MOs

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
            active_mo_inds (np.array): 1D array of active occupied MO indices
            enviro_mo_inds (np.array): 1D array of environment occupied MO indices
            c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
            c_enviro (np.array): C matrix of localized occupied ennironment MOs
            c_loc_occ (np.array): full C matrix of localized occupied MOs
        """
        if self.spinless:
            logger.debug("Running SPADE for only one spin.")
            localized_system = self._localize_spin(
                self._global_scf.mo_coeff,
                self._global_scf.mo_occ,
                self.n_mo_overwrite[0],
            )

            localized_system.dm_active *= 2.0
            localized_system.dm_enviro *= 2.0

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
            localized_system = LocalizedSystem(
                np.array([alpha.active_mo_inds, beta.active_mo_inds]),
                np.array([alpha.enviro_mo_inds, beta.enviro_mo_inds]),
                np.array([alpha.c_active, beta.c_active]),
                np.array([alpha.c_enviro, beta.c_enviro]),
                np.array([alpha.c_loc_occ, beta.c_loc_occ]),
            )
            # to ensure the same number of alpha and beta orbitals are included
            # use the sum of occupancies
            if set(alpha.active_mo_inds) != set(beta.active_mo_inds) or set(
                alpha.enviro_mo_inds
            ) != set(beta.enviro_mo_inds):
                logger.debug(
                    "Recalculating occupied embedded C matrices to enforce equal number between spins."
                )
                mo_occ_sum = np.sum(self._global_scf.mo_occ, axis=0)
                alpha_consistent = self._localize_spin(
                    self._global_scf.mo_coeff[0],
                    mo_occ_sum,
                    self.n_mo_overwrite[0],
                )
                consistent = self._localize_spin(
                    self._global_scf.mo_coeff[1],
                    mo_occ_sum,
                    self.n_mo_overwrite[1],
                )
                localized_system = LocalizedSystem(
                    np.array([alpha.active_mo_inds, beta.active_mo_inds]),
                    np.array([alpha.enviro_mo_inds, beta.enviro_mo_inds]),
                    np.array([alpha_consistent.c_active, consistent.c_active]),
                    np.array([alpha_consistent.c_enviro, consistent.c_enviro]),
                    np.array([alpha_consistent.c_loc_occ, consistent.c_loc_occ]),
                )

        logger.debug("Localization complete.")
        logger.debug("Localized orbitals:")
        logger.debug(f"{localized_system.active_mo_inds=}")
        logger.debug(f"{localized_system.enviro_mo_inds=}")
        logger.debug(f"{localized_system.c_active.shape=}")
        logger.debug(f"{localized_system.c_enviro.shape=}")
        logger.debug(f"{localized_system.c_loc_occ.shape=}")

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


def check_values(
    localized_system: LocalizedSystem, global_scf: StreamObject
) -> None:  # Needs clarification
    """Check that output values make sense.

    - Same number of active and environment orbitals in alpha and beta
    - Total DM is sum of active and environment DM
    - Total number of electrons conserved

    """
    logger.debug("Running localizer sense check.")
    warn_flag = False
    if localized_system.active_mo_inds.ndim == 2:
        logger.debug("Checking spin does not affect localization.")
        active_number_match = (
            localized_system.active_mo_inds[0].shape
            == localized_system.active_mo_inds[1].shape
        )
        logger.debug(f"{active_number_match=}")
        enviro_number_match = (
            localized_system.enviro_mo_inds[0].shape
            == localized_system.enviro_mo_inds[1].shape
        )
        logger.debug(f"{enviro_number_match=}")
        if not active_number_match or not enviro_number_match:
            logger.error("Number of alpha and beta orbitals do not match.")
            warn_flag = True

    # checking denisty matrix parition sums to total
    logger.debug("Checking density matrix partition.")
    match localized_system.c_loc_occ.ndim:
        case 2:
            # In a restricted system we have two electrons per orbital
            dm_localised_full_system = (
                localized_system.c_loc_occ @ localized_system.c_loc_occ.conj().T
            )
            dm_sum = localized_system.dm_active + localized_system.dm_enviro
            density_match = np.allclose(2 * dm_localised_full_system, dm_sum)
            logger.debug(f"Restricted {density_match=}")
        case 3:
            dm_localised_full_system = (
                localized_system.c_loc_occ
                @ localized_system.c_loc_occ.conj().swapaxes(-1, -2)
            )
            dm_sum = localized_system.dm_active + localized_system.dm_enviro

            # both need to be correct
            alpha_density_match = np.allclose(dm_localised_full_system, dm_sum)
            logger.debug(f"Unrestricted {alpha_density_match=}")
            density_match = np.allclose(dm_localised_full_system, dm_sum)
            logger.debug(f"Unrestricted {density_match=}")
            density_match = alpha_density_match and density_match

    if not density_match:
        logger.error("Density matrix partition does not sum to total.")
        warn_flag = True

    # check number of electrons is still the same after orbitals have been localized (change of basis)
    logger.debug("Checking electron number conserverd.")
    s_ovlp = global_scf.get_ovlp()

    match localized_system.dm_active.ndim:
        case 2:
            n_active_electrons = np.trace(localized_system.dm_active @ s_ovlp)
            n_enviro_electrons = np.trace(localized_system.dm_enviro @ s_ovlp)

        case 3:
            n_active_electrons = np.trace(localized_system.dm_active[0] @ s_ovlp)
            n_enviro_electrons = np.trace(localized_system.dm_enviro[0] @ s_ovlp)
            n_active_electrons += np.trace(localized_system.dm_active[1] @ s_ovlp)
            n_enviro_electrons += np.trace(localized_system.dm_enviro[1] @ s_ovlp)

    n_all_electrons = global_scf.mol.nelectron
    electron_number_match = np.isclose(
        (n_active_electrons + n_enviro_electrons), n_all_electrons
    )
    logger.debug(f"{electron_number_match=}")
    if not electron_number_match:
        logger.error("Number of electrons in localized orbitals is not consistent.")
        logger.debug(f"N total electrons: {n_all_electrons}")
        warn_flag = True

    if warn_flag:
        logger.error("Localizer sense check failed.")
        raise NbedLocalizerError("Localizer sense check failed.\n")
    else:
        logger.debug("Localizer sense check passed.")
