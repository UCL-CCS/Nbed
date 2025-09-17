"""Class defining the data output from Localizers."""

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class LocalizedSystem:
    """Required data from localized system.

    active_occ_inds (np.array[bool]): 1D array, true for active occupied MO indices
    enviro_occ_inds (np.array[bool]): 1D array, true for of environment occupied MO indices
    c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
    c_enviro (np.array): C matrix of localized occupied ennironment MOs
    c_loc_occ (np.array): C matrix of localized occupied MOs
    c_loc_virt (np.array | None): C matrix of localized virual MOs.
    dm_active (np.array): active system density matrix
    dm_enviro (np.array): environment system density matrix
    """

    active_occ_inds: NDArray[np.bool]
    enviro_occ_inds: NDArray[np.bool]
    c_active: NDArray
    c_enviro: NDArray
    c_loc_occ: NDArray
    c_loc_virt: NDArray | None = None
    dm_active: NDArray = field(init=False)
    dm_enviro: NDArray = field(init=False)
    dm_loc_occ: NDArray = field(init=False)

    def __post_init__(self):
        """Post init for derived attributes."""
        self.dm_active = self.c_active @ self.c_active.swapaxes(-1, -2)
        self.dm_enviro = self.c_enviro @ self.c_enviro.swapaxes(-1, -2)
        self.dm_loc_occ = self.c_loc_occ @ self.c_loc_occ.swapaxes(-1, -2)

        # For spinless systems we need twice as many electrons
        # In a single density matrix.
        self.dm_active *= 2 if self.dm_active.ndim == 2 else 1
        self.dm_enviro *= 2 if self.dm_enviro.ndim == 2 else 1
        self.dm_loc_occ *= 2 if self.dm_loc_occ.ndim == 2 else 1

        logger.debug("LocalizedSystem built.")
        logger.debug(f"{self.active_occ_inds=}")
        logger.debug(f"{self.enviro_occ_inds=}")
        logger.debug(f"{self.c_active.shape=}")
        logger.debug(f"{self.c_enviro.shape=}")
        logger.debug(f"{self.c_loc_occ.shape=}")

    def from_spin_components(
        alpha: "LocalizedSystem", beta: "LocalizedSystem"
    ) -> "LocalizedSystem":
        """Construct a spin-aware LocalizedSystem from two spinless ones.

        Args:
            alpha (LocalizedSystem): The localized alpha spins
            beta (LocalizedSystem): The localized beta spins.

        Returns:
            LocalizedSystem: A combined localized system with spins (alpha, beta).
        """
        active_occ_inds = np.array([alpha.active_occ_inds, beta.active_occ_inds])
        enviro_occ_inds = np.array([alpha.enviro_occ_inds, beta.enviro_occ_inds])
        c_active = np.array([alpha.c_active, beta.c_active])
        c_enviro = np.array([alpha.c_enviro, beta.c_enviro])
        c_loc_occ = np.array([alpha.c_loc_occ, beta.c_loc_occ])
        c_loc_virt = np.array([alpha.c_loc_virt, beta.c_loc_virt])
        return LocalizedSystem(
            active_occ_inds, enviro_occ_inds, c_active, c_enviro, c_loc_occ, c_loc_virt
        )
