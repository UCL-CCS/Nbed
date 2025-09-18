"""Class defining the data output from Localizers."""

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class LocalizedSystem:
    """Required data from localized system.

    active_occ_inds (np.array): 1D array of active occupied MO indices
    enviro_occ_inds (np.array): 1D array of environment occupied MO indices
    c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
    c_enviro (np.array): C matrix of localized occupied ennironment MOs
    c_loc_occ (np.array): C matrix of localized occupied MOs
    c_loc_virt (np.array | None): C matrix of localized virual MOs.
    dm_active (np.array): active system density matrix
    dm_enviro (np.array): environment system density matrix
    """

    active_occ_inds: NDArray
    enviro_occ_inds: NDArray
    c_loc_occ: NDArray
    dm_active: NDArray
    dm_enviro: NDArray
    c_loc_virt: NDArray | None = None
    dm_loc_occ: NDArray = field(init=False)

    def __post_init__(self):
        """Post init for derived attributes."""
        self.dm_loc_occ = self.c_loc_occ @ self.c_loc_occ.swapaxes(-1, -2)

        if self.c_loc_occ.ndim == 2:
            self.dm_active *= 2
            self.dm_enviro *= 2
            self.dm_loc_occ *= 2

        logger.debug("LocalizedSystem created.")
        logger.debug(f"{self.active_occ_inds}")
        logger.debug(f"{self.enviro_occ_inds}")
        logger.debug(f"{self.c_loc_occ.shape=}")
        logger.debug(f"{self.dm_active.shape=}")
        logger.debug(f"{self.dm_enviro.shape=}")

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
        dm_active = 0.5 * np.array([alpha.dm_active, beta.dm_active])
        dm_enviro = 0.5 * np.array([alpha.dm_enviro, beta.dm_enviro])
        c_loc_occ = np.array([alpha.c_loc_occ, beta.c_loc_occ])
        if alpha.c_loc_virt is not None and beta.c_loc_virt is not None:
            c_loc_virt = np.array([alpha.c_loc_virt, beta.c_loc_virt])
        else:
            c_loc_virt = None
        return LocalizedSystem(
            active_occ_inds,
            enviro_occ_inds,
            c_loc_occ,
            dm_active,
            dm_enviro,
            c_loc_virt,
        )
