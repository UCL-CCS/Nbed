"""Class defining the data output from Localizers."""

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy import dtype

logger = logging.getLogger(__name__)


@dataclass
class LocalizedSystem:
    """Required data from localized system.

    active_MO_inds (np.array): 1D array of active occupied MO indices
    enviro_MO_inds (np.array): 1D array of environment occupied MO indices
    c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
    c_enviro (np.array): C matrix of localized occupied ennironment MOs
    c_loc_occ (np.array): C matrix of localized occupied MOs
    c_loc_virt (np.array | None): C matrix of localized virual MOs.
    dm_active (np.array): active system density matrix
    dm_enviro (np.array): environment system density matrix
    """

    active_MO_inds: np.ndarray[tuple[int, ...], dtype[np.bool]]
    enviro_MO_inds: np.ndarray[tuple[int, ...], dtype[np.bool]]
    c_loc_occ: np.ndarray[tuple[int, ...], dtype[np.floating]]
    c_active: np.ndarray[tuple[int, ...], dtype[np.floating]]
    c_enviro: np.ndarray[tuple[int, ...], dtype[np.floating]]
    c_loc_virt: np.ndarray[tuple[int, ...], dtype[np.floating]] | None = None
    dm_active: np.ndarray[tuple[int, ...], dtype[np.floating]] = field(init=False)
    dm_enviro: np.ndarray[tuple[int, ...], dtype[np.floating]] = field(init=False)
    dm_loc_occ: np.ndarray[tuple[int, ...], dtype[np.floating]] = field(init=False)

    def __post_init__(self):
        """Post init for derived attributes."""
        self.dm_loc_occ = self.c_loc_occ @ self.c_loc_occ.swapaxes(-1, -2)
        self.dm_active = self.c_active @ self.c_active.swapaxes(-1, -2)
        self.dm_enviro = self.c_enviro @ self.c_enviro.swapaxes(-1, -2)

        # if self.c_loc_occ.ndim == 2:
        #     self.dm_active *= 2
        #     self.dm_enviro *= 2
        #     self.dm_loc_occ *= 2

        logger.debug("LocalizedSystem created.")
        logger.debug(f"{self.active_MO_inds}")
        logger.debug(f"{self.enviro_MO_inds}")
        logger.debug(f"{self.c_loc_occ.shape=}")
        logger.debug(f"{self.dm_active.shape=}")
        logger.debug(f"{self.dm_enviro.shape=}")

    @staticmethod
    def unrestricted_from_spin_components(
        alpha: "LocalizedSystem", beta: "LocalizedSystem"
    ) -> "LocalizedSystem":
        """Construct a spin-aware LocalizedSystem from two spinless ones.

        Args:
            alpha (LocalizedSystem): The localized alpha spins
            beta (LocalizedSystem): The localized beta spins.

        Returns:
            LocalizedSystem: A combined localized system with spins (alpha, beta).
        """
        logger.debug("Creating LocalizedSystem from spin components.")
        active_MO_inds = np.array([alpha.active_MO_inds, beta.active_MO_inds])
        enviro_MO_inds = np.array([alpha.enviro_MO_inds, beta.enviro_MO_inds])
        c_active = np.array([alpha.c_active, beta.c_active])
        c_enviro = np.array([alpha.c_enviro, beta.c_enviro])
        c_loc_occ = np.array([alpha.c_loc_occ, beta.c_loc_occ])

        if alpha.c_loc_virt is not None and beta.c_loc_virt is not None:
            c_loc_virt = np.array([alpha.c_loc_virt, beta.c_loc_virt])
        else:
            c_loc_virt = None

        return LocalizedSystem(
            active_MO_inds,
            enviro_MO_inds,
            c_loc_occ,
            c_active,
            c_enviro,
            c_loc_virt,
        )
