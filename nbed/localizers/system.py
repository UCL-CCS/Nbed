"""Class defining the data output from Localizers."""

import logging
from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np
from numpy import dtype

type OneSpinMatrix[M: int] = np.ndarray[tuple[M, M], np.dtype[np.floating]]
type TwoSpinMatrix[M: int] = np.ndarray[tuple[Literal[2], M, M], np.dtype[np.floating]]
type AnySpinMatrix[M: int] = Union[OneSpinMatrix[M], TwoSpinMatrix[M]]

logger = logging.getLogger(__name__)


@dataclass
class LocalizedSystem:
    """Required data from localized system.

    active_occ_inds (np.array): 1D array of active occupied MO indices
    enviro_occ_inds (np.array): 1D array of environment occupied MO indices
    c_loc_occ (np.array): C matrix of localized occupied MOs
    dm_active (np.array): active system density matrix
    dm_enviro (np.array): environment system density matrix
    c_loc_virt (np.array | None): C matrix of localized virual MOs.
    """

    active_occ_inds: np.ndarray
    enviro_occ_inds: np.ndarray
    c_loc_occ: np.ndarray
    dm_active: np.ndarray
    dm_enviro: np.ndarray
    c_loc_virt: np.ndarray | None = None
    dm_loc_occ: np.ndarray = field(init=False)

    def __post_init__(self):
        """Post init for derived attributes."""
        self.dm_loc_occ = self.c_loc_occ @ self.c_loc_occ.swapaxes(-1, -2)

        logger.debug("LocalizedSystem created.")
        logger.debug(f"{self.active_occ_inds}")
        logger.debug(f"{self.enviro_occ_inds}")
        logger.debug(f"{self.c_loc_occ.shape=}")
        logger.debug(f"{self.dm_active.shape=}")
        logger.debug(f"{self.dm_enviro.shape=}")


@dataclass
class RestrictedLS(LocalizedSystem):
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

    active_occ_inds: np.ndarray[tuple[int], dtype[np.bool]]
    enviro_occ_inds: np.ndarray[tuple[int], dtype[np.bool]]
    c_loc_occ: OneSpinMatrix
    dm_active: OneSpinMatrix
    dm_enviro: OneSpinMatrix
    c_loc_virt: OneSpinMatrix | None = None
    dm_loc_occ: OneSpinMatrix = field(init=False)

    def __post_init__(self):
        """post-init."""
        super().__post_init__()


@dataclass
class RestrictedOpenLS(LocalizedSystem):
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

    active_occ_inds: np.ndarray[tuple[int, int], dtype[np.bool]]
    enviro_occ_inds: np.ndarray[tuple[int, int], dtype[np.bool]]
    c_loc_occ: OneSpinMatrix
    dm_active: OneSpinMatrix
    dm_enviro: OneSpinMatrix
    c_loc_virt: OneSpinMatrix | None = None
    dm_loc_occ: OneSpinMatrix = field(init=False)

    def __post_init__(self):
        """post-init."""
        super().__post_init__()


@dataclass
class UnrestrictedLS(LocalizedSystem):
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

    active_occ_inds: np.ndarray[tuple[int, int], dtype[np.bool]]
    enviro_occ_inds: np.ndarray[tuple[int, int], dtype[np.bool]]
    c_loc_occ: TwoSpinMatrix
    dm_active: TwoSpinMatrix
    dm_enviro: TwoSpinMatrix
    c_loc_virt: TwoSpinMatrix | None = None
    dm_loc_occ: TwoSpinMatrix = field(init=False)

    def __post_init__(self):
        """post-init."""
        super().__post_init__()

    @staticmethod
    def from_spin_components(
        alpha: RestrictedLS, beta: RestrictedLS
    ) -> "UnrestrictedLS":
        """Construct a spin-aware LocalizedSystem from two spinless ones.

        Args:
            alpha (LocalizedSystem): The localized alpha spins
            beta (LocalizedSystem): The localized beta spins.

        Returns:
            LocalizedSystem: A combined localized system with spins (alpha, beta).
        """
        logger.debug("Creating LocalizedSystem from spin components.")
        active_occ_inds = np.array([alpha.active_occ_inds, beta.active_occ_inds])
        enviro_occ_inds = np.array([alpha.enviro_occ_inds, beta.enviro_occ_inds])
        dm_active: TwoSpinMatrix = np.array([alpha.dm_active, beta.dm_active])
        dm_enviro: TwoSpinMatrix = np.array([alpha.dm_enviro, beta.dm_enviro])
        c_loc_occ: TwoSpinMatrix = np.array([alpha.c_loc_occ, beta.c_loc_occ])

        if alpha.c_loc_virt is not None and beta.c_loc_virt is not None:
            c_loc_virt = np.array([alpha.c_loc_virt, beta.c_loc_virt])
        else:
            c_loc_virt = None

        return UnrestrictedLS(
            active_occ_inds,
            enviro_occ_inds,
            c_loc_occ,
            dm_active,
            dm_enviro,
            c_loc_virt,
        )
