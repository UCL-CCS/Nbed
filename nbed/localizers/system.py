"""Class defining the data output from Localizers."""

from dataclasses import dataclass, field

from numpy.typing import NDArray


@dataclass
class LocalizedSystem:
    """Required data from localized system.

    active_mo_inds (np.array): 1D array of active occupied MO indices
    enviro_mo_inds (np.array): 1D array of environment occupied MO indices
    c_active (np.array): C matrix of localized occupied active MOs (columns define MOs)
    c_enviro (np.array): C matrix of localized occupied ennironment MOs
    c_loc_occ (np.array): C matrix of localized occupied MOs
    c_loc_virt (np.array | None): C matrix of localized virual MOs.
    dm_active (np.array): active system density matrix
    dm_enviro (np.array): environment system density matrix
    """

    active_mo_inds: NDArray
    enviro_mo_inds: NDArray
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
