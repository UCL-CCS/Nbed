"""ACE of SPADE localization."""

import numpy as np
from pyscf import gto
from scipy.optimize import curve_fit, minimize

from .spade import SPADELocalizer


class ACELocalizer:
    """Implements ACE of SPADE along coordinate path."""

    def __init__(
        self,
        global_scf_list: gto.Mole,
        n_active_atoms: int,
        max_shells: int = 4,
    ):
        """Initialize."""
        self.global_scf_list = global_scf_list
        self.n_active_atoms = n_active_atoms
        self.max_shells = max_shells

    def localize_path(self) -> int:
        """Find the number of MOs to use over the reaction coordinates.

        NOTE: Only returns one number for both spins
        """
        localized_systems = []
        for scf_object in self.global_scf_list:
            localized_systems += SPADELocalizer(
                scf_object, self.n_active_atoms, self.max_shells
            )

        singular_values = [loc.enviro_selection_condition for loc in localized_systems]

        def fermi_dist(diff_i_max, beta):
            return (
                beta
                * np.exp(beta * diff_i_max)
                / (1 + np.exp(beta * diff_i_max)) ** 1.5
            )

        max_vals = []
        for val_set in singular_values:
            diffs = singular_values[:-1] - singular_values[1:]
            max_i = np.argmax(diffs)
            diff_i_max = [i - max_i for i in range(len(val_set))]
            beta_fit = curve_fit(fermi_dist, diff_i_max, val_set)

            def neg_fermi_dist(diff_i_max):
                return -1 * fermi_dist(diff_i_max, beta_fit)

            res = minimize(neg_fermi_dist, max_i)
            max_vals += res.x

        mean_max = np.mean(max_vals)
        mean_max = int(mean_max)

        return [
            SPADELocalizer(
                scf_object,
                self.n_active_atoms,
                self.max_shells,
                n_mo_overwrite=mean_max,
            )
            for scf_object in self.global_scf_list
        ]
