"""ACE of SPADE localization."""

import logging

import numpy as np
from pyscf import lib
from scipy.optimize import curve_fit, minimize

from .spade import SPADELocalizer

logger = logging.getLogger(__name__)


class ACELocalizer:
    """Implements ACE of SPADE along coordinate path."""

    def __init__(
        self,
        global_scf_list: lib.StreamObject,
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
        logger.debug("Running ACE of SPADE across reaction coordinates.")
        localized_systems = []
        for scf_object in self.global_scf_list:
            loc = SPADELocalizer(scf_object, self.n_active_atoms, self.max_shells)
            localized_systems.append(loc)

        # only does restricted atm
        singular_values = [
            loc.enviro_selection_condition[0] for loc in localized_systems
        ]

        def fermi_dist(diff_i_max, beta):
            return (
                beta
                * np.exp(beta * diff_i_max)
                / (1 + np.exp(beta * diff_i_max)) ** 1.5
            )

        max_vals = []
        for val_set in singular_values:
            diffs = np.array(singular_values[:-1]) - np.array(singular_values[1:])
            max_i = np.argmax(diffs)
            logger.debug(f"{diffs=}")
            logger.debug(f"{max_i=}")

            diff_i_max = [i - max_i for i in range(len(val_set))]
            logger.debug(f"{diff_i_max=}")

            beta_fit = curve_fit(fermi_dist, diff_i_max, val_set)
            logger.debug(f"{beta_fit=}")

            def neg_fermi_dist(diff_i_max):
                return -1 * fermi_dist(diff_i_max, beta_fit)

            res = minimize(neg_fermi_dist, max_i)
            max_vals.append(res.x)

        mean_max = np.mean(max_vals)
        mean_max = int(mean_max)

        return mean_max
