"""ACE of SPADE localization.

Based on 10.1021/acs.jctc.3c00653
"""

import logging

import numpy as np
from pyscf import dft, lib, scf
from scipy.optimize import curve_fit, minimize

from nbed.localizers.spade import SPADELocalizer

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

    def localize_path(self) -> tuple[int, int]:
        """Find the number of MOs to use over the reaction coordinates.

        NOTE: Only returns one number for both spins.

        Returns:
            tuple(int,int): Number of molecular orbitals for spin alpha, beta.
        """
        logger.debug("Running ACE of SPADE across reaction coordinates.")
        localized_systems = []
        for scf_object in self.global_scf_list:
            loc = SPADELocalizer(scf_object, self.n_active_atoms, self.max_shells)
            localized_systems.append(loc)

        # only does restricted atm
        singular_values = [loc.enviro_selection_condition for loc in localized_systems]

        if isinstance(scf_object, (scf.hf.RHF, dft.rks.RKS)):
            alpha = self.localize_spin([s[0] for s in singular_values])
            beta = alpha
        elif isinstance(scf_object, (scf.uhf.UHF, dft.uks.UKS)):
            alpha = self.localize_spin([s[0] for s in singular_values])
            beta = self.localize_spin([s[1] for s in singular_values])
        else:
            error_string = f"SCF object of type {type(scf_object)} cannot be used."
            logger.error(error_string)
            raise TypeError(error_string)
        return (alpha, beta)

    def localize_spin(self, singular_values) -> int:
        """Run ACE of SPADE for a single spin.

        Args:
            singular_values (np.ndarray[float]): Singular values from SPADE for each geometry.

        Returns:
            int: Numer of Molecular Orbitals to use.
        """
        logger.debug("Running ACE of SPADE for a single spin.")

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

            beta_fit, beta_cov = curve_fit(fermi_dist, diff_i_max, val_set)
            logger.debug(f"{beta_fit=}")

            def neg_fermi_dist(diff_i_max):
                return -1 * fermi_dist(diff_i_max, beta_fit)

            res = minimize(neg_fermi_dist, max_i)
            max_vals.append(res.x)
            logger.debug(f"{max_vals=}")

        mean_max = np.mean(max_vals)
        # we want to round to the nearesrt 1, we cam do this with int(val+0.5)
        nmo = mean_max + np.argwhere(diff_i_max == np.int64(0)) + 0.5
        nmo = int(nmo)
        logger.debug(f"Using {nmo} Molecular Orbitals")
        return nmo
