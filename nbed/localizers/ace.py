"""ACE of SPADE localization.

Based on 10.1021/acs.jctc.3c00653
"""

import logging

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.optimize import curve_fit, minimize

from nbed.config import NbedConfig, XYZGeometry, parse_config
from nbed.driver import NbedDriver

logger = logging.getLogger(__name__)


def _get_spade_singular_values(
    occupancy: NDArray, mo_coeff: NDArray, n_act_aos: int, ao_overlap: NDArray
):
    """Find the singular values for a SPADE localization of a single spin.

    Args:
        occupancy (NDArray): Occupancy of molecular orbitals.
        mo_coeff (NDArray): Molecular orbital coefficients.
        n_act_aos (int): Number of active atomic orbitals.
        ao_overlap (NDArray): The Atomic Orbital overlap matrix.

    Returns:
        NDArray: Singular values, used to partition a system in SPADE.
    """
    n_occupied_orbitals = np.count_nonzero(occupancy)
    occupied_orbitals = mo_coeff[:, :n_occupied_orbitals]
    rotated_orbitals = (
        linalg.fractional_matrix_power(ao_overlap, 0.5) @ occupied_orbitals
    )
    _, sigma, _ = linalg.svd(rotated_orbitals[:n_act_aos, :])
    return sigma


def _fermi_dist(diff_i_max: int, beta: float) -> float:
    """Fermi Distribution Function.

    Args:
        diff_i_max (int): Relative distance from the maximum value.
        beta (float): Fitting parameter.

    Returns:
        float: Function value.
    """
    return beta * np.exp(beta * diff_i_max) / (1 + np.exp(beta * diff_i_max)) ** 1.5


def _best_fit_nmo(all_singular_values: list[list[float]]) -> tuple[int, int]:
    """Find the best fit number of molecular orbitals.

    Args:
        all_singular_values (list[list[float]]): Singular values for each geometry.
    Reurns:
        tuple[int, int]: number of (alpha, beta) molecular orbitals.
    """
    max_vals = []
    for geometry_singular_values in all_singular_values:
        logger.debug(f"{geometry_singular_values=}")
        diffs = np.array(geometry_singular_values[:-1]) - np.array(
            geometry_singular_values[1:]
        )
        max_i = np.argmax(diffs)

        diff_i_max = [i - max_i for i in range(len(geometry_singular_values))]
        logger.debug(f"{diff_i_max=}")

        beta_fit, _ = curve_fit(_fermi_dist, diff_i_max, geometry_singular_values)
        logger.debug(f"{beta_fit=}")

        def neg_fermi_dist(diff_i_max):
            return -1 * _fermi_dist(diff_i_max, beta_fit)

        res = minimize(neg_fermi_dist, max_i)
        max_vals.append(res.x[0])
        logger.debug(f"{max_vals=}")

    mean_max = np.mean(max_vals)
    # we want to round to the nearesrt 1, we cam do this with int(val+0.5)
    nmo = mean_max + np.argwhere(diff_i_max == np.int64(0)) + 0.5
    nmo = int(nmo) + 1
    logger.debug(f"Using {nmo} Molecular Orbitals")
    return nmo


def ace_of_spade(
    geometries: list[XYZGeometry], config: NbedConfig, split_spins: bool = False
) -> tuple[int, int]:
    """Find the number of MOs to use over the reaction coordinates.

    Args:
        geometries (list[XYZGeometry]): A list of valid XYZ geometry strings.
        config (NbedConfig): A validated config model.
        split_spins (bool): When true, ACE-of-SPADE will be run for each spin independently.

    Returns:
        tuple(int,int): Number of molecular orbitals for spin alpha, beta.

    Note: For restricted systems, a tuple of (equal) values is still given.
    """
    logger.debug("Running ACE of SPADE across reaction coordinates.")

    alpha_singular_values: list[tuple[float]] = []
    beta_singular_values: list[tuple[float]] = []
    for geo in geometries:
        config = parse_config(config, geometry=geo)
        driver = NbedDriver(config=config)
        global_ks = driver._global_ks()

        occupancy = global_ks.mo_occ
        n_act_aos = driver._global_ks.mol.aoslice_by_atom()[config.n_active_atoms - 1][
            -1
        ]
        ao_overlap = driver._global_ks.get_ovlp()

        match global_ks.mo_coeff.ndim:
            case 2:
                alpha = _get_spade_singular_values(
                    occupancy, global_ks.mo_coeff, n_act_aos, ao_overlap
                )
            case 3:
                alpha = _get_spade_singular_values(
                    occupancy[0], global_ks.mo_coeff[0, :, :], n_act_aos, ao_overlap
                )
                beta = _get_spade_singular_values(
                    occupancy[1], global_ks.mo_coeff[1, :, :], n_act_aos, ao_overlap
                )
                beta_singular_values.append(beta)

        alpha_singular_values.append(alpha)

    if beta_singular_values == []:
        alpha_nmos = _best_fit_nmo(alpha_singular_values)
        beta_nmos = alpha_nmos
    elif split_spins is False:
        alpha_nmos = _best_fit_nmo(alpha_singular_values + beta_singular_values)
        beta_nmos = alpha_nmos
    else:
        alpha_nmos = _best_fit_nmo(alpha_singular_values)
        beta_nmos = _best_fit_nmo(beta_singular_values)

    return (alpha_nmos, beta_nmos)
