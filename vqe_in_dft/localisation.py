"""
File to contain localisations.
"""

from typing import Callable, Tuple

import numpy as np
from scipy import linalg
import logging

logger = logging.getLogger(__name__)


def spade(
    scf_method: Callable, active_atoms: int
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Localise orbitals using SPADE.
    """
    logger.info("Localising with SPADE.")
    n_occupied_orbitals = np.count_nonzero(scf_method.mo_occ == 2)
    occupied_orbitals = scf_method.mo_coeff[:, :n_occupied_orbitals]

    n_act_aos = scf_method.mol.aoslice_by_atom()[active_atoms - 1][-1]
    logger.debug(f"{n_act_aos} active AOs.")

    ao_overlap = scf_method.get_ovlp()

    # Orbital rotation and partition into subsystems A and B
    # rotation_matrix, sigma = embed.orbital_rotation(occupied_orbitals,
    #    n_act_aos, ao_overlap)

    rotated_orbitals = (
        linalg.fractional_matrix_power(ao_overlap, 0.5) @ occupied_orbitals
    )
    _, sigma, right_vectors = linalg.svd(rotated_orbitals[:n_act_aos, :])

    logger.debug(f"Singular Values: {sigma}")

    # n_act_mos, n_env_mos = embed.orbital_partition(sigma)
    value_diffs = sigma[:-1] - sigma[1:]
    n_act_mos = np.argmax(value_diffs) + 1
    n_env_mos = n_occupied_orbitals - n_act_mos
    logger.debug(f"{n_act_mos} active MOs.")
    logger.debug(f"{n_env_mos} environment MOs.")

    # Defining active and environment orbitals and density
    act_orbitals = occupied_orbitals @ right_vectors.T[:, :n_act_mos]
    env_orbitals = occupied_orbitals @ right_vectors.T[:, n_act_mos:]
    act_density = 2.0 * act_orbitals @ act_orbitals.T
    env_density = 2.0 * env_orbitals @ env_orbitals.T
    return n_act_mos, n_env_mos, act_density, env_density


def mullikan(
    scf_method: Callable, active_atoms: int
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Localise orbitals using Mullikan population analysis.
    """
    raise NotImplementedError("Mullikan localisation is not implemented, use spade.")


def boys(
    scf_method: Callable, active_atoms: int
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    raise NotImplementedError("Boys localisation is not implemented, use spade.")


def ibo(
    scf_method: Callable, active_atoms: int
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    raise NotImplementedError("IBO localisation is not implemented, use spade.")
