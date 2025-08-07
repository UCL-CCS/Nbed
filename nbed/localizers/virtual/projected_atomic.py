"""Projected Atomic Orbitals."""

import logging

import numpy as np
from numpy.typing import NDArray
from pyscf.lib import StreamObject
from scipy.linalg import fractional_matrix_power

from nbed.localizers.virtual.base import VirtualLocalizer

logger = logging.getLogger(__name__)


class PAOLocalizer(VirtualLocalizer):
    """Projected Atomic Orbitals Localizer."""

    def __init__(
        self,
        global_scf: StreamObject,
        n_active_atoms: int,
        c_loc_occ: NDArray,
        norm_cutoff: float = 0.05,
        overlap_cutoff=1e-5,
    ):
        """Init PAO Localizer."""
        super().__init__(n_active_atoms)
        self.global_scf = global_scf
        self.norm_cutoff = norm_cutoff
        self.overlap_cutoff = overlap_cutoff
        self.c_loc_occ = c_loc_occ

    def localize_virtual(self) -> StreamObject:
        """Run projected atomic orbitals localization."""
        n_act_aos = self.global_scf.mol.aoslice_by_atom()[self._n_active_atoms - 1][-1]
        ao_overlap = self.global_scf.get_ovlp()

        match self.c_loc_occ.ndim:
            case 2:
                logger.debug("Runing PAO for spinless system.")
                virtuals = _localize_virtual_spin_pao(
                    self.c_loc_occ[0],
                    ao_overlap,
                    n_act_aos,
                    self.norm_cutoff,
                    self.overlap_cutoff,
                )
                logger.debug(f"{virtuals.shape=}")

            case 3:  # Restricted open shell
                logger.debug("Running PAO for each spin separately.")
                alpha_virtuals = _localize_virtual_spin_pao(
                    self.c_loc_occ[0],
                    ao_overlap,
                    n_act_aos,
                    self.norm_cutoff,
                    self.overlap_cutoff,
                )
                beta_virtuals = _localize_virtual_spin_pao(
                    self.c_loc_occ[1],
                    ao_overlap,
                    n_act_aos,
                    self.norm_cutoff,
                    self.overlap_cutoff,
                )
                logger.debug(f"{alpha_virtuals.shape=}")
                logger.debug(f"{beta_virtuals.shape=}")
                virtuals = np.array([alpha_virtuals, beta_virtuals])

        return virtuals

    # where should the cutoff values come from?


def _localize_virtual_spin_pao(
    c_loc_occ: NDArray,
    ao_overlap: NDArray,
    n_act_aos: int,
    norm_cutoff: float = 0.05,
    overlap_cutoff: float = 1e-5,
) -> NDArray:
    """Localize a single spin using Projected Atomic Orbitals.

    Returns:
        NDArray: The localized atomic orbitals
    """
    logger.debug("Calculating Projected Atomic Orbitals.")
    pao_projector = (
        np.identity(ao_overlap.shape[-1]) - c_loc_occ @ c_loc_occ.T @ ao_overlap
    )

    # Seems like in the paper they do indices (MOs, AOs?)
    logger.debug(f"{n_act_aos=}")
    logger.debug(f"{pao_projector[:n_act_aos].shape=}")
    pao_norms = np.einsum(
        "ji,ji->i",
        pao_projector[:n_act_aos],
        ao_overlap[:n_act_aos, :n_act_aos] @ pao_projector[:n_act_aos],
    )
    logger.debug(f"{pao_norms=}")

    # Take the columns of C matrix (MOs)
    truncated_paos = pao_projector[:, np.abs(pao_norms) > norm_cutoff]

    s_half = fractional_matrix_power(ao_overlap, 0.5)

    renormalized_paos = s_half @ truncated_paos
    renormalized_paos = renormalized_paos / np.sqrt(
        np.einsum("ij,ij->j", renormalized_paos, renormalized_paos)
    )

    diagonalized_overlap = renormalized_paos.T @ ao_overlap @ renormalized_paos

    eigvals, eigvecs = np.linalg.eigh(diagonalized_overlap)

    logger.debug(f"Overlap eigenvalues {eigvals}")
    logger.debug(f"{overlap_cutoff=}")

    logger.debug(f"{eigvecs.shape=}")
    # How to transform the truncated paos?
    final_paos = renormalized_paos[:, eigvals > overlap_cutoff]
    logger.debug(f"{final_paos=}")

    if (n_paos := final_paos.shape[-1]) == 0:
        logger.warning("No projected atomic orbitals!")
        logger.warning(
            "This suggests your active region has no virtual Atomic Orbitals."
        )
    else:
        logger.info("Complete virtual localisation with PAO")
        logger.info(f"{n_paos=}")
    return final_paos
