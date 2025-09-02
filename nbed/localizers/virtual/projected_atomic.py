"""Projected Atomic Orbitals."""

import logging

import numpy as np
from numpy.typing import NDArray
from pyscf.lib import StreamObject

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
                    self.global_scf.get_fock(),
                    self.norm_cutoff,
                    self.overlap_cutoff,
                )
                logger.debug(f"{virtuals.shape=}")
                n_aos = self.global_scf.mo_coeff.shape[0]
                n_virtuals = np.sum(self.global_scf.mo_occ == 0)
                n_empty_virtuals = n_virtuals - virtuals.shape[-1]
                virtuals = np.hstack((virtuals, np.zeros(n_aos, n_empty_virtuals)))

            case 3:  # Restricted open shell
                fock_alpha, fock_beta = self.global_scf.get_fock()
                logger.debug("Running PAO for each spin separately.")
                alpha_virtuals = _localize_virtual_spin_pao(
                    self.c_loc_occ[0],
                    ao_overlap,
                    n_act_aos,
                    fock_alpha,
                    self.norm_cutoff,
                    self.overlap_cutoff,
                )
                beta_virtuals = _localize_virtual_spin_pao(
                    self.c_loc_occ[1],
                    ao_overlap,
                    n_act_aos,
                    fock_beta,
                    self.norm_cutoff,
                    self.overlap_cutoff,
                )
                logger.debug(f"{alpha_virtuals.shape=}")
                logger.debug(f"{beta_virtuals.shape=}")
                n_aos = self.global_scf.mo_coeff.shape[-2]
                # TODO this isn't going to work, fix after the weekend.
                alpha_empty_virtuals = (
                    np.sum(self.global_scf.mo_occ[0]) - alpha_virtuals.shape[-1]
                )
                beta_empty_virtuals = (
                    np.sum(self.global_scf.mo_occ[1]) - beta_virtuals.shape[-1]
                )
                alpha_virtuals = np.hstack(
                    (alpha_virtuals, np.zeros(n_aos, alpha_empty_virtuals))
                )
                beta_virtuals = np.hstack(
                    (beta_virtuals, np.zeros(n_aos, beta_empty_virtuals))
                )

                virtuals = np.array([alpha_virtuals, beta_virtuals])

        return virtuals

    # where should the cutoff values come from?


def _localize_virtual_spin_pao(
    c_loc_occ: NDArray,
    ao_overlap: NDArray,
    n_act_aos: int,
    fock: NDArray,
    norm_cutoff: float = 0.05,
    overlap_cutoff: float = 1e-5,
) -> NDArray:
    """Localize a single spin using Projected Atomic Orbitals.

    Returns:
        NDArray: The localized atomic orbitals
    """
    logger.debug("Calculating Projected Atomic Orbitals.")
    logger.debug(f"{n_act_aos=}")
    logger.debug(f"{norm_cutoff=}")
    logger.debug(f"{overlap_cutoff=}")
    pao_projector = (
        np.identity(ao_overlap.shape[-1]) - c_loc_occ @ c_loc_occ.T @ ao_overlap
    )

    # Seems like in the paper they do indices (MOs, AOs?)
    logger.debug(f"{pao_projector[:n_act_aos].shape=}")
    pao_norms = np.einsum(
        "ji,ji->i",
        pao_projector[:n_act_aos],
        (ao_overlap @ pao_projector)[:n_act_aos],
    )
    # do we need to scale the norm by the system size?
    # pao_norms /= n_act_aos
    logger.debug(f"{pao_norms=}")

    # Take the columns of C matrix (MOs)
    truncated_paos = pao_projector[:, np.abs(pao_norms) > norm_cutoff]

    renormalized_paos = truncated_paos
    renormalized_paos = renormalized_paos / np.sqrt(
        np.einsum("ij,ij->j", renormalized_paos, renormalized_paos)
    )

    diagonalized_overlap = renormalized_paos.T @ ao_overlap @ renormalized_paos

    eigvals, eigvecs = np.linalg.eigh(diagonalized_overlap)

    logger.debug(f"Overlap eigenvalues {eigvals}")

    logger.debug(f"{eigvecs.shape=}")
    # How to transform the truncated paos?
    reduced_paos = renormalized_paos @ eigvecs[:, np.abs(eigvals) > overlap_cutoff]
    reduced_paos /= eigvals[np.abs(eigvals) > overlap_cutoff]
    logger.debug(f"{reduced_paos.shape=}")

    # canonicalise
    _, rotation = np.linalg.eigh(reduced_paos.T @ fock @ reduced_paos)
    final_paos = reduced_paos @ rotation

    if (n_paos := final_paos.shape[-1]) == 0:
        logger.warning("No projected atomic orbitals!")
        logger.warning(
            "This suggests your active region has no virtual Atomic Orbitals."
        )
    else:
        logger.info("Complete virtual localisation with PAO")
        logger.info(f"{n_paos=}")
    return final_paos
