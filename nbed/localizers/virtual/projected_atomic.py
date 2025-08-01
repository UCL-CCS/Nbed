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
        embedded_scf: StreamObject,
        n_active_atoms: int,
        c_loc_occ: tuple[NDArray, NDArray | None],
        norm_cutoff: float = 0.05,
        overlap_cutoff=1e-5,
    ):
        """Init PAO Localizer."""
        super().__init__(embedded_scf, n_active_atoms)
        self.norm_cutoff = norm_cutoff
        self.overlap_cutoff = overlap_cutoff
        self.c_loc_occ = c_loc_occ

    def localize_virtual(self) -> StreamObject:
        """Run projected atomic orbitals localization."""
        n_act_aos = self.embedded_scf.mol.aoslice_by_atom()[self._n_active_atoms - 1][
            -1
        ]
        ao_overlap = self.embedded_scf.get_ovlp()

        if self.c_loc_occ[1] is None:
            logger.debug("Runing PAO for spinless system.")
            virtuals = _localize_virtual_spin_pao(
                self.c_loc_occ[0],
                ao_overlap,
                n_act_aos,
                self.norm_cutoff,
                self.overlap_cutoff,
            )
            logger.debug(f"{virtuals.shape=}")
            occ_mo_coeff = self.embedded_scf.mo_coeff[:, self.embedded_scf.mo_occ > 0]
            mo_coeff = np.hstack((occ_mo_coeff, virtuals))

            n_mos = mo_coeff.shape[-1]
            n_occ = np.count_nonzero(self.embedded_scf.mo_occ[0])
            mo_occ = np.array(n_occ * [1] + (n_mos - n_occ) * [0])

        else:  # Restricted open shell
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
            alpha_occ_mo_coeff = self.embedded_scf.mo_coeff[0][
                :, self.embedded_scf.mo_occ[0] > 0
            ]
            beta_occ_mo_coeff = self.embedded_scf.mo_coeff[1][
                :, self.embedded_scf.mo_occ[1] > 0
            ]
            mo_coeff = np.array(
                [
                    np.hstack((alpha_occ_mo_coeff, alpha_virtuals)),
                    np.hstack((beta_occ_mo_coeff, beta_virtuals)),
                ]
            )
            n_mos = mo_coeff.shape[-1]
            alpha_n_occ = np.count_nonzero(self.embedded_scf.mo_occ[0])
            beta_n_occ = np.count_nonzero(self.embedded_scf.mo_occ[1])
            mo_occ = np.vstack(
                (
                    np.array(alpha_n_occ * [1] + (n_mos - alpha_n_occ) * [0]),
                    np.array(beta_n_occ * [1] + (n_mos - beta_n_occ) * [0]),
                )
            )

        self.embedded_scf.mo_coeff = mo_coeff
        self.embedded_scf.mo_occ = mo_occ

        return self.embedded_scf

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
    # logger.debug(f"{c_loc_occ @ c_loc_occ.T @ ao_overlap}")
    pao_projector = (
        np.identity(ao_overlap.shape[-1]) - c_loc_occ @ c_loc_occ.T @ ao_overlap
    )
    logger.debug(f"{pao_projector[0]=}")

    # Seems like in the paper they do indices (MOs, AOs?)
    logger.debug(f"{n_act_aos=}")
    logger.debug(f"{pao_projector[:n_act_aos].shape=}")
    pao_norms = np.einsum(
        "ji,ji->i",
        pao_projector[:n_act_aos],
        ao_overlap[:n_act_aos, :n_act_aos] @ pao_projector[:n_act_aos],
    )
    logger.debug(f"{pao_norms=}")
    logger.debug(len(pao_norms))
    # Take the columns of C matrix (MOs)
    truncated_paos = pao_projector[:, np.abs(pao_norms) > norm_cutoff]
    logger.debug(f"{truncated_paos.shape=}")
    logger.debug(f"{truncated_paos=}")

    s_half = fractional_matrix_power(ao_overlap, 0.5)

    renormalized_paos = s_half @ truncated_paos
    logger.debug(f"{renormalized_paos=}")
    logger.debug(f"{renormalized_paos.shape=}")
    logger.debug(f"{np.einsum("ij,ij->j", renormalized_paos, renormalized_paos)=}")
    renormalized_paos = renormalized_paos / np.sqrt(
        np.einsum("ij,ij->j", renormalized_paos, renormalized_paos)
    )
    logger.debug(
        f"{np.einsum("ij,ij->j", renormalized_paos, np.conj(renormalized_paos))=}"
    )

    diagonalized_overlap = renormalized_paos.T @ ao_overlap @ renormalized_paos

    logger.debug(f"{diagonalized_overlap.shape=}")
    logger.debug(f"{diagonalized_overlap=}")

    eigvals, eigvecs = np.linalg.eigh(diagonalized_overlap)

    logger.debug(f"Overlap eigenvalues {eigvals}")
    logger.debug(f"{overlap_cutoff=}")

    logger.debug(f"{eigvecs.shape=}")
    # How to transform the truncated paos?
    final_paos = renormalized_paos[:, eigvals > overlap_cutoff]
    logger.debug(f"{final_paos=}")

    if final_paos.shape[-1] == 0:
        logger.warning("No projected atomic orbitals!")
        logger.warning(
            "This suggests your active region has no virtual Atomic Orbitals."
        )
    return final_paos
