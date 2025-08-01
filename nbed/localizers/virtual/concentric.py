"""Concentric Localization of virtual orbitals."""

import logging

import numpy as np
from pyscf import gto, scf
from pyscf.lib import StreamObject

from .base import VirtualLocalizer

logger = logging.getLogger(__name__)


class ConcentricLocalizer(VirtualLocalizer):
    """Class to localize virtual orbitals using concentric localization.

    Attributes:
        max_shells (int): Maximum number of shells to localize.
        _n_active_atoms (int): Number of active atoms in the system.
        projected_overlap (np.ndarray): Projected overlap matrix.
        overlap_two_basis (np.ndarray): Overlap matrix between two basis sets.
        n_act_proj_aos (int): Number of active projected atomic orbitals.
        shells (list): List of shell sizes.
        singular_values (list): List of singular values from SVD.

    Methods:
        localize_virtual(StreamObject): Localize virtual orbitals using concentric localization.
        _localize_virtual_spin(np.ndarray, np.ndarray, np.ndarray): Run concentric localization for each spin separately.
    """

    def __init__(
        self,
        embedded_scf: StreamObject,
        n_active_atoms: int,
        max_shells: int = 4,
    ):
        """Initialize Concentric Localization object.

        Args:
            embedded_scf (StreamObject): SCF object with occupied orbitals localized.
            n_active_atoms (int): Number of active atoms in the system.
            max_shells (int): Maximum number of shells to localize.
        """
        super().__init__(embedded_scf, n_active_atoms)
        self.max_shells = max_shells
        self.projected_overlap = None
        self.overlap_two_basis = None
        self.n_act_proj_aos = None
        self.shells = None
        self.singular_values = None

    def localize_virtual(self) -> StreamObject:
        """Localise virtual (unoccupied) obitals using concentric localization.

        [1] D. Claudino and N. J. Mayhall, "Simple and Efficient Truncation of Virtual
        Spaces in Embedded Wave Functions via Concentric Localization", Journal of Chemical
        Theory and Computation, vol. 15, no. 11, pp. 6085-6096, Nov. 2019,
        doi: 10.1021/ACS.JCTC.9B00682.

        Returns:
            StreamObject: Fully Localized SCF object.
        """
        logger.debug("Localising virtual orbital spin with concentric localization.")

        logger.debug("Creating projected molecule object.")
        embedded_scf = self.embedded_scf
        logger.debug(f"{embedded_scf.mol.atom=}")
        logger.debug(f"{embedded_scf.mol.charge=}")
        logger.debug(f"{embedded_scf.mol.spin=}")

        projected_mol = gto.mole.Mole()
        projected_mol.atom = embedded_scf.mol.atom
        projected_mol.nelec = embedded_scf.mol.nelec
        projected_mol.basis = embedded_scf.mol.basis  # can be anything
        projected_mol.charge = embedded_scf.mol.charge
        projected_mol.spin = embedded_scf.mol.spin
        projected_mol.build()
        projected_mf = scf.UKS(projected_mol)
        n_act_proj_aos = projected_mol.aoslice_by_atom()[self._n_active_atoms - 1][-1]
        logger.debug(f"{n_act_proj_aos=}")

        self.projected_overlap = projected_mf.get_ovlp(embedded_scf.mol)[
            :n_act_proj_aos, :n_act_proj_aos
        ]
        self.overlap_two_basis = gto.intor_cross(
            "int1e_ovlp_sph", embedded_scf.mol, projected_mol
        )[:n_act_proj_aos, :]
        self.n_act_proj_aos = n_act_proj_aos

        spinless = embedded_scf.mo_coeff.ndim == 2

        if spinless:
            localised_virts = self._localize_virtual_spin(
                embedded_scf.mo_occ, embedded_scf.mo_coeff, embedded_scf.get_fock()
            )
            embedded_scf.mo_coeff = localised_virts[0]
            self.shells = localised_virts[1]
            self.singular_values = localised_virts[2]
        else:
            localised_virts_alpha = self._localize_virtual_spin(
                embedded_scf.mo_occ[0],
                embedded_scf.mo_coeff[0],
                embedded_scf.get_fock()[0],
            )
            localised_virts_beta = self._localize_virtual_spin(
                embedded_scf.mo_occ[1],
                embedded_scf.mo_coeff[1],
                embedded_scf.get_fock()[1],
            )
            embedded_scf.mo_coeff = np.array(
                [localised_virts_alpha[0], localised_virts_beta[0]]
            )

            self.shells = (localised_virts_alpha[1], localised_virts_beta[1])
            self.singular_values = (localised_virts_alpha[2], localised_virts_beta[2])

        logger.debug("Completed Concentric Localization.")
        logger.debug(f"{self.shells=}")
        logger.debug(f"{self.singular_values=}")
        return embedded_scf

    def _localize_virtual_spin(
        self, occ: np.ndarray, mo_coeff: np.ndarray, fock_operator: np.ndarray
    ) -> np.ndarray:
        """Run concentric localization for each spin separately.

        NOTE: These cant be done together as the number of occupied orbitals may be different between the two spins.

        Args:
            occ (np.ndarry): MO occupancy
            mo_coeff (np.ndarry): MO coefficient matrix
            fock_operator (np.ndarray): Fock operator for one spin

        Returns:
            np.ndarray: The update MO coefficient matrix
        """
        logger.debug("Running concentric localiztion for single spin.")

        effective_virt = mo_coeff[:, occ == 0]
        logger.debug(f"N effective virtuals: {effective_virt.shape}")

        left = (
            np.linalg.inv(self.projected_overlap)
            @ self.overlap_two_basis
            @ effective_virt
        )
        _, sigma, right_vectors = np.linalg.svd(
            np.swapaxes(left, -1, -2) @ self.overlap_two_basis @ effective_virt
        )
        logger.debug(f"Singular values: {sigma}")

        # record singular values for analysis
        singular_values = []
        singular_values.append(sigma)

        c_total = mo_coeff[:, occ > 0]

        logger.debug(f"Initial {c_total.shape=} (nocc)")
        logger.debug(f"{self.n_act_proj_aos=}")
        shell_size = np.sum(sigma[: self.n_act_proj_aos] >= 1e-15)
        logger.debug(f"{shell_size=}")

        right_vectors = np.swapaxes(right_vectors, -1, -2)
        v_span, v_ker = np.split(
            right_vectors, [shell_size], axis=-1
        )  # 0 but instability

        logger.debug(f"{v_span.shape=}")
        logger.debug(f"{v_ker.shape=}")

        c_ispan = effective_virt @ v_span
        c_iker = effective_virt @ v_ker

        c_total = np.concatenate((c_total, c_ispan), axis=-1)

        # keep track of the number of orbitals in each shell
        shells = []
        shells.append(c_total.shape[-1])
        logger.debug("Created 0th shell.")

        if v_ker.shape[-1] == 0:
            logger.debug("No kernel for 0th shell, cannot perform CL.")
            logger.debug(
                "This is expected for molecules with majority active MOs occupied."
            )
        elif v_ker.shape[-1] == 1:
            logger.debug(
                "Kernel is 1 for 0th shell, ending CL as cannot perform SVD of vector."
            )
            c_total = np.concatenate((c_total, c_iker), axis=-1)
            shells.append(c_total.shape[-1])
        else:
            # why use the overlap for the first shell and then the fock for the rest?
            for ishell in range(0, self.max_shells):
                logger.debug("Beginning Concentric Localization Iteration")
                logger.debug(f"Shell {ishell}.")

                logger.debug(
                    f"{c_total.shape=}, {fock_operator.shape=}, {c_iker.shape=}"
                )
                _, sigma, right_vectors = np.linalg.svd(
                    np.swapaxes(c_total, -1, -2) @ fock_operator @ c_iker
                )
                logger.debug(f"Singular values: {sigma}")
                singular_values.append(sigma)
                logger.debug(f"{right_vectors.shape=}")

                shell_size = np.sum(sigma[: self.n_act_proj_aos] >= 1e-15)
                logger.debug(f"{shell_size=}")
                if shell_size == 0:
                    logger.debug("Empty shell %s, ending CL.", ishell)
                    c_total = np.concatenate((c_total, c_iker), axis=-1)
                    break

                right_vectors = np.swapaxes(right_vectors, -1, -2)
                v_span, v_ker = np.split(
                    right_vectors, [shell_size], axis=-1
                )  # 0 but instability

                logger.debug(f"{v_span.shape=}")
                logger.debug(f"{v_ker.shape=}")

                # span must be done first as both need to use old c_iker
                c_ispan = c_iker @ v_span
                c_total = np.concatenate((c_total, c_ispan), axis=-1)
                shells.append(c_total.shape[-1])

                if v_ker.shape[-1] > 1:
                    logger.debug("Kernel dimension is greater than 1, continuing CL.")
                    # in-place update
                    c_iker = c_iker @ v_ker
                elif v_ker.shape[-1] == 1:
                    c_iker = c_iker @ v_ker
                    logger.debug(
                        "Kernel is 1, ending CL as cannot perform SVD of vector.",
                    )
                    c_total = np.concatenate((c_total, c_iker), axis=-1)
                    shells.append(c_total.shape[-1])
                    break
                else:
                    logger.debug(
                        "Ending Concentric Localization - All virtual MOs localized."
                    )
                    break

                if ishell >= self.max_shells:
                    logger.debug("Max shells reached, not localizing further virtuals.")
                    c_total = np.concatenate((c_total, c_iker), axis=-1)
                    shells.append(c_total.shape[-1])
                    break

        logger.debug(f"Shell indices: {shells}")

        mo_coeff = c_total

        logger.debug(f"{mo_coeff, shells, singular_values}")

        return mo_coeff, shells, singular_values
