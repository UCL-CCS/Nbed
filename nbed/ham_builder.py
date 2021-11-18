from pyscf.lib import StreamObject
from pyscf import ao2mo
from typing import Optional
from openfermion import InteractionOperator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import get_active_space_integrals
import numpy as np
from cached_property import cached_property

class HamiltonianBuilder:
    """Class to build molecular hamiltonians."""

    def __init__(
        self,
        scf_method: StreamObject,
        constant_e_shift: Optional[float] = 0,
        active_indices: Optional[list] = None,
        occupied_indices: Optional[list] = None,
    ) -> None:
        self.scf_method = scf_method
        self.constant_e_shift = constant_e_shift
        self.active_indices = active_indices
        self.occupied_indices = occupied_indices

    @cached_property
    def _one_body_integrals(self) -> np.ndarray:
        """Get the one electron integrals."""
        c_matrix_active = self.scf_method.mo_coeff

        # one body terms
        one_body_integrals = (
            c_matrix_active.T @ self.scf_method.get_hcore() @ c_matrix_active
        )
        return one_body_integrals
    
    @cached_property
    def _two_body_integrals(self) -> np.ndarray:
        """Get the two electron integrals."""
        c_matrix_active = self.scf_method.mo_coeff
        n_orbs = c_matrix_active.shape[1]

        two_body_compressed = ao2mo.kernel(self.scf_method.mol, c_matrix_active)

        # get electron repulsion integrals
        eri = ao2mo.restore(1, two_body_compressed, n_orbs)  # no permutation symmetry

        # Openfermion uses physicist notation whereas pyscf uses chemists
        two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order="C")
        return two_body_integrals

    def build(self) -> InteractionOperator:
        """Returns second quantized fermionic molecular Hamiltonian.

        constant_e_shift is a constant energy addition... in this code this will be the classical embedding energy
        that corrects for the full system.

        The active_indices and occupied indices are an active space approximation... where occupied and virtual orbitals
        can be frozen. This is different to removing the environment orbitals, as core_constant terms must be added to
        make this approximation.

        Args:
            scf_method (StreamObject): A pyscf self-consistent method.
            constant_e_shift (float): constant energy term to add to Hamiltonian
            active_indices (list): A list of spatial orbital indices indicating which orbitals should be
                                considered active.
            occupied_indices (list):  A list of spatial orbital indices indicating which orbitals should be
                                    considered doubly occupied.

        Returns:
            molecular_hamiltonian (InteractionOperator): fermionic molecular Hamiltonian
        """
        # C_matrix containing orbitals to be considered
        # if there are any environment orbs that have been projected out... these should NOT be present in the
        # scf_method.mo_coeff array (aka columns should be deleted!)
        c_matrix_active = self.scf_method.mo_coeff
        n_orbs = c_matrix_active.shape[1]

        # one body terms
        one_body_integrals = (
            c_matrix_active.T @ self.scf_method.get_hcore() @ c_matrix_active
        )

        two_body_compressed = ao2mo.kernel(self.scf_method.mol, c_matrix_active)

        # get electron repulsion integrals
        eri = ao2mo.restore(1, two_body_compressed, n_orbs)  # no permutation symmetry

        # Openfermion uses physicist notation whereas pyscf uses chemists
        two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order="C")

        if self.occupied_indices or self.active_indices:
            (
                core_constant,
                one_body_integrals,
                two_body_integrals,
            ) = get_active_space_integrals(
                one_body_integrals,
                two_body_integrals,
                occupied_indices=self.occupied_indices,
                active_indices=self.active_indices,
            )
        else:
            core_constant = 0

        one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
            one_body_integrals, two_body_integrals
        )

        molecular_hamiltonian = InteractionOperator(
            (self.constant_e_shift + core_constant),
            one_body_coefficients,
            0.5 * two_body_coefficients,
        )

        return molecular_hamiltonian
