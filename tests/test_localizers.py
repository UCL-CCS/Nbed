"""Tests for localization functions."""

from pathlib import Path
from pyscf import gto, scf
from nbed.localizers.pyscf import PMLocalizer
import scipy as sp
import numpy as np

water_filepath = Path("tests/molecules/water.xyz").absolute()


def test_PMLocalizer_local_basis_transform() -> None:
    """Check change of basis operator (from canonical to localized) is correct"""
    basis = "STO-3G"
    charge = 0
    xc_functional = "b3lyp"
    convergence = 1e-6
    pyscf_print_level = 1
    max_ram_memory = 4_000
    n_active_atoms = 1
    occ_cutoff = 0.95
    virt_cutoff = 0.95
    run_virtual_localization = False

    # define RKS DFT object
    full_mol = gto.Mole(
        atom=str(water_filepath),
        basis=basis,
        charge=charge,
    ).build()
    global_rks = scf.RKS(full_mol)
    global_rks.conv_tol = convergence
    global_rks.xc = xc_functional
    global_rks.max_memory = max_ram_memory
    global_rks.verbose = pyscf_print_level
    global_rks.kernel()

    # run Localizer
    loc_system = PMLocalizer(
        global_rks,
        n_active_atoms=n_active_atoms,
        occ_cutoff=occ_cutoff,
        virt_cutoff=virt_cutoff,
        run_virtual_localization=run_virtual_localization,
    )
    change_basis = loc_system._local_basis_transform

    # check manual
    s_mat = global_rks.get_ovlp()
    s_half = sp.linalg.fractional_matrix_power(s_mat, 0.5)
    s_neg_half = sp.linalg.fractional_matrix_power(s_mat, -0.5)

    # find orthogonal orbitals
    ortho_std = s_half @ global_rks.mo_coeff
    ortho_loc = s_half @ loc_system.c_loc_occ_and_virt

    # Build change of basis operator (maps between orthonormal basis (canonical and localized)
    unitary_ORTHO_std_onto_loc = np.einsum("ik,jk->ij", ortho_std, ortho_loc)

    # move back into non orthogonal basis
    matrix_std_to_loc = s_neg_half @ unitary_ORTHO_std_onto_loc @ s_half

    # Check U_ORTHO_std_onto_loc*C_ortho_loc ==  C_ortho_STD
    assert np.allclose(unitary_ORTHO_std_onto_loc @ ortho_loc, ortho_std)

    # Change of basis (U_ORTHO_std_onto_loc) is not Unitary
    assert np.allclose(
        unitary_ORTHO_std_onto_loc.conj().T @ unitary_ORTHO_std_onto_loc,
        np.eye(unitary_ORTHO_std_onto_loc.shape[0]),
    )

    # Check change of basis incorrect... U_std*C_std !=  C_loc_occ_and_virt
    assert np.allclose(
        change_basis @ loc_system.c_loc_occ_and_virt, global_rks.mo_coeff
    )

    return None


if __name__ == "__main__":
    pass
