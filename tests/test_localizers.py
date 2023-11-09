"""Tests for localization functions."""

from pathlib import Path

import numpy as np
import scipy as sp
from pyscf import gto, scf

from nbed.localizers.pyscf import PMLocalizer
from nbed.localizers.spade import SPADELocalizer

water_filepath = Path("tests/molecules/water.xyz").absolute()
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

global_uks = scf.UKS(full_mol)
global_uks.conv_tol = convergence
global_uks.xc = xc_functional
global_uks.max_memory = max_ram_memory
global_uks.verbose = pyscf_print_level
global_uks.kernel()

def test_PMLocalizer_local_basis_transform() -> None:
    """Check change of basis operator (from canonical to localized) is correct"""

    # run Localizer
    loc_system = PMLocalizer(
        global_rks,
        n_active_atoms=n_active_atoms,
        occ_cutoff=occ_cutoff,
        virt_cutoff=virt_cutoff,
        run_virtual_localization=run_virtual_localization,
    )
    dm_full_std = global_rks.make_rdm1()
    dm_active_sys = loc_system.dm_active
    dm_enviro_sys = loc_system.dm_enviro
    # y_active + y_enviro = y_total
    assert np.allclose(dm_full_std, dm_active_sys + dm_enviro_sys)

    n_all_electrons = global_rks.mol.nelectron
    s_ovlp = global_rks.get_ovlp()
    n_active_electrons = np.trace(dm_active_sys @ s_ovlp)
    n_enviro_electrons = np.trace(dm_enviro_sys @ s_ovlp)

    # check number of electrons is still the same after orbitals have been localized (change of basis)
    assert np.isclose(n_all_electrons, n_active_electrons + n_enviro_electrons)


def test_spade_spins_match() -> None:
    """Check that localization of restricted and unrestricted match."""
    # define RKS DFT object

    restricted = SPADELocalizer(
        global_rks,
        n_active_atoms=n_active_atoms,
        occ_cutoff=occ_cutoff,
        virt_cutoff=virt_cutoff,
        run_virtual_localization=run_virtual_localization,
    )

    unrestricted = SPADELocalizer(
        global_uks,
        n_active_atoms=n_active_atoms,
        occ_cutoff=occ_cutoff,
        virt_cutoff=virt_cutoff,
        run_virtual_localization=run_virtual_localization,
    )

    # assert loc_system.active_MO_inds
    assert restricted.beta_active_MO_inds is None
    assert np.all(unrestricted.active_MO_inds == unrestricted.beta_active_MO_inds)
    assert np.all(restricted.active_MO_inds == unrestricted.active_MO_inds)

def test_cl_shells() -> None:
    pass

def test_cl_open_shell() -> None:
    pass

if __name__ == "__main__":
    pass
