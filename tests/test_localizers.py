"""Tests for localization functions."""

import numpy as np
import pytest
from pyscf import gto, scf

from nbed.localizers.base import Localizer
from nbed.localizers.pyscf import PMLocalizer
from nbed.localizers.spade import SPADELocalizer

xc_functional = "b3lyp"
convergence = 1e-6
pyscf_print_level = 1
max_ram_memory = 4_000
n_active_atoms = 1
occ_cutoff = 0.95
virt_cutoff = 0.95
run_virtual_localization = False


@pytest.fixture
def molecule(water_filepath) -> gto.Mole:
    return gto.Mole(
        atom=str(water_filepath),
        basis="6-31g",
        charge=0,
    ).build()


@pytest.fixture
def global_rks(molecule) -> scf.RKS:
    global_rks = scf.RKS(molecule)
    global_rks.conv_tol = convergence
    global_rks.xc = xc_functional
    global_rks.max_memory = max_ram_memory
    global_rks.verbose = pyscf_print_level
    global_rks.kernel()
    return global_rks


@pytest.fixture
def global_uks(molecule) -> scf.UKS:
    global_uks = scf.UKS(molecule)
    global_uks.conv_tol = convergence
    global_uks.xc = xc_functional
    global_uks.max_memory = max_ram_memory
    global_uks.verbose = pyscf_print_level
    global_uks.kernel()
    return global_uks


def test_base_localizer(global_rks) -> None:
    """Check the base class can be instantiated."""
    with pytest.raises(TypeError) as excinfo:
        Localizer(global_rks, n_active_atoms=n_active_atoms)

    assert "_localize_spin" in str(excinfo.value)
    assert "localize_virtual" in str(excinfo.value)


def test_PM_arguments(global_rks) -> None:
    """Check the internal test of values."""
    with pytest.raises(ValueError):
        PMLocalizer(
            global_rks,
            n_active_atoms=n_active_atoms,
            occ_cutoff=1.1,
            virt_cutoff=virt_cutoff,
        )

    with pytest.raises(ValueError):
        PMLocalizer(
            global_rks,
            n_active_atoms=n_active_atoms,
            occ_cutoff=occ_cutoff,
            virt_cutoff=1.1,
        )

    with pytest.raises(ValueError):
        PMLocalizer(
            global_rks,
            n_active_atoms=n_active_atoms,
            occ_cutoff=-0.1,
            virt_cutoff=virt_cutoff,
        )

    with pytest.raises(ValueError):
        PMLocalizer(
            global_rks,
            n_active_atoms=n_active_atoms,
            occ_cutoff=occ_cutoff,
            virt_cutoff=-0.1,
        )


def test_PM_check_values(global_rks, global_uks) -> None:
    """Check the internal test of values."""
    for ks in [global_rks, global_uks]:
        PMLocalizer(
            ks,
            n_active_atoms=n_active_atoms,
            occ_cutoff=occ_cutoff,
            virt_cutoff=virt_cutoff,
        ).run(check_values=True)


def test_SPADE_check_values(global_rks, global_uks) -> None:
    """Check the internal test of values."""
    for ks in [global_rks, global_uks]:
        SPADELocalizer(
            ks,
            n_active_atoms=n_active_atoms,
        ).run(check_values=True)


def test_PM_mo_indices(global_rks, global_uks) -> None:
    restricted_loc_system = PMLocalizer(
        global_rks,
        n_active_atoms=n_active_atoms,
        occ_cutoff=occ_cutoff,
        virt_cutoff=virt_cutoff,
    )
    assert restricted_loc_system.beta_active_MO_inds is None
    assert restricted_loc_system.beta_enviro_MO_inds is None
    assert restricted_loc_system.beta_c_active is None
    assert restricted_loc_system.beta_c_enviro is None
    assert restricted_loc_system._beta_c_loc_occ is None

    unrestricted_loc_system = PMLocalizer(
        global_uks,
        n_active_atoms=n_active_atoms,
        occ_cutoff=occ_cutoff,
        virt_cutoff=virt_cutoff,
    )
    assert np.all(
        restricted_loc_system.active_MO_inds == unrestricted_loc_system.active_MO_inds
    )
    assert np.all(
        restricted_loc_system.enviro_MO_inds == unrestricted_loc_system.enviro_MO_inds
    )
    assert np.all(
        unrestricted_loc_system.active_MO_inds
        == unrestricted_loc_system.beta_active_MO_inds
    )
    assert np.all(
        unrestricted_loc_system.enviro_MO_inds
        == unrestricted_loc_system.beta_enviro_MO_inds
    )


def test_SPADE_mo_indices(global_rks, global_uks) -> None:
    restricted_loc_system = SPADELocalizer(
        global_rks,
        n_active_atoms=n_active_atoms,
    )
    assert restricted_loc_system.beta_active_MO_inds is None
    assert restricted_loc_system.beta_enviro_MO_inds is None
    assert restricted_loc_system.beta_c_active is None
    assert restricted_loc_system.beta_c_enviro is None
    assert restricted_loc_system._beta_c_loc_occ is None

    unrestricted_loc_system = SPADELocalizer(
        global_uks,
        n_active_atoms=n_active_atoms,
    )
    assert np.all(
        restricted_loc_system.active_MO_inds == unrestricted_loc_system.active_MO_inds
    )
    assert np.all(
        restricted_loc_system.enviro_MO_inds == unrestricted_loc_system.enviro_MO_inds
    )
    assert np.all(
        unrestricted_loc_system.active_MO_inds
        == unrestricted_loc_system.beta_active_MO_inds
    )
    assert np.all(
        unrestricted_loc_system.enviro_MO_inds
        == unrestricted_loc_system.beta_enviro_MO_inds
    )


def test_PMLocalizer_local_basis_transform(global_rks) -> None:
    """Check change of basis operator (from canonical to localized) is correct"""
    # run Localizer
    loc_system = PMLocalizer(
        global_rks,
        n_active_atoms=n_active_atoms,
        occ_cutoff=occ_cutoff,
        virt_cutoff=virt_cutoff,
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


def test_spade_spins_match(global_rks, global_uks) -> None:
    """Check that localization of restricted and unrestricted match."""
    # define RKS DFT object

    restricted = SPADELocalizer(
        global_rks,
        n_active_atoms=n_active_atoms,
    )

    unrestricted = SPADELocalizer(
        global_uks,
        n_active_atoms=n_active_atoms,
    )

    # assert loc_system.active_MO_inds
    assert restricted.beta_active_MO_inds is None
    assert np.all(unrestricted.active_MO_inds == unrestricted.beta_active_MO_inds)
    assert np.all(restricted.active_MO_inds == unrestricted.active_MO_inds)


def test_cl_shell_numbers(global_rks, global_uks) -> None:
    restricted = SPADELocalizer(
        global_rks,
        n_active_atoms=n_active_atoms,
    )
    restricted.localize_virtual(restricted._global_scf)

    unrestricted = SPADELocalizer(
        global_uks,
        n_active_atoms=n_active_atoms,
    )
    unrestricted.localize_virtual(unrestricted._global_scf)

    assert restricted.shells == [12, 13]
    assert restricted.shells == unrestricted.shells

def test_cl_reduces_orbitals(pfoa_filepath, driver_args):
    driver_args["geometry"] = str(pfoa_filepath)
    driver_args["n_active_atoms"] = 2

    from nbed.driver import NbedDriver

    novirt = NbedDriver(**driver_args, run_virtual_localization=False)
    novirt_mos = novirt.embedded_scf.mo_coeff

    virt = NbedDriver(**driver_args, run_virtual_localization=True, max_shells=2)
    virt_mos = virt.embedded_scf.mo_coeff

    assert novirt_mos.shape[-1] < virt_mos.shape[-1]

if __name__ == "__main__":
    pass
