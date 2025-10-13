"""Tests for localization functions."""

from pubchempy import request
import numpy as np
import pytest
from pyscf import gto, scf, dft

from nbed.localizers import occupied
from nbed.localizers.occupied import OccupiedLocalizer, PMLocalizer, SPADELocalizer, BOYSLocalizer, IBOLocalizer
from nbed.localizers.occupied.pyscf import PySCFLocalizer
from nbed.localizers.virtual import ConcentricLocalizer
from nbed.localizers.ace import ACELocalizer
from nbed.localizers import LocalizedSystem

import logging
logger = logging.getLogger(__name__)

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

# @pytest.fixture
# def molecule_spin(water_filepath) -> gto.Mole:
#     return gto.Mole(
#         atom=str(water_filepath),
#         basis="6-31g",
#         charge=0,
#         spin=2,
#     ).build()

# @pytest.fixture
# def molecule_spin_charge(water_filepath) -> gto.Mole:
#     return gto.Mole(
#         atom=str(water_filepath),
#         basis="6-31g",
#         charge=1,
#         spin=1,
#     ).build()


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

# @pytest.fixture
# def global_uks_spin(molecule_spin) -> scf.UKS:
#     global_uks = scf.UKS(molecule_spin)
#     global_uks.conv_tol = convergence
#     global_uks.xc = xc_functional
#     global_uks.max_memory = max_ram_memory
#     global_uks.verbose = pyscf_print_level
#     global_uks.kernel()
#     return global_uks

# @pytest.fixture
# def global_uks_spin_charge(molecule_spin_charge) -> scf.UKS:
#     global_uks = scf.UKS(molecule_spin_charge)
#     global_uks.conv_tol = convergence
#     global_uks.xc = xc_functional
#     global_uks.max_memory = max_ram_memory
#     global_uks.verbose = pyscf_print_level
#     global_uks.kernel()
#     return global_uks

# @pytest.fixture
# def global_roks(molecule) -> scf.ROKS:
#     global_roks = scf.ROKS(molecule)
#     global_roks.conv_tol = convergence
#     global_roks.xc = xc_functional
#     global_roks.max_memory = max_ram_memory
#     global_roks.verbose = pyscf_print_level
#     global_roks.kernel()
#     return global_roks

# @pytest.fixture
# def global_roks_spin_charge(molecule_spin_charge) -> scf.ROKS:
#     global_roks = scf.ROKS(molecule_spin_charge)
#     global_roks.conv_tol = convergence
#     global_roks.xc = xc_functional
#     global_roks.max_memory = max_ram_memory
#     global_roks.verbose = pyscf_print_level
#     global_roks.kernel()
#     return global_roks

def test_base_localizer(global_rks) -> None:
    """Check the base class can be instantiated."""
    with pytest.raises(TypeError) as excinfo:
        OccupiedLocalizer(global_rks, n_active_atoms=n_active_atoms).localize()
    assert "localize_spin" in str(excinfo.value)


def test_PM_arguments(global_rks) -> None:
    """Check the internal test of values."""
    with pytest.raises(ValueError):
        PMLocalizer(
            global_rks,
            n_active_atoms=n_active_atoms,
            occ_cutoff=1.1,
            virt_cutoff=virt_cutoff,
        ).localize()

    with pytest.raises(ValueError):
        PMLocalizer(
            global_rks,
            n_active_atoms=n_active_atoms,
            occ_cutoff=occ_cutoff,
            virt_cutoff=1.1,
        ).localize()

    with pytest.raises(ValueError):
        PMLocalizer(
            global_rks,
            n_active_atoms=n_active_atoms,
            occ_cutoff=-0.1,
            virt_cutoff=virt_cutoff,
        ).localize()

    with pytest.raises(ValueError):
        PMLocalizer(
            global_rks,
            n_active_atoms=n_active_atoms,
            occ_cutoff=occ_cutoff,
            virt_cutoff=-0.1,
        ).localize()


def check_partition(
    localized_system
) -> None:  # Needs clarification
    """Check that output values make sense.
    - Same number of active and environment orbitals in alpha and beta
    - Total DM is sum of active and environment DM
    - Total number of electrons conserved

    """
    if localized_system.active_occ_inds.ndim == 2:
        assert (
            localized_system.active_occ_inds[0].shape
            == localized_system.active_occ_inds[1].shape
        )
        assert (
            localized_system.enviro_occ_inds[0].shape
            == localized_system.enviro_occ_inds[1].shape
        )
        assert (
            localized_system.dm_active[0].shape ==
            localized_system.dm_active[1].shape
        )
        assert (
            localized_system.dm_enviro[0].shape ==
            localized_system.dm_enviro[1].shape
        )

    # checking denisty matrix parition sums to total
    logger.debug("Checking density matrix partition.")
    dm_localised_full_system = (
        localized_system.c_loc_occ
        @ localized_system.c_loc_occ.conj().swapaxes(-1, -2)
    )
    dm_sum = localized_system.dm_active + localized_system.dm_enviro
    match localized_system.c_loc_occ.ndim:
        case 2:
            # In a restricted system we have two electrons per orbital
            assert np.allclose(2 * dm_localised_full_system, dm_sum)
        case 3:
            # both need to be correct
            assert np.allclose(dm_localised_full_system, dm_sum)

def check_charge_conservation(localized_system, global_scf):
    # check number of electrons is still the same after orbitals have been localized (change of basis)
    logger.debug("Checking electron number conserverd.")
    s_ovlp = global_scf.get_ovlp()

    match localized_system.dm_active.ndim:
        case 2:
            n_active_electrons = np.trace(localized_system.dm_active @ s_ovlp)
            n_enviro_electrons = np.trace(localized_system.dm_enviro @ s_ovlp)

        case 3:
            n_active_electrons = np.trace(localized_system.dm_active[0] @ s_ovlp)
            n_enviro_electrons = np.trace(localized_system.dm_enviro[0] @ s_ovlp)
            n_active_electrons += np.trace(localized_system.dm_active[1] @ s_ovlp)
            n_enviro_electrons += np.trace(localized_system.dm_enviro[1] @ s_ovlp)

    n_all_electrons = global_scf.mol.nelectron
    assert np.isclose(
        (n_active_electrons + n_enviro_electrons), n_all_electrons
    )


@pytest.mark.parametrize("scf", ["global_rks", "global_uks"])#,"global_uks_spin", "global_uks_spin_charge", "global_roks", "global_roks_spin_charge"])
def test_PM_check_values(scf, request) -> None:
    """Check the internal test of values."""
    scf = request.getfixturevalue(scf)

    localizer = PMLocalizer(
        scf,
        n_active_atoms=n_active_atoms,
        occ_cutoff=occ_cutoff,
        virt_cutoff=virt_cutoff,)
    ls = localizer.localize()
    check_partition(ls)
    check_charge_conservation(ls, localizer._global_scf)

@pytest.mark.parametrize("scf", ["global_rks", "global_uks"])#,"global_uks_spin", "global_uks_spin_charge", "global_roks", "global_roks_spin_charge"])
def test_SPADE_check_values(scf, request) -> None:
    """Check the internal test of values."""
    scf = request.getfixturevalue(scf)
    localizer = SPADELocalizer(
        scf,
        n_active_atoms=n_active_atoms,
    )
    ls = localizer.localize()
    assert isinstance(ls, LocalizedSystem)
    check_partition(ls)
    check_charge_conservation(ls, localizer._global_scf)


def test_PM_mo_indices(global_rks, global_uks) -> None:
    restricted_loc_system = PMLocalizer(
        global_rks,
        n_active_atoms=n_active_atoms,
        occ_cutoff=occ_cutoff,
        virt_cutoff=virt_cutoff,
    ).localize()

    unrestricted_loc_system = PMLocalizer(
        global_uks,
        n_active_atoms=n_active_atoms,
        occ_cutoff=occ_cutoff,
        virt_cutoff=virt_cutoff,
    ).localize()
    assert np.all(
        restricted_loc_system.active_occ_inds == unrestricted_loc_system.active_occ_inds[0]
    )
    assert np.all(
        restricted_loc_system.enviro_occ_inds == unrestricted_loc_system.enviro_occ_inds[1]
    )
    assert np.all(
        unrestricted_loc_system.active_occ_inds[0]
        == unrestricted_loc_system.active_occ_inds[1]
    )
    assert np.all(
        unrestricted_loc_system.enviro_occ_inds[0]
        == unrestricted_loc_system.enviro_occ_inds[1]
    )


def test_SPADE_mo_indices(global_rks, global_uks) -> None:
    restricted_loc_system = SPADELocalizer(
        global_rks,
        n_active_atoms=n_active_atoms,
    ).localize()

    unrestricted_loc_system = SPADELocalizer(
        global_uks,
        n_active_atoms=n_active_atoms,
    ).localize()
    assert np.all(
        restricted_loc_system.active_occ_inds == unrestricted_loc_system.active_occ_inds[0]
    )
    assert np.all(
        restricted_loc_system.enviro_occ_inds == unrestricted_loc_system.enviro_occ_inds[0]
    )
    assert np.all(
        unrestricted_loc_system.active_occ_inds[0]
        == unrestricted_loc_system.active_occ_inds[1]
    )
    assert np.all(
        unrestricted_loc_system.enviro_occ_inds[0]
        == unrestricted_loc_system.enviro_occ_inds[1]
    )


def test_PMLocalizer_local_basis_transform(global_rks) -> None:
    """Check change of basis operator (from canonical to localized) is correct"""
    # run Localizer
    loc_system = PMLocalizer(
        global_rks,
        n_active_atoms=n_active_atoms,
        occ_cutoff=occ_cutoff,
        virt_cutoff=virt_cutoff,
    ).localize()
    dm_full_std = global_rks.make_rdm1()
    dm_active_sys = loc_system.dm_active
    dm_enviro_sys = loc_system.dm_enviro
    # y_active + y_enviro = y_total
    assert np.all(dm_full_std.shape==dm_active_sys.shape)
    assert np.all(dm_full_std.shape==dm_enviro_sys.shape)
    # assert np.allclose(dm_full_std, dm_active_sys + dm_enviro_sys)

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
    ).localize()

    unrestricted = SPADELocalizer(
        global_uks,
        n_active_atoms=n_active_atoms,
    ).localize()

    # assert loc_system.active_occ_inds
    assert restricted.active_occ_inds.ndim == 1
    assert np.all(unrestricted.active_occ_inds[0] == unrestricted.active_occ_inds[1])
    assert np.all(restricted.active_occ_inds == unrestricted.active_occ_inds[0])


def test_cl_shell_numbers(global_rks, global_uks) -> None:
    restricted_occ = SPADELocalizer(
        global_rks,
        n_active_atoms=n_active_atoms,
    )
    restricted_occ.localize()
    restricted_virt = ConcentricLocalizer(
        restricted_occ._global_scf, n_active_atoms=n_active_atoms
    )
    restricted_virt.localize_virtual()

    unrestricted_occ = SPADELocalizer(
        global_uks,
        n_active_atoms=n_active_atoms,
    )
    unrestricted_occ.localize()
    unrestricted_virt = ConcentricLocalizer(
        unrestricted_occ._global_scf, n_active_atoms=n_active_atoms
    )
    unrestricted_virt.localize_virtual()

    assert np.all(restricted_virt.shells == [12, 13])
    assert np.all(
        restricted_virt.shells
        == unrestricted_virt.shells[0])
    assert np.all( restricted_virt.shells
        == unrestricted_virt.shells[1]
    )

def test_pao_localizer(global_rks, global_uks) -> None:
    pass

def test_ace_localizer(global_rks, global_uks) -> None:
    restricted = ACELocalizer(
        global_scf_list=[global_rks] * 3, n_active_atoms=n_active_atoms
    ).localize_path()
    unrestricted = ACELocalizer(
        global_scf_list=[global_uks] * 3, n_active_atoms=n_active_atoms
    ).localize_path()

    restricted_spade = SPADELocalizer(
        global_rks,
        n_active_atoms=n_active_atoms,
        n_mo_overwrite=restricted,
    )
    restricted_spade.localize()

    unrestricted_spade = SPADELocalizer(
        global_uks,
        n_active_atoms=n_active_atoms,
        n_mo_overwrite=unrestricted,
    )
    unrestricted_spade.localize()
    print(restricted_spade.enviro_selection_condition)
    print(unrestricted_spade.enviro_selection_condition)
    assert restricted == unrestricted == (3, 3)
    assert restricted[0] == restricted[1]
    assert unrestricted[0] == unrestricted[1]
    assert np.all(
        restricted[0] - 1
        == np.argmax(
            restricted_spade.enviro_selection_condition[0][:-1]
            - restricted_spade.enviro_selection_condition[0][1:]
        )
    )
    assert np.all(
        unrestricted[0] - 1
        == np.argmax(
            unrestricted_spade.enviro_selection_condition[0][:-1]
            - unrestricted_spade.enviro_selection_condition[0][1:]
        )
    )

def test_pyscf_subtypes():
    assert issubclass(PMLocalizer, PySCFLocalizer)
    assert issubclass(BOYSLocalizer, PySCFLocalizer)
    assert issubclass(IBOLocalizer, PySCFLocalizer)

@pytest.mark.parametrize("localizer", [PMLocalizer, SPADELocalizer])
@pytest.mark.parametrize("scf",["global_rks", "global_uks"])
def test_localized_system(localizer, scf, request):
    scf = request.getfixturevalue(scf)

    match localizer:
        case occupied.PMLocalizer | occupied.BOYSLocalizer | occupied.IBOLocalizer:
            ls = localizer(scf, n_active_atoms, occ_cutoff, virt_cutoff).localize()
        case occupied.SPADELocalizer:
            ls = localizer(scf, n_active_atoms).localize()
        case _:
            raise ValueError("Invalid localizer.")

    assert isinstance(ls, LocalizedSystem)
    if isinstance(scf, dft.rks.RKS):
        assert ls.active_occ_inds.ndim == 1
        assert ls.enviro_occ_inds.ndim == 1
        assert ls.active_occ_inds.shape == ls.enviro_occ_inds.shape

        assert ls.c_loc_occ.ndim == 2
        assert ls.c_loc_occ.shape[0] == scf.mo_coeff.shape[0]
        assert ls.c_loc_occ.shape[1] <= scf.mo_coeff.shape[1]

    elif isinstance(scf, dft.uks.UKS):
        assert ls.active_occ_inds.ndim == 2
        assert ls.enviro_occ_inds.ndim == 2
        assert ls.active_occ_inds.shape == ls.enviro_occ_inds.shape

        assert ls.c_loc_occ.ndim == 3
        assert ls.c_loc_occ.shape[0] == 2
        assert ls.c_loc_occ.shape[0] == scf.mo_coeff.shape[0]
        assert ls.c_loc_occ.shape[1] == scf.mo_coeff.shape[1]
        assert ls.c_loc_occ.shape[2] <= scf.mo_coeff.shape[2]



if __name__ == "__main__":
    pass
