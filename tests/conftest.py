"""Shared fixtures for tests."""

from pathlib import Path

import pytest
from pyscf import gto, scf,dft
from pyscf.lib import StreamObject

from nbed.driver import NbedDriver
from nbed.config import NbedConfig


@pytest.fixture(scope="module")
def config_file() -> Path:
    return Path("tests/test_config.json").absolute()


@pytest.fixture(scope="module")
def water_filepath() -> Path:
    return Path("tests/molecules/water.xyz").absolute()


@pytest.fixture(scope="module")
def pfoa_filepath() -> Path:
    return Path("tests/molecules/pfoa.xyz").absolute()


@pytest.fixture(scope="module")
def water_molecule(water_filepath) -> gto.Mole:
    mol_args = {
        "atom": str(water_filepath),
        "n_active_atoms": 1,
        "basis": "STO-3G",
        "unit": "angstrom",
    }
    return gto.Mole(**mol_args, charge=0, spin=0).build()


@pytest.fixture(scope="module")
def water_rhf(water_molecule) -> StreamObject:
    rhf = scf.rhf.RHF(water_molecule)
    rhf.kernel()
    return rhf

@pytest.fixture(scope="module")
def water_uhf(water_molecule) -> StreamObject:
    uhf = scf.uhf.UHF(water_molecule)
    uhf.kernel()
    return uhf

@pytest.fixture(scope="module")
def water_rks(water_molecule) -> StreamObject:
    rks = dft.RKS(water_molecule)
    rks.kernel()
    return rks

@pytest.fixture(scope="module")
def water_uks(water_molecule) -> StreamObject:
    uks = dft.UKS(water_molecule)
    uks.kernel()
    return uks


@pytest.fixture(scope="module")
def nbed_args(water_filepath) -> dict:
    args = {
        "geometry": water_filepath,
        "n_active_atoms": 2,
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "convergence": 1e-06,
        "charge": 0,
        "spin": 0,
        "symmetry": False,
        "mu_level_shift": 1000000.0,
        "run_ccsd_emb": False,
        "run_fci_emb": False,
        "run_virtual_localization": True,
        "n_mo_overwrite": (None, None),
        "run_dft_in_dft": False,
        "max_ram_memory": 4000,
        "pyscf_print_level": 1,
        "occupied_threshold": 0.95,
        "virtual_threshold": 0.95,
        "max_shells": 4,
        "init_huzinaga_rhf_with_mu": False,
        "max_hf_cycles": 50,
        "max_dft_cycles": 50,
        "force_unrestricted": False,
        "mm_coords": None,
        "mm_charges": None,
        "mm_radii": None,
    }
    return args


@pytest.fixture(scope="module")
def nbed_config(nbed_args) -> NbedConfig:
    return NbedConfig(**nbed_args)


@pytest.fixture(scope="module")
def spinless_driver():
    water_xyz_raw = (
        "3\n \nH\t0.2774\t0.8929\t0.2544\nO\t0\t0\t0\nH\t0.6068\t-0.2383\t-0.7169"
    )
    args = {
        "geometry": water_xyz_raw,
        "n_active_atoms": 2,
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "convergence": 1e-6,
        "run_ccsd_emb": False,
        "run_fci_emb": False,
    }

    config = NbedConfig(**args)
    driver = NbedDriver(config)
    driver.embed()
    return driver


@pytest.fixture(scope="module")
def unrestricted_driver():
    water_xyz_raw = (
        "3\n \nH\t0.2774\t0.8929\t0.2544\nO\t0\t0\t0\nH\t0.6068\t-0.2383\t-0.7169"
    )
    args = {
        "geometry": water_xyz_raw,
        "n_active_atoms": 2,
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "convergence": 1e-6,
        "run_ccsd_emb": False,
        "run_fci_emb": False,
        "force_unrestricted": True,
    }
    config = NbedConfig(**args)
    driver = NbedDriver(config)
    driver.embed()
    return driver
