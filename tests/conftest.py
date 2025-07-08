"""Shared fixtures for tests."""

from pathlib import Path

import pytest
from pyscf import gto, scf
from pyscf.lib import StreamObject

from nbed.driver import NbedDriver


@pytest.fixture(scope="module")
def water_filepath() -> Path:
    return Path("tests/molecules/water.xyz").absolute()

@pytest.fixture(scope="module")
def pfoa_filepath() -> Path:
    return Path("tests/molecules/pfoa.xyz").absolute()

@pytest.fixture(scope="module")
def water_mol(water_filepath) -> gto.Mole:
    mol_args = {
        "atom": str(water_filepath),
        "n_active_atoms": 1,
        "basis": "STO-3G",
        "unit": "angstrom",
    }
    return gto.Mole(**mol_args, charge=0, spin=0).build()


@pytest.fixture(scope="module")
def water_rhf(water_molecule) -> StreamObject:
    rhf = scf.RHF(water_molecule)
    rhf.kernel()
    return rhf


@pytest.fixture(scope="module")
def driver_args(water_filepath) -> dict:
    return {
        'geometry': water_filepath,
        'n_active_atoms': 2,
        'basis': 'STO-3G',
        'xc_functional': 'b3lyp',
        'projector': 'mu',
        'localization': 'spade',
        'convergence': 1e-06,
        'charge': 0,
        'spin': 0,
        'symmetry': False,
        'mu_level_shift': 1000000.0,
        'run_ccsd_emb': False,
        'run_fci_emb': False,
        'run_virtual_localization': True,
        'n_mo_overwrite': (None, None),
        'run_dft_in_dft': False,
        'max_ram_memory': 4000,
        'pyscf_print_level': 1,
        'occupied_threshold': 0.95,
        'virtual_threshold': 0.95,
        'max_shells': 4,
        'init_huzinaga_rhf_with_mu': False,
        'max_hf_cycles': 50,
        'max_dft_cycles': 50,
        'force_unrestricted': False,
        'run_qmmm': False,
        'mm_coords': None,
        'mm_charges': None,
        'mm_radii': None
        }

@pytest.fixture(scope="module")
def restricted_driver():
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

    driver = NbedDriver(**args)
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

    driver = NbedDriver(**args)
    return driver
