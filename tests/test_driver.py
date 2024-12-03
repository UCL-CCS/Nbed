"""
File to contain tests of the driver.py script.
"""

import logging
from pathlib import Path

import pytest
from numpy import isclose
from pyscf.lib.misc import StreamObject

from nbed.driver import NbedDriver
from nbed.exceptions import NbedConfigError

logger = logging.getLogger(__name__)


water_filepath = Path("tests/molecules/water.xyz").absolute()


def test_incorrect_geometry_path() -> None:
    """test to make sure that FileNotFoundError is thrown if invalid path to xyz geometry file is given"""

    molecule = "THIS/IS/NOT/AN/XYZ/FILE"

    args = {
        "geometry": molecule,
        "n_active_atoms": 1,
        "basis": "STO-3G",
        "xc_functional": "b3lyp5",
        "projector": "mu",
        "localization": "spade",
        "convergence": 1e-6,
        "run_ccsd_emb": True,
        "run_fci_emb": True,
    }

    with pytest.raises(RuntimeError, match="Unsupported atom symbol .*"):
        # match will match with any printed error message
        NbedDriver(**args)


def test_driver_standard_xyz_file_input() -> None:
    """test to check driver works... path to xyz file given"""

    args = {
        "geometry": str(water_filepath),
        "n_active_atoms": 1,
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "convergence": 1e-6,
        "run_ccsd_emb": False,
        "run_fci_emb": False,
    }

    driver = NbedDriver(**args)
    assert isinstance(driver.embedded_scf, StreamObject)
    assert isclose(driver.classical_energy, -14.229079481431608)


def test_driver_standard_xyz_string_input() -> None:
    """test to check driver works... raw xyz string given"""
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
    assert isinstance(driver.embedded_scf, StreamObject)
    assert isclose(driver.classical_energy, -3.5867934952241356)


def test_n_active_atoms_validation() -> None:
    """test to check driver works... path to xyz file given"""

    args = {
        "geometry": str(water_filepath),
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "convergence": 1e-6,
        "run_ccsd_emb": True,
        "run_fci_emb": True,
    }
    error_msg = "Invalid number of active atoms. Choose a number from 1 to 2."

    with pytest.raises(NbedConfigError, match=error_msg):
        NbedDriver(n_active_atoms=0, **args)

    with pytest.raises(NbedConfigError, match=error_msg):
        NbedDriver(n_active_atoms=3, **args)


def test_subsystem_dft() -> None:
    """Check thatcmponenets match total dft energy."""
    args = {
        "geometry": str(water_filepath),
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

    energy_DFT_components = (
        driver.e_act
        + driver.e_env
        + driver.two_e_cross
        + driver._global_ks.energy_nuc()
    )

    assert isclose(energy_DFT_components, driver._global_ks.e_tot)


def test_subsystem_dft_spin_consistency() -> None:
    """Check restricted & unrestricted components match."""
    args = {
        "geometry": str(water_filepath),
        "n_active_atoms": 1,
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "convergence": 1e-6,
        "run_ccsd_emb": True,
        "run_fci_emb": True,
    }

    restricted_driver = NbedDriver(
        geometry=args["geometry"],
        n_active_atoms=args["n_active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        projector=args["projector"],
        localization=args["localization"],
        convergence=args["convergence"],
        run_ccsd_emb=args["run_ccsd_emb"],
        run_fci_emb=args["run_fci_emb"],
    )

    unrestricted_driver = NbedDriver(
        geometry=args["geometry"],
        n_active_atoms=args["n_active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        projector=args["projector"],
        localization=args["localization"],
        convergence=args["convergence"],
        run_ccsd_emb=args["run_ccsd_emb"],
        run_fci_emb=args["run_fci_emb"],
        force_unrestricted=True,
    )
    # Could be problems with caching here

    assert isclose(restricted_driver.e_act, unrestricted_driver.e_act)
    assert isclose(restricted_driver.e_env, unrestricted_driver.e_env)
    assert isclose(restricted_driver.two_e_cross, unrestricted_driver.two_e_cross)
    assert isclose(
        restricted_driver.classical_energy, unrestricted_driver.classical_energy
    )


if __name__ == "__main__":
    pass
