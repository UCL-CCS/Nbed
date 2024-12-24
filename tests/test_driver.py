"""
File to contain tests of the driver.py script.
"""

import logging
from pathlib import Path

import pytest
from numpy import isclose
from pyscf.lib.misc import StreamObject
import numpy as np

from nbed.driver import NbedDriver
from nbed.exceptions import NbedConfigError

logger = logging.getLogger(__name__)

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


def test_driver_standard_xyz_file_input(water_filepath) -> None:
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
    assert np.allclose(driver.embedded_scf.mo_coeff, 
                       np.array([[-3.88142342e-03,  3.02684557e-01,  4.52415720e-01,
                            -1.27605620e-05, -7.61743817e-01,  8.49826960e-01],
                        [ 9.95680230e-01, -2.14527741e-01,  1.05457231e-01,
                            -2.70846025e-06, -1.29395721e-01,  4.54348546e-03],
                        [ 2.14382088e-02,  8.09145086e-01, -5.27807618e-01,
                            1.53185771e-05,  8.58088321e-01, -2.80005533e-02],
                        [-3.37254332e-03, -1.14106506e-01,  4.36409575e-01,
                            6.36016563e-01,  5.60822186e-01,  1.95992040e-01],
                        [ 4.16624471e-03,  1.97283723e-01,  5.76343342e-01,
                            -3.87702724e-01,  3.90462503e-01, -7.39775077e-01],
                        [ 5.63571421e-03,  2.23407976e-01, -8.10867198e-02,
                            6.67210258e-01, -3.07731005e-01, -6.16688715e-01],
                        [-1.49279774e-02, -1.68597526e-01,  3.95805971e-02,
                            -7.65177031e-06, -8.10573832e-01, -8.05367765e-01]]))


def test_n_active_atoms_validation(water_filepath) -> None:
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


def test_subsystem_dft(water_filepath) -> None:
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


def test_subsystem_dft_spin_consistency(water_filepath) -> None:
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
