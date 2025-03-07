"""File to contain tests of the driver.py script."""

import logging

import numpy as np
import pytest
from numpy import isclose
from pyscf.lib.misc import StreamObject

from nbed.driver import NbedDriver
from nbed.exceptions import NbedConfigError

logger = logging.getLogger(__name__)

@pytest.fixture
def mu_driver(driver_args) -> NbedDriver:
    driver_args["projector"] = "mu"
    return NbedDriver(**driver_args).embed()

@pytest.fixture
def mu_unrestricted_driver(driver_args) -> NbedDriver:
    driver_args["projector"] = "mu"
    driver_args["force_unrestricted"] = True
    return NbedDriver(**driver_args).embed()

@pytest.fixture
def huz_driver(driver_args) -> NbedDriver:
    driver_args["projector"] = "huzinaga"
    return NbedDriver(**driver_args).embed()

@pytest.fixture
def huz_unrestricted_driver(driver_args) -> NbedDriver:
    driver_args["projector"] = "huzinaga"
    driver_args["force_unrestricted"] = True
    return NbedDriver(**driver_args).embed()

def test_restricted_projector_results_match(mu_driver, huz_driver) -> None:
    assert mu_driver._mu is not {} and mu_driver._huzinaga is None
    assert huz_driver._huzinaga is not {} and huz_driver._mu is None
    assert mu_driver._mu.keys() == huz_driver._huzinaga.keys()

def test_unrestricted_projector_results_match(mu_unrestricted_driver, huz_unrestricted_driver) -> None:
    assert mu_unrestricted_driver._mu is not {} and mu_unrestricted_driver._huzinaga is None
    assert huz_unrestricted_driver._huzinaga is not {} and huz_unrestricted_driver._mu is None
    assert mu_unrestricted_driver._mu.keys() == huz_unrestricted_driver._huzinaga.keys()

def test_projectors_scf_match(mu_driver, huz_driver) -> None:
    mu_scf = mu_driver.embedded_scf
    huz_scf = huz_driver.embedded_scf
    assert mu_scf.converged is True
    assert huz_scf.converged is True

    assert type(mu_scf) is type(huz_scf)
    assert mu_scf.mo_coeff.shape == huz_scf.mo_coeff.shape
    assert mu_scf.mo_occ.shape == huz_scf.mo_occ.shape
    assert mu_scf.mo_energy.shape == huz_scf.mo_energy.shape
    assert np.isclose(mu_scf.e_tot, huz_scf.e_tot)

def test_unrestricted_projectors_scf_match(mu_unrestricted_driver, huz_unrestricted_driver) -> None:
    mu_scf = mu_unrestricted_driver.embedded_scf
    huz_scf = huz_unrestricted_driver.embedded_scf
    assert mu_scf.converged is True
    assert huz_scf.converged is True

    assert type(mu_scf) is type(huz_scf)
    assert mu_scf.mo_coeff.shape == huz_scf.mo_coeff.shape
    assert mu_scf.mo_occ.shape == huz_scf.mo_occ.shape
    assert mu_scf.mo_energy.shape == huz_scf.mo_energy.shape
    assert np.isclose(mu_scf.e_tot, huz_scf.e_tot)

def test_incorrect_geometry_path() -> None:
    """Test to make sure that FileNotFoundError is thrown if invalid path to xyz geometry file is given"""
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

def test_driver_standard_xyz_string_input(restricted_driver) -> None:
    """test to check driver works... raw xyz string given"""

    assert isinstance(restricted_driver.embedded_scf, StreamObject)
    assert isclose(restricted_driver.classical_energy, -3.5867934952241356)
    assert np.allclose(
        restricted_driver.embedded_scf.mo_coeff,
        np.array([
            [-3.88142342e-03,  3.02684557e-01,  4.52415720e-01, -1.27604882e-05, 1.13974737e+00,  5.86125954e-02],
            [ 9.95680230e-01, -2.14527741e-01,  1.05457231e-01, -2.70841960e-06, 9.44244077e-02, -8.85885271e-02],
            [ 2.14382088e-02,  8.09145086e-01, -5.27807618e-01,  1.53183816e-05,-6.24665203e-01,  5.88976215e-01],
            [-3.37254332e-03, -1.14106506e-01,  4.36409575e-01,  6.36016563e-01,-2.56248456e-01,  5.35976802e-01],
            [ 4.16624471e-03,  1.97283723e-01,  5.76343342e-01, -3.87702724e-01,-7.99990268e-01, -2.44425249e-01],
            [ 5.63571422e-03,  2.23407976e-01, -8.10867200e-02,  6.67210258e-01,-2.20570850e-01, -6.52956233e-01],
            [-1.49279773e-02, -1.68597526e-01,  3.95805969e-02, -7.65174279e-06, 6.66133815e-16, -1.14264919e+00]]
        ),
    )


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

    driver = NbedDriver(**args).embed()

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
    ).embed()

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
    ).embed()
    # Could be problems with caching here

    assert isclose(restricted_driver.e_act, unrestricted_driver.e_act)
    assert isclose(restricted_driver.e_env, unrestricted_driver.e_env)
    assert isclose(restricted_driver.two_e_cross, unrestricted_driver.two_e_cross)
    assert isclose(
        restricted_driver.classical_energy, unrestricted_driver.classical_energy
    )


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


if __name__ == "__main__":
    pass
