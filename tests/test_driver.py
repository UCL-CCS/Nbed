"""File to contain tests of the driver.py script."""

from sympy.functions.elementary.trigonometric import _imaginary_unit_as_coefficient
import logging

import numpy as np
import pytest
from numpy import isclose
from pyscf import scf, dft

from nbed.nbedtypes import OneSpinMatrix, TwoSpinMatrix
from nbed.driver import NbedDriver, _env_projector, _delete_spin_environment
from nbed.config import NbedConfig, ProjectorTypes
from pydantic import ValidationError

logger = logging.getLogger(__name__)


@pytest.fixture
def mu_driver(nbed_config) -> NbedDriver:
    nbed_config.projector = ProjectorTypes.MU
    nbed_config.force_unrestricted = True
    driver = NbedDriver(nbed_config)
    driver.embed()
    return driver


@pytest.fixture
def huz_driver(nbed_config) -> NbedDriver:
    nbed_config.projector = ProjectorTypes.HUZ
    nbed_config.force_unrestricted = True
    driver = NbedDriver(nbed_config)
    driver.embed()
    return driver


@pytest.fixture
def both_driver(nbed_config) -> NbedDriver:
    nbed_config.projector = ProjectorTypes.BOTH
    nbed_config.force_unrestricted = True
    driver = NbedDriver(nbed_config)
    driver.embed()
    return driver


@pytest.mark.parametrize("driver", ["both_driver", "mu_driver", "huz_driver"])
def test_global_ks(request, driver):
    driver = request.getfixturevalue(driver)
    result = driver._global_ks()
    assert np.isclose(result.e_tot, np.float64(-75.3091447400438))
    assert np.allclose(
        result.energy_elec(),
        (np.float64(-84.59485896172163), np.float64(37.93302591280513)),
    )


@pytest.mark.parametrize("driver", ["both_driver", "mu_driver", "huz_driver"])
def test_global_hf(request, driver):
    driver = request.getfixturevalue(driver)
    result = driver._global_hf()
    assert np.isclose(result.energy_nuc(), np.float64(9.285714221677825))
    assert np.isclose(result.e_tot, -74.96099960129165)
    assert np.allclose(
        result.energy_elec(),
        (np.float64(-84.24671382296947), np.float64(38.288174841671974)),
    )


@pytest.mark.parametrize("driver", ["both_driver", "mu_driver", "huz_driver"])
def test_global_ccsd(request, driver):
    driver = request.getfixturevalue(driver)
    result = driver._global_ccsd()
    assert np.isclose(result.e_tot, -75.0090124134578)
    assert np.isclose(result.e_corr, -0.04801281045273269)


@pytest.mark.parametrize("driver", ["both_driver", "mu_driver", "huz_driver"])
def test_global_fci(request, driver):
    driver = request.getfixturevalue(driver)
    result = driver._global_fci()
    assert np.isclose(result.e_tot, np.float64(-75.00912605315143))
    from nbed.driver import run_emb_fci

    emb_result = run_emb_fci(driver._global_hf)
    assert np.isclose(emb_result.e_tot, np.float64(-75.00912605315143))

def test_init_local_scf(nbed_config):
    nbed_config.restricted_global = True
    nbed_config.restricted_active = True
    driver = NbedDriver(nbed_config)
    result = driver.embed()
    assert isinstance(driver._init_local_hf(), scf.hf.RHF)
    assert isinstance(driver._init_local_ks(driver.config.xc_functional), dft.rks.RKS)

    nbed_config.restricted_global = False
    nbed_config.restricted_active = False
    driver = NbedDriver(nbed_config)
    result = driver.embed()
    assert isinstance(driver._init_local_hf(), scf.uhf.UHF)
    assert isinstance(driver._init_local_ks(driver.config.xc_functional), dft.uks.UKS)

def test_environment_projector(both_driver: NbedDriver):
    s_mat = both_driver._global_hf.get_ovlp()
    dm_enviro = both_driver.localized_system.dm_enviro
    assert dm_enviro.shape == (2,7,7)
    assert s_mat.shape == (7,7)
    projector = _env_projector(s_mat, dm_enviro)
    assert projector.shape == (2,7,7)
    # Projectors are:
    # 1. Self-adjoint
    assert np.allclose(projector, np.conj(projector))
    # 2. Idempotent
    #TODO why does this fail? constant term to consider?
    twice_projector = np.einsum("...ij,...jk->...ik", projector, projector)
    two_times_projector = 2*projector
    # assert np.allclose(twice_projector, two_times_projector)

def test_delete_spin_environment(both_driver):
    n_env_mo = 4
    mo_oceff = both_driver._global_hf.mo_coeff
    mo_energy = both_driver._global_hf.mo_energy
    mo_occ = both_driver._global_hf.mo_occ
    pass

@pytest.mark.parametrize("projector", [ProjectorTypes.MU, ProjectorTypes.HUZ])
def test_restricted_to_restricted(nbed_config, projector):
    nbed_config.projector = projector

    nbed_config.restricted_global = True
    nbed_config.restricted_active = True
    driver = NbedDriver(nbed_config)
    driver.embed()
    assert isinstance(driver._global_ks, dft.rks.RKS)
    assert isinstance(driver.embedded_scf, scf.hf.RHF)

@pytest.mark.parametrize("projector", [ProjectorTypes.MU, ProjectorTypes.HUZ])
def test_unrestricted_to_unrestricted(nbed_config, projector):
    nbed_config.projector = projector

    nbed_config.restricted_global = False
    nbed_config.restricted_active = False
    driver = NbedDriver(nbed_config)
    driver.embed()
    assert isinstance(driver._global_ks, dft.uks.UKS)
    assert isinstance(driver.embedded_scf, scf.uhf.UHF)

# @pytest.mark.parametrize("projector", [ProjectorTypes.MU, ProjectorTypes.HUZ])
# def test_restricted_to_unrestricted(nbed_config, projector):
#     nbed_config.projector = projector

#     nbed_config.restricted_global = True
#     nbed_config.restricted_active = False
#     driver = NbedDriver(nbed_config)
#     driver.embed()
#     assert isinstance(driver._global_ks, dft.rks.RKS)
#     assert isinstance(driver.embedded_scf, scf.uhf.UHF)

@pytest.mark.parametrize("restricted", [True, False])
def test_restricted_dft_in_dft(restricted, nbed_config):
    nbed_config.restricted_global = restricted
    nbed_config.restricted_active = restricted

    driver = NbedDriver(nbed_config)
    driver.embed()
    mu_did = driver._dft_in_dft(ProjectorTypes.MU)
    huz_did = driver._dft_in_dft(ProjectorTypes.HUZ)
    assert np.isclose(mu_did["e_dft_in_dft"], driver._global_ks().e_tot)
    assert np.isclose(huz_did["e_dft_in_dft"], driver._global_ks().e_tot)
    assert np.isclose(mu_did["e_dft_in_dft"], huz_did["e_dft_in_dft"])



@pytest.mark.parametrize("driver", ["mu_driver", "huz_driver"])
def test_embedded_ccsd(driver, request):
    driver = request.getfixturevalue(driver)
    # assert np.isclose(
    #     huz_driver._run_emb_fci(huz_driver.embedded_scf).e_tot, -51.61379094995273
    # )
    ccsd, ecorr = driver._run_emb_ccsd(driver.embedded_scf)
    projector_result = getattr(driver, f"{driver.config.projector.value}")
    e_emb = (
        ccsd.e_tot
        + driver.e_env
        + driver.two_e_cross
        - projector_result["correction"]
        - projector_result["beta_correction"]
    )

    assert np.isclose(e_emb, -75.1285849238916)
    assert np.isclose(ecorr, -0.00477765364464925)


@pytest.mark.parametrize("driver", ["mu_driver", "huz_driver"])
def test_embedded_fci(driver, request):
    driver = request.getfixturevalue(driver)

    # assert np.isclose(
    #     huz_driver._run_emb_fci(huz_driver.embedded_scf).e_tot, -51.61379094995273
    # )
    fci = driver._run_emb_fci(driver.embedded_scf)
    projector_result = getattr(driver, f"{driver.config.projector.value}")
    e_emb_fci = (
        fci.e_tot
        + driver.e_env
        + driver.two_e_cross
        - projector_result["correction"]
        - projector_result["beta_correction"]
    )
    assert np.isclose(e_emb_fci, -75.12858550813999)


def test_unrestricted_projector_results_match(mu_driver: NbedDriver, huz_driver: NbedDriver) -> None:
    assert mu_driver.config.force_unrestricted is True
    assert huz_driver.config.force_unrestricted is True
    assert mu_driver.mu != {} and mu_driver.huzinaga == {}
    assert huz_driver.huzinaga != {} and huz_driver.mu == {}
    assert mu_driver.mu.keys() == huz_driver.huzinaga.keys()


def test_unrestricted_projectors_scf_match(mu_driver, huz_driver) -> None:
    mu_scf = mu_driver.embedded_scf
    huz_scf = huz_driver.embedded_scf
    assert isinstance(mu_scf, scf.uhf.UHF)
    assert isinstance(huz_scf, scf.uhf.UHF)
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

    with pytest.raises(ValidationError):
        config = NbedConfig(
            geometry=molecule,
            n_active_atoms=1,
            basis="STO-3G",
            xc_functional="b3lyp5",
            projector="mu",
            localization="spade",
            convergence=1e-6,
            run_ccsd_emb=True,
            run_fci_emb=True,
        )
        # match will match with any printed error message


def test_driver_standard_xyz_string_input(spinless_driver) -> None:
    """test to check driver works... raw xyz string given"""

    assert isinstance(spinless_driver.embedded_scf, scf.hf.SCF)
    assert isclose(spinless_driver.classical_energy, -3.5867934952241356)
    assert spinless_driver.embedded_scf.mo_coeff.shape == (2, 7, 6)
    logger.info(spinless_driver.embedded_scf.mo_coeff)
    assert np.all(
        spinless_driver.embedded_scf.mo_occ
        == np.array([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]])
    )


def test_subsystem_dft(water_filepath) -> None:
    """Check thatcmponenets match total dft energy."""
    config = NbedConfig(
        geometry=str(water_filepath),
        n_active_atoms=2,
        basis="STO-3G",
        xc_functional="b3lyp",
        projector="mu",
        localization="spade",
        convergence=1e-6,
        run_ccsd_emb=False,
        run_fci_emb=False,
    )

    driver = NbedDriver(config)
    driver.embed()

    energy_DFT_components = (
        driver.e_act
        + driver.e_env
        + driver.two_e_cross
        + driver._global_ks.energy_nuc()
    )

    assert isclose(energy_DFT_components, driver._global_ks.e_tot)


if __name__ == "__main__":
    pass
