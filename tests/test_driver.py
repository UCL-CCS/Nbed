"""File to contain tests of the driver.py script."""

import logging

import numpy as np
import pytest
from numpy import isclose
from pyscf.lib.misc import StreamObject

from nbed.driver import NbedDriver
from nbed.config import NbedConfig, ProjectorEnum
from pydantic import ValidationError

logger = logging.getLogger(__name__)


@pytest.fixture
def mu_driver(nbed_config) -> NbedDriver:
    nbed_config.projector = ProjectorEnum.MU
    driver = NbedDriver(nbed_config)
    driver.embed()
    return driver

@pytest.fixture
def huz_driver(nbed_config) -> NbedDriver:
    nbed_config.projector = ProjectorEnum.HUZ
    driver = NbedDriver(nbed_config)
    driver.embed()
    return driver

@pytest.fixture
def both_driver(nbed_config) -> NbedDriver:
    nbed_config.projector = ProjectorEnum.BOTH
    driver = NbedDriver(nbed_config)
    driver.embed()
    return driver

@pytest.mark.parametrize("driver",["mu_driver", "huz_driver"])
def test_global_ks(request, driver):
    driver = request.getfixturevalue(driver)
    result = driver._global_hf()
    assert np.isclose(result.e_tot, np.float64(-74.96099960129165))
    assert np.allclose(result.energy_elec(), (np.float64(-84.24671382296947), np.float64(38.288162980954326)))

@pytest.mark.parametrize("driver",["mu_driver", "huz_driver"])
def test_global_hf(request, driver):
    driver = request.getfixturevalue(driver)
    result = driver._global_hf()
    assert np.isclose(result.energy_nuc(), np.float64(9.285714221677825))
    assert np.isclose(result.e_tot, -74.96099960129165)
    assert np.allclose(result.energy_elec(), (np.float64(-84.24671382296947), np.float64(38.288174841671974)))

@pytest.mark.parametrize("driver",["mu_driver", "huz_driver"])
def test_global_ccsd(request, driver):
    driver = request.getfixturevalue(driver)
    result = driver._global_ccsd()
    assert np.isclose(result.e_tot, -75.0090124134578)
    assert np.isclose(result.e_corr, -0.04801281045273269)

@pytest.mark.parametrize("driver",["mu_driver", "huz_driver"])
def test_global_fci(request, driver):
    driver = request.getfixturevalue(driver)
    result = driver._global_ccsd()
    assert np.isclose(result.e_tot, np.float64(-75.00912605315143))


def test_restricted_dft_in_dft(mu_driver, huz_driver):
    mu_did = mu_driver._dft_in_dft(ProjectorEnum.MU)
    huz_did = huz_driver._dft_in_dft(ProjectorEnum.HUZ)
    assert np.isclose(mu_did["e_dft_in_dft"], mu_driver._global_ks().e_tot)
    assert np.isclose(huz_did["e_dft_in_dft"], huz_driver._global_ks().e_tot)
    assert np.isclose(mu_did["e_dft_in_dft"], huz_did["e_dft_in_dft"])

@pytest.mark.parametrize("driver",["mu_driver", "huz_driver"])
def test_embedded_ccsd(
    driver, request
):
    driver = request.getfixturevalue(driver)
    # assert np.isclose(
    #     huz_driver._run_emb_fci(huz_driver.embedded_scf).e_tot, -51.61379094995273
    # )
    ccsd, ecorr = driver._run_emb_ccsd(driver.embedded_scf)
    assert np.isclose(
        ccsd.e_tot, -62.2617636081909
    )
    assert np.isclose(ecorr, -0.023809582813064136)

    config = driver.config
    config.n_active_atoms = 1
    driver = NbedDriver(driver.config)
    driver.embed()
    ccsd, ecorr = driver._run_emb_ccsd(driver.embedded_scf)
    assert np.isclose(
        ccsd.e_tot, -51.61379094995273
    )
    assert np.isclose(ecorr, -0.004777653647962643)

@pytest.mark.parametrize("driver",["mu_driver", "huz_driver"])
def test_embedded_fci(
    driver, request
):
    driver = request.getfixturevalue(driver)

    # assert np.isclose(
    #     huz_driver._run_emb_fci(huz_driver.embedded_scf).e_tot, -51.61379094995273
    # )
    fci = driver._run_emb_fci(driver.embedded_scf)
    assert np.isclose(
        fci.e_tot, -62.440721085770036
    )

    config = driver.config
    config.n_active_atoms = 1
    driver = NbedDriver(driver.config)
    driver.embed()
    fci = driver._run_emb_fci(driver.embedded_scf)
    assert np.isclose(
        fci.e_tot, -51.61379094995273
    )


def test_restricted_projector_results_match(mu_driver, huz_driver) -> None:
    assert mu_driver._mu is not {} and mu_driver._huzinaga is None
    assert huz_driver._huzinaga is not {} and huz_driver._mu is None
    assert mu_driver._mu.keys() == huz_driver._huzinaga.keys()


def test_unrestricted_projector_results_match(
    mu_unrestricted_driver, huz_unrestricted_driver
) -> None:
    assert (
        mu_unrestricted_driver._mu is not {}
        and mu_unrestricted_driver._huzinaga is None
    )
    assert (
        huz_unrestricted_driver._huzinaga is not {}
        and huz_unrestricted_driver._mu is None
    )
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


def test_unrestricted_projectors_scf_match(
    mu_unrestricted_driver, huz_unrestricted_driver
) -> None:
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

    assert isinstance(spinless_driver.embedded_scf, StreamObject)
    assert isclose(spinless_driver.classical_energy, -3.5867934952241356)
    assert spinless_driver.embedded_scf.mo_coeff.shape == (2, 7, 6)
    logger.info(spinless_driver.embedded_scf.mo_coeff)
    assert np.allclose(
        spinless_driver.embedded_scf.mo_coeff,
        np.array(
            [
                [
                    [
                        -3.88169197e-03,
                        3.02544038e-01,
                        4.52042238e-01,
                        -1.27384527e-05,
                        1.13993327e00,
                        5.86099198e-02,
                    ],
                    [
                        9.95679868e-01,
                        -2.14543928e-01,
                        1.05426258e-01,
                        -2.70618621e-06,
                        9.44340443e-02,
                        -8.85833494e-02,
                    ],
                    [
                        2.14398611e-02,
                        8.09243697e-01,
                        -5.27600623e-01,
                        1.53044886e-05,
                        -6.24745187e-01,
                        5.88951777e-01,
                    ],
                    [
                        -3.37223926e-03,
                        -1.14058553e-01,
                        4.36488700e-01,
                        6.36016567e-01,
                        -2.56122680e-01,
                        5.35986995e-01,
                    ],
                    [
                        4.16618276e-03,
                        1.97370077e-01,
                        5.76609892e-01,
                        -3.87702724e-01,
                        -7.99774028e-01,
                        -2.44438427e-01,
                    ],
                    [
                        5.63538835e-03,
                        2.23412436e-01,
                        -8.10072900e-02,
                        6.67210255e-01,
                        -2.20565105e-01,
                        -6.52973606e-01,
                    ],
                    [
                        -1.49286196e-02,
                        -1.68628326e-01,
                        3.95756986e-02,
                        -7.65096581e-06,
                        -2.22044605e-16,
                        -1.14263308e00,
                    ],
                ],
                [
                    [
                        -3.88120258e-03,
                        3.02828867e-01,
                        4.52785715e-01,
                        -1.27880477e-05,
                        1.13956165e00,
                        5.86137538e-02,
                    ],
                    [
                        9.95680570e-01,
                        -2.14511239e-01,
                        1.05488050e-01,
                        -2.71291533e-06,
                        9.44155408e-02,
                        -8.85936201e-02,
                    ],
                    [
                        2.14366213e-02,
                        8.09044525e-01,
                        -5.28013196e-01,
                        1.53442900e-05,
                        -6.24588147e-01,
                        5.89000849e-01,
                    ],
                    [
                        -3.37272618e-03,
                        -1.14154088e-01,
                        4.36332266e-01,
                        6.36016559e-01,
                        -2.56372168e-01,
                        5.35966138e-01,
                    ],
                    [
                        4.16632660e-03,
                        1.97195765e-01,
                        5.76078098e-01,
                        -3.87702730e-01,
                        -8.00205704e-01,
                        -2.44412221e-01,
                    ],
                    [
                        5.63593611e-03,
                        2.23402226e-01,
                        -8.11671077e-02,
                        6.67210259e-01,
                        -2.20578093e-01,
                        -6.52938508e-01,
                    ],
                    [
                        -1.49271185e-02,
                        -1.68567613e-01,
                        3.95850370e-02,
                        -7.66058991e-06,
                        1.11022302e-16,
                        -1.14266556e00,
                    ],
                ],
            ]
        ),
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
