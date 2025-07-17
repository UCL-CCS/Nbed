"""Test SCF Functions."""

from nbed.scf import huzinaga_scf
import pytest
from numpy.typing import NDArray
import numpy as np


@pytest.fixture()
def dft_potential(spinless_driver) -> NDArray:
    return spinless_driver.embedding_potential


@pytest.fixture()
def dm_environment(spinless_driver) -> NDArray:
    return np.array(
        [
            spinless_driver.localized_system.dm_enviro,
            spinless_driver.localized_system.beta_dm_enviro,
        ]
    )


def test_rks_output(water_rks, dft_potential, dm_environment):
    scf_result = huzinaga_scf(
        water_rks, dft_potential=dft_potential[0], dm_environment=dm_environment[0]
    )
    assert np.allclose(scf_result[0].shape, (7, 7))
    assert np.allclose(
        scf_result[1],
        [
            -17.44629099,
            -0.27614116,
            0.37893061,
            0.89022282,
            1.12092664,
            3.32762378,
            3.86532114,
        ],
    )
    assert np.allclose(scf_result[2].shape, (7, 7))
    assert np.allclose(np.mean(scf_result[2]), 0.1822057642580939)
    assert np.allclose(scf_result[3].shape, (7, 7))
    assert np.allclose(np.mean(scf_result[3]), -0.011214890666261626)
    assert np.allclose(scf_result[4], True)


def test_uks_output(water_uks, dft_potential, dm_environment):
    scf_result = huzinaga_scf(
        water_uks, dft_potential=dft_potential, dm_environment=dm_environment
    )
    assert np.allclose(scf_result[0].shape, (2, 7, 7))
    assert np.allclose(
        scf_result[1],
        [
            [
                -17.29060406,
                -0.28451256,
                0.31504139,
                0.60348835,
                1.0520797,
                2.22020625,
                3.8346852,
            ],
            [
                -17.29048252,
                -0.2845074,
                0.31505414,
                0.60339465,
                1.05207751,
                2.22026345,
                3.83468313,
            ],
        ],
    )
    assert np.allclose(scf_result[2].shape, (2, 7, 7))
    assert np.allclose(np.mean(scf_result[2]), 0.09276688041715254)
    assert np.allclose(scf_result[3].shape, (2, 7, 7))
    assert np.allclose(np.mean(scf_result[3]), -0.02251188710459783)
    assert np.allclose(scf_result[4], True)


def test_rhf_output(water_rhf, dft_potential, dm_environment):
    scf_result = huzinaga_scf(
        water_rhf, dft_potential=dft_potential[0], dm_environment=dm_environment[0]
    )
    assert np.allclose(scf_result[0].shape, (7, 7))
    assert np.allclose(
        scf_result[1],
        [
            -19.346243,
            -0.59741322,
            0.12747464,
            0.6132579,
            0.79561917,
            3.56833278,
            4.1655741,
        ],
    )
    assert np.allclose(scf_result[2].shape, (7, 7))
    assert np.allclose(np.mean(scf_result[2]), 0.17985591319811933)
    assert np.allclose(scf_result[3].shape, (7, 7))
    assert np.allclose(np.mean(scf_result[3]), -0.01224642921175508)
    assert np.allclose(scf_result[4], True)


def test_uhf_output(water_uhf, dft_potential, dm_environment):
    scf_result = huzinaga_scf(
        water_uhf, dft_potential=dft_potential, dm_environment=dm_environment
    )
    assert np.allclose(scf_result[0].shape, (2, 7, 7))
    assert np.allclose(
        scf_result[1],
        [
            [
                -19.18005207,
                -0.618383,
                0.07366692,
                0.39496279,
                0.72192366,
                2.44806433,
                4.12874389,
            ],
            [
                -19.17991953,
                -0.6183819,
                0.07366408,
                0.39491023,
                0.72191934,
                2.44812268,
                4.12874047,
            ],
        ],
    )
    assert np.allclose(scf_result[2].shape, (2, 7, 7))
    assert np.allclose(np.mean(scf_result[2]), 0.0920247346776863)
    assert np.allclose(scf_result[3].shape, (2, 7, 7))
    assert np.allclose(np.mean(scf_result[3]), -0.024315876434944768)
    assert np.allclose(scf_result[4], True)
