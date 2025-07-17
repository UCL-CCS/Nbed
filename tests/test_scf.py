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


def test_fock_operator_restriction_match():
    pass


def test_fock_diis_output():
    pass


def test_hf_energy_output():
    pass


def test_hf_energy_restriction_match():
    pass


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


def test_ks_energy_restriction_match():
    pass


def test_scf_convergence():
    pass


def test_scf_convergence_warning():
    pass


def test_hf_output():
    pass


def test_hf_restriction_match():
    pass


def test_ks_output():
    pass


def test_ks_restriction_match():
    pass
