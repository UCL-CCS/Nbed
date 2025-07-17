"""Test SCF Functions."""

from nbed.scf import huzinaga_scf
from nbed.scf.huzinaga_hf import huzinaga_HF
from nbed.scf.huzinaga_ks import huzinaga_KS
import pytest
from numpy.typing import NDArray
import numpy as np

@pytest.fixture()
def dft_potential(spinless_driver) -> NDArray:
    return spinless_driver.embedding_potential

@pytest.fixture()
def dm_environment(spinless_driver)-> NDArray:
    return np.array([spinless_driver.localized_system.dm_enviro, spinless_driver.localized_system.beta_dm_enviro])

def test_fock_operator_restriction_match():
    pass


def test_fock_diis_output():
    pass


def test_hf_energy_output():
    pass


def test_hf_energy_restriction_match():
    pass


def test_rks_output(water_rks, dft_potential, dm_environment):
    scf_result = huzinaga_scf(water_rks, dft_potential=dft_potential[0], dm_environment=dm_environment[0])
    assert np.allclose(ks_result.shape[0],0)
    assert np.allclose(ks_result.shape[1],0)
    assert np.allclose(ks_result.shape[2],0)
    assert np.allclose(ks_result.shape[3],0)
    assert np.allclose(ks_result[4],0)

def test_uks_output(water_uks, dft_potential, dm_environment):
    scf_result = huzinaga_scf(water_uks, dft_potential=dft_potential, dm_environment=dm_environment)
    assert np.allclose(ks_result.shape[0],0)
    assert np.allclose(ks_result.shape[1],0)
    assert np.allclose(ks_result.shape[2],0)
    assert np.allclose(ks_result.shape[3],0)
    assert np.allclose(ks_result[4],0)

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
