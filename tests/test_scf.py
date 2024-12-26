"""Test SCF Functions."""

from nbed.scf import _huzinaga_fock_operator, huzinaga_scf


def test_fock_operator_output(restricted_driver):
    scf = restricted_driver.embedded_scf
    dm_env = restricted_driver.localized_system.dm_enviro
    _huzinaga_fock_operator(scf, 0, 0, dm_env, None)


def test_fock_operator_restriction_match():
    pass


def test_fock_diis_output():
    pass


def test_hf_energy_output():
    pass


def test_hf_energy_restriction_match():
    pass


def test_ks_energy_output():
    pass


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
