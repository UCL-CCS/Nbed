"""Tests for choosing the fragment / environment partition."""

import numpy as np
import pytest
from conftest import MOLECULES, build_mol, global_ks
from hypothesis import given, settings
from hypothesis import strategies as st
from pyscf import dft, gto

from nbed.act_env_space import (
    describe_orbital,
    lowdin_populations,
    orbital_spread,
    select_act_env_space,
)

names = st.sampled_from(sorted(MOLECULES))


@st.composite
def molecule_and_atoms(draw, basis="sto-3g"):
    """A cached mean field together with a non-empty subset of its atoms."""
    name = draw(names)
    mf = global_ks(name, basis)
    atoms = draw(
        st.lists(
            st.integers(0, mf.mol.natm - 1), min_size=1, max_size=mf.mol.natm, unique=True
        )
    )
    return mf, sorted(atoms)


@settings(max_examples=30)
@given(molecule_and_atoms())
def test_populations_partition_each_orbital(case):
    """Lowdin populations over every atom add up to one per MO."""
    mf, atoms = case
    everything = list(range(mf.mol.natm))
    per_atom, total = lowdin_populations(mf.mol, mf.mo_coeff, everything, drop_core_1s=False)
    assert np.allclose(total, 1.0)
    assert np.all(per_atom > -1e-12)

    _, subset = lowdin_populations(mf.mol, mf.mo_coeff, atoms, drop_core_1s=False)
    assert np.all(subset <= total + 1e-12)


@settings(max_examples=20)
@given(molecule_and_atoms(), st.randoms(use_true_random=False))
def test_populations_follow_column_permutation(case, rng):
    """Permuting MO columns permutes the populations in the same way."""
    mf, atoms = case
    perm = list(range(mf.mo_coeff.shape[1]))
    rng.shuffle(perm)
    _, pop = lowdin_populations(mf.mol, mf.mo_coeff, atoms)
    _, pop_perm = lowdin_populations(mf.mol, mf.mo_coeff[:, perm], atoms)
    assert np.allclose(pop[perm], pop_perm)


def test_populations_bad_index():
    """An atom index outside the molecule raises an error."""
    mf = global_ks("water")
    with pytest.raises(IndexError):
        lowdin_populations(mf.mol, mf.mo_coeff, [3])


def test_populations_all_core_raises():
    """A single-shell atom has nothing left once its core has been dropped."""
    # an O carrying only a 1s shell has no valence AOs left
    mol = gto.M(atom="O 0 0 0; H 0 0 1", basis={"O": [[0, [1.0, 1.0]]], "H": "sto-3g"},
                spin=1, verbose=0)
    with pytest.raises(ValueError, match="no AOs left"):
        lowdin_populations(mol, np.eye(mol.nao), [0])


@settings(max_examples=10)
@given(names, st.tuples(*[st.floats(-5, 5)] * 3))
def test_spread_translation_invariant(name, shift):
    """Orbital spreads are non-negative and unchanged by a rigid translation."""
    mf = global_ks(name)
    mol = mf.mol
    moved = mol.set_geom_(mol.atom_coords(unit="Bohr") + np.array(shift), unit="Bohr",
                          inplace=False)
    spread = orbital_spread(mol, mf.mo_coeff)
    assert np.all(spread >= 0)
    # the AO basis moves rigidly with the atoms, so the same coefficients apply
    assert np.allclose(spread, orbital_spread(moved, mf.mo_coeff), atol=1e-8)


def test_describe_orbital():
    """The composition string names only atoms above the cutoff."""
    mf = global_ks("water")
    per_atom, _ = lowdin_populations(mf.mol, mf.mo_coeff, [0, 1, 2], drop_core_1s=False)
    text = describe_orbital(mf.mol, [0, 1, 2], per_atom, 0)
    assert text.startswith("O0:")
    assert describe_orbital(mf.mol, [0], per_atom, 0, cutoff=2.0) == ""


@st.composite
def partition_request(draw):
    """A molecule, fragment atoms and a feasible active orbital count."""
    mf, atoms = draw(molecule_and_atoms())
    n_occ_total = int((mf.mo_occ > 0).sum())
    n_vir_total = int((mf.mo_occ == 0).sum())
    n_occ = draw(st.integers(1, n_occ_total))
    n_vir = draw(st.integers(0, n_vir_total))
    return mf, atoms, n_occ, n_vir


@settings(max_examples=40)
@given(partition_request())
def test_partition_is_complete_and_disjoint(case):
    """Active and environment indices tile the MOs with the requested counts."""
    mf, atoms, n_occ, n_vir = case
    act, env, pop, spread = select_act_env_space(mf, atoms, n_occ, n_vir)
    nmo = mf.mo_coeff.shape[1]
    assert len(np.intersect1d(act, env)) == 0
    assert np.array_equal(np.union1d(act, env), np.arange(nmo))
    assert (mf.mo_occ[act] > 0).sum() == n_occ
    assert (mf.mo_occ[act] == 0).sum() == n_vir
    assert pop.shape == spread.shape == (nmo,)

    # the chosen occupied orbitals are the most populated ones on the fragment
    occ = np.where(mf.mo_occ > 0)[0]
    chosen = np.intersect1d(act, occ)
    rest = np.setdiff1d(occ, chosen)
    if len(rest):
        assert pop[chosen].min() >= pop[rest].max() - 1e-12


def test_partition_defaults_virtuals_to_occupied():
    """With n_vir_active omitted, the fragment gets as many virtuals as occupied."""
    mf = global_ks("methanol")
    act, _, _, _ = select_act_env_space(mf, [0, 1], 2)
    assert (mf.mo_occ[act] == 0).sum() == 2


@settings(max_examples=15)
@given(names, st.floats(0.5, 5.0))
def test_max_spread_filters(name, max_spread):
    """No selected orbital is more diffuse than max_spread."""
    mf = global_ks(name)
    atoms = MOLECULES[name][1]
    spread = orbital_spread(mf.mol, mf.mo_coeff)
    n_occ_ok = int(((mf.mo_occ > 0) & (spread <= max_spread)).sum())
    n_vir_ok = int(((mf.mo_occ == 0) & (spread <= max_spread)).sum())
    if n_occ_ok == 0:
        with pytest.raises(ValueError):
            select_act_env_space(mf, atoms, 1, 0, max_spread=max_spread)
        return
    act, _, _, _ = select_act_env_space(mf, atoms, 1, min(1, n_vir_ok), max_spread=max_spread)
    assert np.all(spread[act] <= max_spread)


def test_partition_rejects_unrestricted():
    """UKS orbitals are refused."""
    mf = dft.UKS(build_mol("water"), xc="lda").run()
    with pytest.raises(ValueError, match="unrestricted"):
        select_act_env_space(mf, [0], 1, 1)


def test_partition_rejects_fractional_occupations():
    """Smeared occupations cannot be split into integer electron counts."""
    mf = global_ks("water")
    fake = mf.copy()
    fake.mo_occ = mf.mo_occ.copy().astype(float)
    fake.mo_occ[4] = 1.5
    fake.mo_occ[5] = 0.5
    with pytest.raises(ValueError, match="fractional"):
        select_act_env_space(fake, [0], 1, 1)


def test_partition_too_many_orbitals():
    """Asking for more orbitals than exist raises a ValueError."""
    mf = global_ks("water")
    with pytest.raises(ValueError, match="eligible"):
        select_act_env_space(mf, [0], 99, 1)
