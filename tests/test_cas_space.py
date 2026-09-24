"""Tests for choosing and diagnosing an active space inside the embedded fragment."""

import numpy as np
import pytest
from conftest import CLOSED_SHELL, MOLECULES, embedded_run, embedding
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from pyscf import mp

from nbed.cas_space import (
    cas_columns_excluding_env,
    diagnose_cas,
    environment_overlap,
    fragment_valence_projector,
    pool_mp2_natural_orbitals,
    reference_weight,
    report_cas_orbitals,
    run_casci,
    run_casscf,
    select_cas_by_mp2_no,
    semicanonicalize,
    single_orbital_entropies,
    unpaired_electrons,
    valence_virtual_pool,
    valence_virtual_rotation,
    von_neumann_entropy,
)

FRAGMENTS = [name for name in CLOSED_SHELL if name != "h2"]


def embedded_hf(name):
    """Embedded huzinaga HF, its environment columns and the EmbedSCF object."""
    run = embedded_run(name, "hf", "huz")
    return run.mf, run.env_cols, embedding(name)


def env_free_virtuals(mf, env_cols):
    """Virtual columns of the embedded mean field that are not environment."""
    return np.setdiff1d(np.where(mf.mo_occ == 0)[0], env_cols)


def span_projector(c, ovlp):
    """AO projector onto the span of S-orthonormal columns."""
    return c @ c.T @ ovlp


@pytest.mark.parametrize("name", FRAGMENTS)
@pytest.mark.parametrize("drop_core", [True, False])
def test_valence_projector_bounds(name, drop_core):
    """C^T P C has eigenvalues in [0, 1] and rank equal to the reference dimension."""
    mf, _, _ = embedded_hf(name)
    atoms = MOLECULES[name][1]
    p_ref, n_ref = fragment_valence_projector(mf.mol, atoms, drop_core=drop_core)
    evals = np.linalg.eigvalsh(mf.mo_coeff.T @ p_ref @ mf.mo_coeff)
    assert evals.min() > -1e-10
    assert evals.max() < 1 + 1e-8
    assert int((evals > 1e-6).sum()) == n_ref


@pytest.mark.parametrize("name", FRAGMENTS)
def test_valence_rotation(name):
    """The rotated block is orthonormal, spans the input and is ordered by weight."""
    mf, env_cols, _ = embedded_hf(name)
    ovlp = mf.get_ovlp()
    c_vir = mf.mo_coeff[:, env_free_virtuals(mf, env_cols)]
    c_rot, weights, energies = valence_virtual_rotation(
        mf.mol, c_vir, MOLECULES[name][1], fock_mo=mf.mo_energy[env_free_virtuals(mf, env_cols)]
    )
    assert np.allclose(c_rot.T @ ovlp @ c_rot, np.eye(c_rot.shape[1]), atol=1e-8)
    assert np.allclose(span_projector(c_rot, ovlp), span_projector(c_vir, ovlp), atol=1e-8)
    assert np.all(np.diff(weights) <= 1e-12)
    assert np.all((weights > -1e-10) & (weights < 1 + 1e-10))
    # energies are expectation values, so their sum is the trace of the block
    assert np.isclose(energies.sum(), mf.mo_energy[env_free_virtuals(mf, env_cols)].sum())


@settings(max_examples=25)
@given(st.sampled_from(FRAGMENTS), st.data())
def test_semicanonicalize(name, data):
    """Any rotated subset of virtuals comes back with a diagonal Fock and the same span."""
    mf, env_cols, _ = embedded_hf(name)
    ovlp = mf.get_ovlp()
    vir = env_free_virtuals(mf, env_cols)
    cols = data.draw(st.lists(st.sampled_from(list(vir)), min_size=1, unique=True))
    k = len(cols)
    raw = data.draw(arrays(np.float64, (k, k), elements=st.floats(-1, 1)))
    q, r = np.linalg.qr(raw + 3 * np.eye(k))
    assume(np.abs(np.diag(r)).min() > 1e-3)
    c_block = mf.mo_coeff[:, cols] @ q

    c_new, energies = semicanonicalize(c_block, mf.mo_coeff, mf.mo_energy, ovlp)
    assert np.allclose(span_projector(c_new, ovlp), span_projector(c_block, ovlp), atol=1e-8)
    rot = mf.mo_coeff.T @ ovlp @ c_new
    fock = np.einsum("ip,i,iq->pq", rot, mf.mo_energy, rot)
    assert np.allclose(fock, np.diag(energies), atol=1e-8)
    assert np.all(np.diff(energies) >= -1e-12)
    assert np.allclose(np.sort(energies), np.sort(mf.mo_energy[cols]), atol=1e-8)


def test_pool_too_large():
    """Asking for more pool orbitals than virtuals raises an error."""
    mf, env_cols, _ = embedded_hf("water")
    c_vir = mf.mo_coeff[:, env_free_virtuals(mf, env_cols)]
    with pytest.raises(ValueError, match="pool"):
        valence_virtual_pool(mf.mol, c_vir, [0], c_vir.shape[1] + 1)


@pytest.mark.parametrize("name", FRAGMENTS)
def test_pool_mp2_matches_frozen_mp2(name):
    """With every environment-free virtual in the pool, pool MP2 is plain frozen MP2."""
    mf, env_cols, _ = embedded_hf(name)
    occ = np.where(mf.mo_occ > 0)[0]
    pool = env_free_virtuals(mf, env_cols)
    result = pool_mp2_natural_orbitals(mf, occ, pool)

    ref = mp.MP2(mf, frozen=[int(i) for i in env_cols])
    ref.verbose = 0
    ref.kernel()
    assert np.isclose(result.e_corr, ref.e_corr)
    assert result.e_corr < 0
    assert np.all((result.occ_vir >= -1e-10) & (result.occ_vir <= 2))
    assert np.all((result.occ_occ >= 0) & (result.occ_occ <= 2 + 1e-10))
    assert np.all(np.diff(result.occ_vir) <= 1e-12)
    assert np.all(np.diff(result.occ_occ) >= -1e-12)
    # MP2 conserves the electron count within the correlated space
    assert np.isclose(result.occ_vir.sum() + result.occ_occ.sum(), 2 * len(occ))


def test_pool_mp2_rejects_environment():
    """A pool that still holds a huge-energy environment orbital is refused."""
    run = embedded_run("water", "hf", "mu")
    mf = run.mf
    occ = np.where(mf.mo_occ > 0)[0]
    with pytest.raises(ValueError, match="environment"):
        pool_mp2_natural_orbitals(mf, occ, np.where(mf.mo_occ == 0)[0])


@pytest.mark.parametrize("name", FRAGMENTS)
@pytest.mark.parametrize("n_cas_occ", [None, 1])
def test_select_cas_by_mp2_no(name, n_cas_occ):
    """The selected CAS is orthonormal, environment free and electron-consistent."""
    mf, env_cols, emb = embedded_hf(name)
    ovlp = mf.get_ovlp()
    n_vir = min(2, len(env_free_virtuals(mf, env_cols)))
    sel = select_cas_by_mp2_no(mf, MOLECULES[name][1], n_vir, n_cas_occ=n_cas_occ,
                               env_cols=env_cols, valence_tol=None)
    assert np.allclose(sel.c_full.T @ ovlp @ sel.c_full, np.eye(sel.c_full.shape[1]),
                       atol=1e-8)
    assert len(np.intersect1d(sel.mo_cas_idxs, env_cols)) == 0
    assert sel.ncas == len(sel.mo_cas_idxs)
    n_occ = len(sel.cas_occ_cols)
    assert sel.nelecas == (n_occ, n_occ)
    assert np.all(np.diff(sel.cas_natural_occ) <= 1e-12)
    # still orthogonal to the originally frozen environment
    c_env = emb.C_full_reidx[:, emb.env_idx_occ]
    assert environment_overlap(sel.c_full[:, sel.mo_cas_idxs], c_env, ovlp) < 1e-6

    report = report_cas_orbitals(mf.mol, sel.c_full[:, sel.mo_cas_idxs], MOLECULES[name][1],
                                 n_occ, occ_no=sel.cas_natural_occ)
    assert len(report.splitlines()) == sel.ncas + 2


def test_select_cas_rejects_open_shell():
    """MP2 natural-orbital selection is closed-shell only."""
    run = embedded_run("methyl", "hf", "huz")
    with pytest.raises(NotImplementedError):
        select_cas_by_mp2_no(run.mf, [0], 1, env_cols=run.env_cols)


def test_select_cas_bad_sizes():
    """Impossible pool or occupied counts raise ValueError."""
    mf, env_cols, _ = embedded_hf("methanol")
    with pytest.raises(ValueError):
        select_cas_by_mp2_no(mf, [0, 1], 2, n_pool=1, env_cols=env_cols)
    with pytest.raises(ValueError):
        select_cas_by_mp2_no(mf, [0, 1], 1, n_cas_occ=99, env_cols=env_cols)


@st.composite
def occupations_and_env(draw):
    """Aufbau occupations with some virtual columns marked as environment."""
    n_occ = draw(st.integers(1, 6))
    n_vir = draw(st.integers(1, 10))
    mo_occ = np.array([2] * n_occ + [0] * n_vir)
    vir = list(range(n_occ, n_occ + n_vir))
    env = draw(st.lists(st.sampled_from(vir), unique=True, max_size=n_vir))
    return mo_occ, np.array(sorted(env), dtype=int)


@given(occupations_and_env(), st.integers(1, 6), st.integers(0, 10))
def test_cas_columns_excluding_env(case, n_occ, n_vir):
    """Environment columns are never chosen, whatever the request."""
    mo_occ, env = case
    n_free = int((mo_occ == 0).sum()) - len(env)
    if n_occ > (mo_occ > 0).sum() or n_vir > n_free:
        with pytest.raises(ValueError):
            cas_columns_excluding_env(mo_occ, env, n_occ, n_vir)
        return
    idx, nelecas, window, collision = cas_columns_excluding_env(
        mo_occ, env, n_occ, n_vir, nelectron=int(mo_occ.sum()))
    assert len(idx) == n_occ + n_vir
    assert len(np.intersect1d(idx, env)) == 0
    assert nelecas == (n_occ, n_occ)
    assert np.array_equal(collision, np.intersect1d(window, env))
    # the occupied part is the frontier, the virtual part the lowest safe virtuals
    assert np.all(mo_occ[idx[:n_occ]] == 2)
    assert idx[n_occ - 1] == np.where(mo_occ > 0)[0][-1]


occupations = st.lists(st.floats(0, 2), min_size=1, max_size=12)


@given(occupations, st.randoms(use_true_random=False))
def test_correlation_measures(occ, rng):
    """Unpaired count and entropy are non-negative, bounded and permutation invariant."""
    n_u, n_u_nl = unpaired_electrons(occ)
    s = von_neumann_entropy(occ)
    assert 0 <= n_u <= len(occ) + 1e-12
    assert 0 <= n_u_nl <= len(occ) + 1e-12
    assert -1e-12 <= s <= len(occ) * np.log(2) + 1e-12

    shuffled = list(occ)
    rng.shuffle(shuffled)
    assert np.isclose(unpaired_electrons(shuffled)[0], n_u)
    assert np.isclose(von_neumann_entropy(shuffled), s)


@given(st.lists(st.sampled_from([0.0, 2.0]), min_size=1, max_size=8))
def test_correlation_measures_vanish_for_determinant(occ):
    """A closed-shell determinant has no unpaired electrons and no entropy."""
    assert unpaired_electrons(occ) == (0.0, 0.0)
    assert von_neumann_entropy(occ) == 0.0


def test_correlation_measures_peak_at_half_filling():
    """A singly occupied natural orbital is maximally unpaired."""
    assert unpaired_electrons([1.0]) == (1.0, 1.0)
    assert np.isclose(von_neumann_entropy([1.0]), np.log(2))


@given(arrays(np.float64, (3, 4), elements=st.floats(-1, 1)))
def test_reference_weight(ci):
    """c0^2 <= largest weight <= 1 for a normalised CI vector."""
    norm = np.linalg.norm(ci)
    assume(norm > 1e-3)
    c0, dom = reference_weight(ci / norm)
    assert 0 <= c0 <= dom + 1e-12 <= 1 + 1e-12
    assert reference_weight([ci / norm]) == (c0, dom)


@pytest.mark.parametrize("name", FRAGMENTS)
def test_casci_and_diagnostics(name):
    """CASCI lies below the embedded HF it contains, and its diagnostics are sane."""
    mf, env_cols, _ = embedded_hf(name)
    idx, nelecas, _, _ = cas_columns_excluding_env(mf.mo_occ, env_cols, 2, 2)
    casci = run_casci(mf, len(idx), nelecas, mo_cas_idxs=idx)
    assert casci.e_tot <= mf.e_tot + 1e-8

    diag = diagnose_cas(casci, verbose=False)
    assert np.isclose(diag.natural_occ.sum(), sum(nelecas))
    assert np.all(np.diff(diag.natural_occ) <= 1e-12)
    assert 0 < diag.reference_weight <= diag.dominant_weight <= 1 + 1e-12
    s, total = single_orbital_entropies(casci)
    assert np.all((s >= -1e-12) & (s <= np.log(4) + 1e-12))
    assert np.isclose(total, s.sum())
    assert "verdict" in diag.report()


@pytest.mark.parametrize("name", ["water", "methanol"])
def test_casscf_with_mu_stays_out_of_environment(name):
    """With the projector in hcore (mu), CASSCF keeps clear of the environment."""
    run = embedded_run(name, "hf", "mu")
    mf, emb = run.mf, embedding(name)
    idx, nelecas, _, _ = cas_columns_excluding_env(mf.mo_occ, run.env_cols, 2, 2)
    c_env = emb.C_full_reidx[:, emb.env_idx_occ]
    casscf = run_casscf(mf, len(idx), nelecas, mo_cas_idxs=idx, c_env=c_env, env_tol=1e-4)
    casci = run_casci(mf, len(idx), nelecas, mo_cas_idxs=idx)
    assert casscf.e_tot <= casci.e_tot + 1e-8
