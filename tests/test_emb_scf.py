"""Tests for EmbedSCF: partitioned densities, projectors and embedded SCF."""

import numpy as np
import pytest
from conftest import (
    CLOSED_SHELL,
    EMBEDDABLE,
    HUZ_SHIFT,
    MOLECULES,
    XC,
    build_mol,
    embedded_run,
    embedding,
    global_ks,
    make_embedding,
)
from hypothesis import given, settings
from hypothesis import strategies as st
from pyscf import dft
from scipy.spatial.transform import Rotation

from nbed.emb_scf import EmbedSCF


def spin_summed(dm):
    """Total density from a restricted or (alpha, beta) density."""
    return dm if dm.ndim == 2 else dm[0] + dm[1]


@pytest.mark.parametrize("name", EMBEDDABLE)
def test_density_and_energy_partition(name):
    """dm_act + dm_env = dm_full, electron counts add up, and E_DFT splits exactly."""
    emb = embedding(name)
    S = emb.Sao
    assert np.allclose(emb.dm_act + emb.dm_env, emb.dm_full)

    n_act = np.trace(spin_summed(emb.dm_act) @ S)
    n_env = np.trace(spin_summed(emb.dm_env) @ S)
    assert np.isclose(n_act, emb.mol_act.nelectron)
    assert np.isclose(n_act + n_env, emb.global_scf_obj.mol.nelectron)

    assert np.isclose(emb.E_act + emb.E_env + emb.E_cross, emb.E_DFT_global)
    assert np.isclose(emb.E_DFT_global + emb.E_nuclear, emb.global_scf_obj.e_tot)


@pytest.mark.parametrize("name", EMBEDDABLE)
def test_subsystem_molecule(name):
    """The embedded molecule keeps the geometry and basis but only the active electrons."""
    emb = embedding(name)
    mol, mol_act = emb.global_scf_obj.mol, emb.mol_act
    assert np.allclose(mol.atom_coords(), mol_act.atom_coords())
    assert mol.nao == mol_act.nao
    assert mol_act.nelectron == int((emb.mo_occ_act > 0).sum() + (emb.mo_occ_act > 1).sum())
    assert emb.SCF_type == ("open-shell" if mol.spin else "closed-shell")


@pytest.mark.parametrize("name", EMBEDDABLE)
def test_projectors(name):
    """mu projector is symmetric PSD of rank n_env; huzinaga projector is idempotent."""
    emb = embedding(name)
    S, C = emb.Sao, emb.C_full_reidx
    n_env = len(emb.env_idx_occ)

    P_mu = emb.get_mu_projector()
    assert np.allclose(P_mu, P_mu.T)
    evals = np.linalg.eigvalsh(P_mu)
    assert evals.min() > -1e-10

    P_huz = emb.get_huz_projector()
    assert np.allclose(P_huz @ P_huz, P_huz)
    assert np.isclose(np.trace(P_huz), n_env)

    # in the MO basis, P picks out exactly the occupied environment columns
    diag = np.diag(C.T @ S @ P_huz @ C)
    others = np.setdiff1d(np.arange(C.shape[1]), emb.env_idx_occ)
    assert np.allclose(diag[emb.env_idx_occ], 1)
    assert np.allclose(diag[others], 0)


@settings(max_examples=10)
@given(st.sampled_from(EMBEDDABLE), st.floats(0.0, 10.0))
def test_huzinaga_operator(name, shift):
    """The huzinaga operator is Hermitian and does not touch the active orbitals."""
    emb = embedding(name)
    fock = emb.global_scf_obj.get_fock(dm=emb.dm_full)
    if fock.ndim == 3:
        from pyscf.scf.rohf import get_roothaan_fock
        fock = get_roothaan_fock((fock[0], fock[1]), emb.dm_full, emb.Sao)
    O = emb.get_huz_operator(fock, level_shift=shift)
    assert np.allclose(O, O.T)

    C_act = emb.C_full_reidx[:, emb.act_cols]
    C_env = emb.C_full_reidx[:, emb.env_idx_occ]
    # within the active block it vanishes; on the environment it is -2 F (+ shift)
    assert np.allclose(C_act.T @ O @ C_act, 0, atol=1e-10)
    f_env = C_env.T @ fock @ C_env
    assert np.allclose(C_env.T @ O @ C_env, -2 * f_env + shift * np.eye(len(f_env)),
                       atol=1e-8)


@pytest.mark.parametrize("name", EMBEDDABLE)
def test_huzinaga_dft_in_dft_is_exact(name):
    """Huzinaga DFT-in-DFT reproduces the global DFT energy."""
    run = embedded_run(name, "dft", "huz")
    assert run.mf.converged
    assert abs(run.e_tot - global_ks(name).e_tot) < 1e-7


@pytest.mark.parametrize("name", EMBEDDABLE)
def test_mu_dft_in_dft_is_exact(name):
    """mu DFT-in-DFT reproduces the global DFT energy to O(1/mu)."""
    run = embedded_run(name, "dft", "mu")
    assert abs(run.e_tot - global_ks(name).e_tot) < 1e-5


def test_small_mu_is_not_exact():
    """A mu too small to lift the environment gives a visibly wrong energy."""
    mf = global_ks("methanol")
    errors = []
    for mu in (1e1, 1e6):
        emb, _, _ = make_embedding(mf, MOLECULES["methanol"][1], mu=mu)
        e, *_ = emb.build_emb_dft(XC, proj_type="mu", warn=False)
        errors.append(abs(e - mf.e_tot))
    assert errors[0] > 1e-2
    assert errors[1] < 1e-6


def test_unconverged_huzinaga_is_reported(capsys):
    """Formamide HF-in-DFT without a level shift is flagged whenever it fails."""
    emb = embedding("formamide")
    _, mf, *_ = emb.build_emb_hf(proj_type="huz", warn=True)
    printed = "did NOT converge" in capsys.readouterr().out
    assert printed == (not mf.converged)


def test_linearly_dependent_basis_partition():
    """Fewer MOs than AOs (as after removing linear dependencies) still partitions."""
    full = global_ks("water")
    mf = full.copy()
    mf.mo_coeff = full.mo_coeff[:, :-1]
    mf.mo_occ = full.mo_occ[:-1]
    mf.mo_energy = full.mo_energy[:-1]
    emb, act, env = make_embedding(mf, [0], n_occ=2, n_vir=1)
    assert np.array_equal(np.union1d(act, env), np.arange(mf.mo_coeff.shape[1]))
    assert emb.C_full_reidx.shape == mf.mo_coeff.shape
    e, *_ = emb.build_emb_dft(XC, proj_type="huz", warn=False)
    assert abs(e - full.e_tot) < 1e-7


@pytest.mark.parametrize("name", EMBEDDABLE)
@pytest.mark.parametrize("method", ["dft", "hf"])
@pytest.mark.parametrize("proj", ["mu", "huz"])
def test_embedded_orbitals_avoid_environment(name, method, proj):
    """Occupied embedded orbitals are orthogonal to the occupied environment."""
    emb = embedding(name)
    mf = embedded_run(name, method, proj).mf
    C_env = emb.C_full_reidx[:, emb.env_idx_occ]
    overlap = np.abs(C_env.T @ emb.Sao @ mf.mo_coeff[:, mf.mo_occ > 0]).max()
    assert overlap < (1e-6 if proj == "mu" else 1e-8)
    assert mf.mol.nelec == emb.mol_act.nelec


@pytest.mark.parametrize("name", EMBEDDABLE)
def test_check_embedding_and_env_cols(name):
    """check_embedding passes and agrees with the env_cols from the SCF.

    Only for huzinaga: check_embedding demands overlaps below 1e-8, and the mu
    projector leaks at O(1/mu), so it is expected to fail there.
    """
    proj = "huz"
    emb = embedding(name)
    run = embedded_run(name, "hf", proj)
    mf = run.mf
    env_cols = emb.check_embedding(mf.mo_coeff, mf.mo_occ, mf.mo_energy, emb.Sao, proj)
    assert np.array_equal(env_cols, run.env_cols)
    assert len(env_cols) == len(emb.env_idx_occ)
    assert np.all(mf.mo_occ[env_cols] == 0)


@pytest.mark.parametrize("name", CLOSED_SHELL[1:])
def test_hf_in_dft_projectors_agree(name):
    """mu and huzinaga HF-in-DFT converge to the same energy."""
    e_mu = embedded_run(name, "hf", "mu").e_tot
    e_huz = embedded_run(name, "hf", "huz").e_tot
    assert abs(e_mu - e_huz) < 1e-5


def test_invalid_projector():
    """An unknown projector type raises an error."""
    with pytest.raises(ValueError, match="Invalid projection"):
        embedding("water").build_emb_hf(proj_type="nonsense")


def test_bad_partition_rejected():
    """Indices that do not cover the MOs are rejected."""
    mf = global_ks("water")
    with pytest.raises(AssertionError):
        EmbedSCF(mf, [0, 1], [2, 3], mf.mo_coeff, mf.mo_occ, mf.get_ovlp(), 4000)


@pytest.mark.parametrize("builder", ["dft", "hf"])
def test_scf_modify_function_is_respected(builder):
    """Settings changed by scf_modify_function survive into the embedded SCF."""
    emb = embedding("water")

    def modify(mf):
        mf.conv_tol = 1e-5
        mf.max_cycle = 7
        return mf

    if builder == "dft":
        _, mf, *_ = emb.build_emb_dft(XC, scf_modify_function=modify, warn=False)
    else:
        _, mf, *_ = emb.build_emb_hf(scf_modify_function=modify, warn=False)
    assert mf.conv_tol == 1e-5
    assert mf.max_cycle == 7


def random_rotation(angles):
    """Rotation matrix from three Euler angles."""
    return Rotation.from_euler("zyx", angles).as_matrix()


def hf_in_dft_at(coords, prune=dft.gen_grid.nwchem_prune):
    """Global KS and huzinaga HF-in-DFT energies of water at the given coordinates."""
    mol = build_mol("water")
    geometry = [(mol.atom_symbol(i), tuple(c)) for i, c in enumerate(coords)]
    mf = dft.RKS(build_mol("water", geometry=geometry), xc=XC)
    mf.grids.prune = prune
    mf.conv_tol = 1e-10
    mf.kernel()
    emb, _, _ = make_embedding(mf, MOLECULES["water"][1])
    e, *_ = emb.build_emb_hf(proj_type="huz", huz_level_shift=HUZ_SHIFT, warn=False)
    return mf.e_tot, e


@settings(max_examples=5)
@given(st.tuples(*[st.floats(-3, 3)] * 3))
def test_embedding_energy_invariant_to_translation(shift):
    """Translating the molecule leaves the HF-in-DFT energy unchanged.

    The grid and the AOs move rigidly with the atoms, so this holds to round-off.
    """
    coords = build_mol("water").atom_coords(unit="Angstrom")
    e_ks, e_emb = hf_in_dft_at(coords + np.array(shift))
    assert abs(e_ks - global_ks("water").e_tot) < 1e-8
    assert abs(e_emb - embedded_run("water", "hf", "huz").e_tot) < 1e-7


@pytest.mark.parametrize("angles", [(1.0, 0.0, 0.0), (0.3, -1.2, 2.0)])
def test_embedding_energy_rotation_is_grid_limited(angles):
    """Rotating the molecule changes the HF-in-DFT energy only through the DFT grid.

    The global energy is invariant to ~1e-7, but E_xc of the partial active density is
    much more orientation-sensitive, and HF-in-DFT does not cancel it the way
    DFT-in-DFT does. For water/B3LYP at the default grid level this is ~2e-4 Ha
    unpruned and ~4e-4 Ha pruned; it only drops to ~2e-5 at grids.level=5 unpruned.
    """
    coords = build_mol("water").atom_coords(unit="Angstrom")
    rotated = coords @ Rotation.from_euler("zyx", angles).as_matrix().T
    ref = hf_in_dft_at(coords, prune=None)
    moved = hf_in_dft_at(rotated, prune=None)
    assert abs(moved[0] - ref[0]) < 1e-6
    assert abs(moved[1] - ref[1]) < 5e-4


@pytest.mark.parametrize("name", CLOSED_SHELL[1:])
def test_mo_integrals(name):
    """Integrals over every fragment orbital have the right shapes and core energy."""
    emb = embedding(name)
    mf = embedded_run(name, "hf", "huz").mf
    env_cols = embedded_run(name, "hf", "huz").env_cols
    keep = np.setdiff1d(np.arange(mf.mo_coeff.shape[1]), env_cols)
    norb = len(keep)
    e_core, h1, eri = emb.get_mo_integrals(mf, mf.mo_coeff, norb, mf.mol.nelec,
                                           mo_cas_idxs=keep)
    assert h1.shape == (norb, norb)
    assert np.allclose(h1, h1.T)
    assert eri.shape == (norb * (norb + 1) // 2,) * 2
    # every electron is active, so the core energy is the nuclear repulsion
    assert np.isclose(e_core, mf.mol.energy_nuc())
    # the diagonal of h1 + J-K over the occupied orbitals rebuilds the SCF energy
    nocc = mf.mol.nelectron // 2
    from pyscf import ao2mo
    g = ao2mo.restore(1, eri, norb)
    occ = [list(keep).index(i) for i in np.where(mf.mo_occ > 0)[0]]
    e_hf = e_core + sum(2 * h1[i, i] for i in occ) + sum(
        2 * g[i, i, j, j] - g[i, j, j, i] for i in occ for j in occ
    )
    assert nocc == len(occ)
    assert np.isclose(e_hf, mf.e_tot)
