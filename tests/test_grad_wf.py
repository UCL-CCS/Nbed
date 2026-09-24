"""WF-in-DFT gradients from active-space RDMs, as a quantum solver would supply them."""

import types

import numpy as np
import pytest
from conftest import build_mol
from pyscf import dft, fci, gto, mcscf, scf

from nbed.act_env_space import select_act_env_space
from nbed.cas_space import cas_columns_excluding_env, run_casci
from nbed.emb_scf import EmbedSCF
from nbed.grad import (
    _env_columns,
    embedding_gradient,
    rdm_embedding_gradient,
    rdm_fragment_response,
    wf_in_dft_energy,
)

WATER = build_mol("water").atom_coords()


def solve_fci(h1, eri, ncas, nelecas):
    """Stand-in for the downstream (quantum) solver: FCI energy and RDMs."""
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-14
    e, civec = solver.kernel(h1, eri, ncas, nelecas)
    return e, *solver.make_rdm12(civec, ncas, nelecas)


def determinant_rdms(ncas, nocc):
    """RDMs of the closed-shell determinant occupying the first ``nocc`` orbitals."""
    d1 = np.diag([2.0] * nocc + [0.0] * (ncas - nocc))
    d2 = np.einsum("pq,rs->pqrs", d1, d1) - 0.5 * np.einsum("ps,rq->pqrs", d1, d1)
    return d1, d2


class WFInDFT:
    """WF-in-DFT energy of water with the partition and CAS columns held fixed."""

    def __init__(self, xc, proj, rdms="fci", basis="6-31g", n_cas=(2, 2)):
        self.xc, self.proj, self.rdms, self.basis, self.n_cas = xc, proj, rdms, basis, n_cas
        self.act = self.idx = None

    def run(self, coords):
        mol = gto.M(atom=list(zip(["O", "H", "H"], coords)), unit="Bohr",
                    basis=self.basis, verbose=0)
        mf = dft.RKS(mol, xc=self.xc)
        mf.conv_tol = 1e-12
        mf.kernel()
        act, env, _, _ = select_act_env_space(mf, [0], 2, 2)
        # a moderate mu keeps SCF noise out of the finite differences
        emb = EmbedSCF(mf, act, env, mf.mo_coeff, mf.mo_occ, mf.get_ovlp(), 4000,
                       mu_val=1e2)
        e_hf, mf_emb, _, env_cols, env_plus = emb.build_emb_hf(
            proj_type=self.proj, huz_level_shift=1.0, warn=False)
        idx, nelecas, _, _ = cas_columns_excluding_env(mf_emb.mo_occ, env_cols,
                                                       *self.n_cas)
        if self.act is None:
            self.act, self.idx = act, idx
        assert np.array_equal(act, self.act) and np.array_equal(idx, self.idx)
        e_core, h1, eri = emb.get_mo_integrals(mf_emb, mf_emb.mo_coeff, len(idx),
                                               nelecas, mo_cas_idxs=idx)
        if self.rdms == "fci":
            _, d1, d2 = solve_fci(h1, eri, len(idx), nelecas)
        else:
            d1, d2 = determinant_rdms(len(idx), nelecas[0])
        energy = wf_in_dft_energy(e_core, h1, eri, d1, d2, env_plus)
        return types.SimpleNamespace(energy=energy, emb=emb, mf_emb=mf_emb, idx=idx,
                                     d1=d1, d2=d2, e_hf=e_hf)

    def gradient(self, coords):
        r = self.run(coords)
        return rdm_embedding_gradient(r.emb, r.mf_emb, r.idx, r.d1, r.d2,
                                      proj_type=self.proj)

    def finite_difference(self, coords, i, x, h=2e-4):
        """Richardson-extrapolated central difference (O(h^4))."""
        def central(step):
            plus, minus = coords.copy(), coords.copy()
            plus[i, x] += step
            minus[i, x] -= step
            return (self.run(plus).energy - self.run(minus).energy) / (2 * step)
        return (4 * central(h / 2) - central(h)) / 3


@pytest.mark.parametrize("proj", ["mu", "huz"])
@pytest.mark.parametrize("xc", ["hf", "b3lyp"])
def test_wf_in_dft_gradient_matches_finite_difference(xc, proj):
    """CAS(4e,4o)-in-DFT forces from FCI RDMs agree with the energy."""
    model = WFInDFT(xc, proj)
    grad = model.gradient(WATER)
    for i, x in [(0, 2), (1, 1)]:
        assert abs(grad[i, x] - model.finite_difference(WATER, i, x)) < 2e-7
    assert np.abs(grad.sum(axis=0)).max() < 1e-9


@pytest.mark.parametrize("proj", ["mu", "huz"])
def test_determinant_rdms_reproduce_hf_in_dft(proj):
    """With the reference determinant's RDMs the gradient is the HF-in-DFT one."""
    model = WFInDFT("b3lyp", proj, rdms="det")
    r = model.run(WATER)
    assert abs(r.energy - r.e_hf) < 1e-8
    grad = rdm_embedding_gradient(r.emb, r.mf_emb, r.idx, r.d1, r.d2, proj_type=proj)
    assert np.abs(grad - embedding_gradient(r.emb, r.mf_emb, proj_type=proj)).max() < 1e-8


def test_energy_helper_matches_casci():
    """wf_in_dft_energy with FCI RDMs is the CASCI energy plus the DFT corrections."""
    r = WFInDFT("b3lyp", "huz").run(WATER)
    nelecas = (int(round(np.trace(r.d1))) // 2,) * 2
    casci = run_casci(r.mf_emb, len(r.idx), nelecas, mo_cas_idxs=r.idx)
    *_, env_plus = r.emb.build_emb_hf(proj_type="huz", huz_level_shift=1.0, warn=False)
    assert np.isclose(r.energy, casci.e_tot + env_plus, atol=1e-8)


def test_fragment_response_matches_pyscf_casci():
    """Without embedding, the fragment gradient is PySCF's CASCI gradient.

    PySCF caps its CASCI Z-vector at 30 CPHF cycles, which leaves it ~1e-8 Ha/bohr
    from the finite-difference gradient; the conjugate-gradient solve used here is
    tighter, hence the tolerance.
    """
    mol = build_mol("water", "6-31g")
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mc = mcscf.CASCI(mf, 4, 4)
    mc.fcisolver.conv_tol = 1e-14
    mc.kernel()
    d1, d2 = mc.fcisolver.make_rdm12(mc.ci, 4, 4)
    frag = rdm_fragment_response(types.SimpleNamespace(Sao=mf.get_ovlp()), mf,
                                 np.arange(mc.ncore, mc.ncore + 4), d1, d2)
    assert np.abs(frag.grad_fixed_vemb - mc.Gradients().kernel()).max() < 5e-8


def test_relaxed_density_is_hcore_derivative():
    """tr(D_rel V) is the derivative of the fragment energy along hcore + eps V."""
    mol = build_mol("water", "6-31g")
    h0 = scf.RHF(mol).get_hcore()

    def fragment(h):
        mf = scf.RHF(mol)
        mf.get_hcore = lambda *args, **kwargs: h
        mf.conv_tol = 1e-13
        mf.kernel()
        mc = mcscf.CASCI(mf, 4, 4)
        mc.fcisolver.conv_tol = 1e-14
        mc.kernel()
        return mc, mf

    mc, mf = fragment(h0)
    d1, d2 = mc.fcisolver.make_rdm12(mc.ci, 4, 4)
    frag = rdm_fragment_response(types.SimpleNamespace(Sao=mf.get_ovlp()), mf,
                                 np.arange(mc.ncore, mc.ncore + 4), d1, d2)
    v = np.random.default_rng(0).normal(size=h0.shape)
    v = 0.01 * (v + v.T)

    def central(eps):
        return (fragment(h0 + eps * v)[0].e_tot - fragment(h0 - eps * v)[0].e_tot) / (2 * eps)
    numeric = (4 * central(5e-4) - central(1e-3)) / 3
    assert abs(np.einsum("ij,ij", frag.dm_relaxed, v) - numeric) < 1e-8


def test_wf_gradient_rejects_unsupported_input():
    """KS references, environment columns in the CAS and rotated orbitals raise."""
    r = WFInDFT("b3lyp", "huz").run(WATER)
    emb, mf_emb = r.emb, r.mf_emb
    _, mf_dft, *_ = emb.build_emb_dft("b3lyp", proj_type="huz", huz_level_shift=1.0,
                                      warn=False)
    with pytest.raises(NotImplementedError, match="RHF"):
        rdm_embedding_gradient(emb, mf_dft, r.idx, r.d1, r.d2, proj_type="huz")

    bad_idx = np.concatenate([r.idx[:-1], _env_columns(emb, mf_emb)[:1]])
    with pytest.raises(ValueError, match="environment"):
        rdm_embedding_gradient(emb, mf_emb, bad_idx, r.d1, r.d2, proj_type="huz")

    rotated = mf_emb.copy()
    occ = mf_emb.mo_occ > 0
    rotated.mo_coeff = mf_emb.mo_coeff.copy()
    c, s = np.cos(0.3), np.sin(0.3)
    i, j = np.where(occ)[0][:2]
    rotated.mo_coeff[:, [i, j]] = mf_emb.mo_coeff[:, [i, j]] @ np.array([[c, -s], [s, c]])
    with pytest.raises(NotImplementedError, match="canonical"):
        rdm_embedding_gradient(emb, rotated, r.idx, r.d1, r.d2, proj_type="huz")
