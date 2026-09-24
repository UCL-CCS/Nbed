"""Analytic embedding gradients against finite differences of the energy."""

import numpy as np
import pytest
from conftest import build_mol, embedded_run, embedding
from hypothesis import given, settings
from hypothesis import strategies as st
from pyscf import dft, gto, lo

from nbed.act_env_space import select_act_env_space
from nbed.emb_scf import EmbedSCF
from nbed.grad import (
    elec_energy_deriv,
    embedding_gradient,
    fock_deriv,
    fock_deriv_contract,
)

WATER = build_mol("water").atom_coords()
METHANOL = build_mol("methanol").atom_coords()


class EmbeddedEnergy:
    """E_emb(R) with the partition held fixed, for finite differences."""

    def __init__(self, name, xc, method, basis="sto-3g", atoms=(0,), n_occ=2, n_vir=1,
                 mu=1e4):
        mol = build_mol(name, basis)
        self.symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
        self.xc, self.method, self.basis, self.mu = xc, method, basis, mu
        self.atoms, self.n_occ, self.n_vir = list(atoms), n_occ, n_vir
        self.act = None

    def run(self, coords):
        mol = gto.M(atom=list(zip(self.symbols, coords)), unit="Bohr", basis=self.basis,
                    verbose=0)
        mf = dft.RKS(mol, xc=self.xc)
        mf.conv_tol = 1e-12
        mf.kernel()
        act, env, _, _ = select_act_env_space(mf, self.atoms, self.n_occ, self.n_vir)
        if self.act is None:
            self.act = act
        # the finite difference is only meaningful for an unchanged partition
        assert np.array_equal(act, self.act)
        emb = EmbedSCF(mf, act, env, mf.mo_coeff, mf.mo_occ, mf.get_ovlp(), 4000,
                       mu_val=self.mu)
        if self.method == "hf":
            e, mf_emb, *_ = emb.build_emb_hf(warn=False)
        else:
            e, mf_emb, *_ = emb.build_emb_dft(self.xc, warn=False)
        return e, emb, mf_emb

    def finite_difference(self, coords, i, x, h=1e-4):
        """Central difference of E_emb along one Cartesian coordinate."""
        plus, minus = coords.copy(), coords.copy()
        plus[i, x] += h
        minus[i, x] -= h
        return (self.run(plus)[0] - self.run(minus)[0]) / (2 * h)


@pytest.mark.parametrize("xc, method", [
    ("hf", "hf"),       # no xc at all
    ("lda", "dft"),     # LDA-in-LDA
    ("pbe", "hf"),      # GGA kernel
    ("b3lyp", "hf"),    # hybrid, HF-in-DFT
    ("b3lyp", "dft"),   # hybrid DFT-in-DFT
])
def test_gradient_matches_finite_difference(xc, method):
    """Symmetry-unique components agree with a central difference of the energy.

    Water lies in the yz plane with the hydrogens mirror images of each other, so O z
    and H1 y, z determine the rest; those follow from the sum rule and the symmetry.
    """
    energy = EmbeddedEnergy("water", xc, method)
    _, emb, mf_emb = energy.run(WATER)
    grad = embedding_gradient(emb, mf_emb)
    for i, x in [(0, 2), (1, 1), (1, 2)]:
        assert abs(grad[i, x] - energy.finite_difference(WATER, i, x)) < 1e-6
    assert np.abs(grad[:, 0]).max() < 1e-9
    assert np.allclose(grad[1] * [1, -1, 1], grad[2], atol=1e-9)
    assert np.abs(grad.sum(axis=0)).max() < 1e-9


def test_gradient_larger_fragment_and_basis():
    """A distorted methanol in 6-31G with a two-atom fragment."""
    coords = METHANOL + np.random.default_rng(3).normal(scale=0.05, size=METHANOL.shape)
    energy = EmbeddedEnergy("methanol", "b3lyp", "hf", basis="6-31g", atoms=(0, 1),
                            n_occ=2, n_vir=2)
    _, emb, mf_emb = energy.run(coords)
    grad = embedding_gradient(emb, mf_emb)
    for i, x in [(0, 0), (1, 1), (2, 0), (5, 2)]:
        assert abs(grad[i, x] - energy.finite_difference(coords, i, x)) < 1e-6


@settings(max_examples=5)
@given(st.tuples(*[st.floats(-2, 2)] * 3))
def test_gradient_is_translation_invariant(shift):
    """The forces sum to zero, and are the same once the molecule is translated."""
    energy = EmbeddedEnergy("water", "b3lyp", "hf")
    _, emb, mf_emb = energy.run(WATER)
    grad = embedding_gradient(emb, mf_emb)
    _, emb_t, mf_emb_t = energy.run(WATER + np.array(shift))
    grad_t = embedding_gradient(emb_t, mf_emb_t)
    assert np.abs(grad.sum(axis=0)).max() < 1e-9
    assert np.abs(grad - grad_t).max() < 1e-7


def test_grid_response_matters_for_hybrids():
    """Without grid-weight response the B3LYP fragment terms break the sum rule."""
    energy = EmbeddedEnergy("water", "b3lyp", "hf")
    _, emb, mf_emb = energy.run(WATER)
    grad, terms = embedding_gradient(emb, mf_emb, xc_grid_response=False,
                                     return_terms=True)
    assert np.abs(grad.sum(axis=0)).max() > 1e-4
    assert np.abs(terms["-E[gamma_A]"].sum(axis=0)).max() > 1e-4


def test_fock_contraction_routes_agree_without_xc():
    """For pure HF the analytic make_h1 route and the density-space route coincide."""
    mf = dft.RKS(build_mol("water"), xc="hf").run()
    t = np.random.default_rng(0).normal(size=(mf.mol.nao,) * 2)
    t = t + t.T
    analytic = np.einsum("kxij,ij->kx", fock_deriv(mf, mf.mo_coeff, mf.mo_occ), t)
    numeric = fock_deriv_contract(mf.Gradients(), mf.make_rdm1(), t)
    assert np.allclose(analytic, numeric, atol=1e-9)


def test_elec_energy_deriv_is_pyscf_gradient():
    """For the SCF density, adding -tr(W S^x) and E_nuc^x gives PySCF's gradient."""
    from pyscf.grad.rhf import grad_nuc

    from nbed.grad import ovlp_deriv_contract

    mf = dft.RKS(build_mol("water"), xc="b3lyp").run(conv_tol=1e-12)
    g = mf.Gradients()
    occ = mf.mo_occ > 0
    w = 2 * (mf.mo_coeff[:, occ] * mf.mo_energy[occ]) @ mf.mo_coeff[:, occ].T
    ours = elec_energy_deriv(g, mf.make_rdm1()) - ovlp_deriv_contract(mf.mol, w)
    assert np.allclose(ours + grad_nuc(mf.mol), g.kernel(), atol=1e-10)


def test_unsupported_cases():
    """Huzinaga, open shells and localised partitions are refused."""
    emb = embedding("water")
    with pytest.raises(NotImplementedError, match="mu"):
        embedding_gradient(emb, embedded_run("water", "hf", "huz").mf, proj_type="huz")
    with pytest.raises(NotImplementedError, match="closed"):
        embedding_gradient(embedding("methyl"), embedded_run("methyl", "hf", "mu").mf)

    mf = emb.global_scf_obj
    occ = mf.mo_occ > 0
    c_loc = mf.mo_coeff.copy()
    c_loc[:, occ] = lo.PipekMezey(mf.mol, mf.mo_coeff[:, occ]).kernel()
    act, env, _, _ = select_act_env_space(mf, [0], 2, 1, mo_coeff=c_loc)
    emb_loc = EmbedSCF(mf, act, env, c_loc, mf.mo_occ, mf.get_ovlp(), 4000, mu_val=1e4)
    _, mf_emb, *_ = emb_loc.build_emb_hf(warn=False)
    with pytest.raises(NotImplementedError, match="canonical"):
        embedding_gradient(emb_loc, mf_emb)
