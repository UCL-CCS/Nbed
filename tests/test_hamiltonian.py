"""Tests for spin-orbital integrals and OpenFermion Hamiltonians."""

import numpy as np
import pytest
from conftest import embedded_run, embedding
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from openfermion import (
    FermionOperator,
    InteractionOperator,
    QubitOperator,
    count_qubits,
    get_sparse_operator,
    jw_get_ground_state_at_particle_number,
    normal_ordered,
)
from pyscf import ao2mo

from nbed.cas_space import cas_columns_excluding_env, run_casci
from nbed.hamiltonian import (
    build_molecular_H,
    build_number_operator,
    build_spin_integrals,
    qubit_op_to_dict,
)


@st.composite
def spatial_integrals(draw, max_orb=3):
    """Random real symmetric h and an 8-fold symmetric chemist-order eri."""
    n = draw(st.integers(1, max_orb))
    h = draw(arrays(np.float64, (n, n), elements=st.floats(-1, 1)))
    g = draw(arrays(np.float64, (n,) * 4, elements=st.floats(-1, 1)))
    h = h + h.T
    g = g + g.transpose(1, 0, 2, 3)
    g = g + g.transpose(0, 1, 3, 2)
    g = g + g.transpose(2, 3, 0, 1)
    return n, h, g


@given(spatial_integrals())
def test_spin_integral_symmetries(case):
    """Spin integrals are block diagonal in spin and keep the physicist symmetries."""
    n, h, g = case
    h1, g2 = build_spin_integrals(h, ao2mo.restore(4, g, n), n)
    assert h1.shape == (2 * n, 2 * n)
    assert np.allclose(h1[::2, ::2], h) and np.allclose(h1[1::2, 1::2], h)
    assert np.allclose(h1[::2, 1::2], 0)

    assert np.allclose(g2, g2.transpose(1, 0, 3, 2))
    assert np.allclose(g2, g2.transpose(3, 2, 1, 0))
    # an integral vanishes unless both electrons keep their spin
    spin = np.arange(2 * n) % 2
    p, q, r, s = np.meshgrid(spin, spin, spin, spin, indexing="ij")
    assert np.allclose(g2[(p != s) | (q != r)], 0)
    # alpha-alpha block: <pq|rs> with p,s on electron 1 is (ps|qr)
    assert np.allclose(g2[::2, ::2, ::2, ::2], g.transpose(0, 2, 3, 1))


@settings(max_examples=20)
@given(spatial_integrals(max_orb=2), st.floats(-5, 5))
def test_hamiltonian_is_hermitian_and_conserves_particles(case, e_core):
    """The JW Hamiltonian is Hermitian and commutes with the spin number operators."""
    n, h, g = case
    h1, g2 = build_spin_integrals(h, ao2mo.restore(4, g, n), n)
    H_f = build_molecular_H(e_core, h1, g2, return_type="fermion")
    H_q = build_molecular_H(e_core, h1, g2, return_type="jordan_wigner")
    H_i = build_molecular_H(e_core, h1, g2, return_type="other")
    assert isinstance(H_f, FermionOperator)
    assert isinstance(H_q, QubitOperator)
    assert isinstance(H_i, InteractionOperator)

    mat = get_sparse_operator(H_q, n_qubits=2 * n).toarray()
    assert np.allclose(mat, mat.conj().T)
    for N in build_number_operator(2 * n):
        comm = normal_ordered(H_f * N - N * H_f)
        assert comm.induced_norm() < 1e-8


@pytest.mark.parametrize("name", ["water", "methanol"])
def test_ground_state_matches_casci(name):
    """The lowest eigenvalue in the right particle sector is the CASCI energy."""
    emb = embedding(name)
    run = embedded_run(name, "hf", "huz")
    mf = run.mf
    idx, nelecas, _, _ = cas_columns_excluding_env(mf.mo_occ, run.env_cols, 2, 2)
    norb = len(idx)
    e_core, h1, eri = emb.get_mo_integrals(mf, mf.mo_coeff, norb, nelecas, mo_cas_idxs=idx)
    h1_spin, eri_spin = build_spin_integrals(h1, eri, norb)
    H = build_molecular_H(e_core, h1_spin, eri_spin)
    assert count_qubits(H) == 2 * norb

    e_qubit, _ = jw_get_ground_state_at_particle_number(
        get_sparse_operator(H, n_qubits=2 * norb), sum(nelecas))
    casci = run_casci(mf, norb, nelecas, mo_cas_idxs=idx)
    assert np.isclose(e_qubit, casci.e_tot)


@given(st.integers(1, 4))
def test_number_operators(n_orb):
    """Fermionic and JW number operators agree and count alpha and beta separately."""
    n_qubits = 2 * n_orb
    Na_f, Nb_f = build_number_operator(n_qubits, type="fermion")
    Na_q, Nb_q = build_number_operator(n_qubits, type="qubit_jw")
    from openfermion import jordan_wigner
    assert jordan_wigner(Na_f) == Na_q
    assert jordan_wigner(Nb_f) == Nb_q
    total = get_sparse_operator(Na_q + Nb_q, n_qubits=n_qubits).diagonal().real
    assert total.max() == n_qubits


def test_number_operator_bad_type():
    """Unknown encodings are rejected."""
    with pytest.raises(ValueError):
        build_number_operator(4, type="bravyi_kitaev")


@given(st.dictionaries(
    st.lists(st.tuples(st.integers(0, 5), st.sampled_from("XYZ")),
             unique_by=lambda t: t[0], max_size=4).map(lambda t: tuple(sorted(t))),
    st.floats(-10, 10).filter(lambda x: abs(x) > 1e-6),
    min_size=1, max_size=6))
def test_qubit_op_to_dict_roundtrip(terms):
    """Pauli strings in the dict rebuild the same operator."""
    op = QubitOperator()
    for term, coeff in terms.items():
        op += QubitOperator(term, coeff)
    n = count_qubits(op)
    out = qubit_op_to_dict(op)
    rebuilt = QubitOperator()
    for string, coeff in out.items():
        assert len(string) == n
        rebuilt += QubitOperator(
            tuple((i, c) for i, c in enumerate(string) if c != "I"), coeff)
    assert rebuilt == op
