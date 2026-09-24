"""Numeric snapshots of the embedding pipeline for small molecules.

Regenerate with ``pytest tests/test_snapshots.py --snapshot-update`` after an
intentional change, and review the diff of ``tests/snapshots``.
"""

import numpy as np
import pytest
from conftest import EMBEDDABLE, MOLECULES, embedded_run, embedding, global_ks

from nbed.act_env_space import select_act_env_space
from nbed.cas_space import cas_columns_excluding_env
from nbed.hamiltonian import build_molecular_H, build_spin_integrals
from openfermion import count_qubits

BASES = ["sto-3g", "6-31g"]


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_global_and_partition(snapshot, name, basis):
    """Global KS energy and the chosen fragment orbitals."""
    mf = global_ks(name, basis)
    act, env, pop, _ = select_act_env_space(mf, MOLECULES[name][1], 1, 1)
    snapshot({
        "e_tot": mf.e_tot,
        "active_idxs": act,
        "n_env": len(env),
        "population_active": pop[act],
    })


@pytest.mark.parametrize("proj", ["mu", "huz"])
@pytest.mark.parametrize("method", ["dft", "hf"])
@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("name", EMBEDDABLE)
def test_embedded_energies(snapshot, name, basis, method, proj):
    """Embedded mean-field energies and their parts."""
    emb = embedding(name, basis)
    run = embedded_run(name, method, proj, basis)
    snapshot({
        "e_emb": run.e_tot,
        "e_subsystem": run.mf.e_tot,
        "emb_corr": run.emb_corr,
        "env_plus_corrections": run.env_plus_corrections,
        "E_act": emb.E_act,
        "E_env": emb.E_env,
        "E_cross": emb.E_cross,
        "nelec": list(emb.mol_act.nelec),
        "n_env_cols": len(run.env_cols),
    }, rtol=0, atol=1e-6)


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("name", ["water", "methanol", "formamide", "acetonitrile"])
def test_cas_hamiltonian(snapshot, name, basis):
    """Qubit Hamiltonian of a (4e, 4o) CAS in the embedded fragment."""
    emb = embedding(name, basis)
    run = embedded_run(name, "hf", "huz", basis)
    mf = run.mf
    idx, nelecas, _, _ = cas_columns_excluding_env(mf.mo_occ, run.env_cols, 2, 2)
    e_core, h1, eri = emb.get_mo_integrals(mf, mf.mo_coeff, len(idx), nelecas,
                                           mo_cas_idxs=idx)
    h1_spin, eri_spin = build_spin_integrals(h1, eri, len(idx))
    H = build_molecular_H(e_core, h1_spin, eri_spin)
    coeffs = np.array(sorted(abs(c) for c in H.terms.values()))
    snapshot({
        "e_core": e_core,
        "n_qubits": count_qubits(H),
        "n_terms": len(H.terms),
        "identity": H.terms[()].real,
        # orbital phases are arbitrary, so only sign-free summaries are stable
        "abs_coeff_sum": coeffs.sum(),
        "abs_coeff_max": coeffs.max(),
    }, rtol=0, atol=1e-6)
