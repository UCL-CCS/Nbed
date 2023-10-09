"""Functions from PySCF that need to be tweeked to allow our hack of adding Vemb to Hcore."""

from typing import Tuple

import numpy as np
from pyscf import ao2mo


def energy_elec(mf, dm=None, h1e=None, vhf=None) -> Tuple[float, float]:
    """Electronic energy of Unrestricted Hartree-Fock.

    Note this function has side effects which cause mf.scf_summary updated.
    This version is

    Args:
        mf (pyscf.scf.hf.HF): Hartree-Fock object
        dm (np.ndarray): Density matrix
        h1e (np.ndarray): Core Hamiltonian
        vhf (np.ndarray): 2-electron contribution to effective potential

    Returns:
        e_elec (np.ndarray): Hartree-Fock electronic energy
        e_coul (np.ndarray): 2-electron contribution to electronic energy
    """
    if dm is None:
        dm = mf.make_rdm1()
    if h1e is None:
        h1e = mf.get_hcore()
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = np.array((dm * 0.5, dm * 0.5))
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)
    e1 = np.einsum("ij,ji->", h1e[0], dm[0])
    e1 += np.einsum("ij,ji->", h1e[1], dm[1])
    e_coul = (
        np.einsum("ij,ji->", vhf[0], dm[0]) + np.einsum("ij,ji->", vhf[1], dm[1])
    ) * 0.5
    e_elec = (e1 + e_coul).real
    mf.scf_summary["e1"] = e1.real
    mf.scf_summary["e2"] = e_coul.real
    return e_elec, e_coul


def _absorb_h1e(h1e, eri, norb, nelec, fac=1):
    if not isinstance(nelec, (int, np.number)):
        nelec = sum(nelec)
    h1e_a, h1e_b = h1e
    print("h1e_a", h1e_a.shape)
    print("h1e_b", h1e_b.shape)
    h1e_a = np.einsum("jik->jk", h1e_a)
    h1e_b = np.einsum("jik->jk", h1e_b)
    print("h1e_a", h1e_a.shape)
    print("h1e_b", h1e_b.shape)
    h2e_aa = ao2mo.restore(1, eri[0], norb).copy()
    h2e_ab = ao2mo.restore(1, eri[1], norb).copy()
    h2e_bb = ao2mo.restore(1, eri[2], norb).copy()
    print("x1", h2e_aa.shape)
    print("y2", h2e_bb.shape)
    x = np.einsum("jiik->jk", h2e_aa) * 0.5
    y = np.einsum("jiik->jk", h2e_bb) * 0.5
    print("x", x.shape)
    print("y", y.shape)
    f1e_a = h1e_a - np.einsum("jiik->jk", h2e_aa) * 0.5
    f1e_b = h1e_b - np.einsum("jiik->jk", h2e_bb) * 0.5
    f1e_a *= 1.0 / (nelec + 1e-100)
    f1e_b *= 1.0 / (nelec + 1e-100)
    for k in range(norb):
        h2e_aa[:, :, k, k] += f1e_a
        h2e_aa[k, k, :, :] += f1e_a
        h2e_ab[:, :, k, k] += f1e_a
        h2e_ab[k, k, :, :] += f1e_b
        h2e_bb[:, :, k, k] += f1e_b
        h2e_bb[k, k, :, :] += f1e_b
    return (
        ao2mo.restore(4, h2e_aa, norb) * fac,
        ao2mo.restore(4, h2e_ab, norb) * fac,
        ao2mo.restore(4, h2e_bb, norb) * fac,
    )
