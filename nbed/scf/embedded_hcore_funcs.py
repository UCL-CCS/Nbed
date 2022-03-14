"""Functions from PySCF that need to be tweeked to allow our hack of adding Vemb to Hcore."""

from typing import Tuple

import numpy as np


def energy_elec(mf, dm=None, h1e=None, vhf=None) -> Tuple[float, float]:
    """Electronic energy of Unrestricted Hartree-Fock

    Note this function has side effects which cause mf.scf_summary updated.
    This version is

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
