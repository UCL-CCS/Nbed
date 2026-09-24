"""Analytic nuclear gradients of the projection-based embedding energy (prototype).

Follows the Lagrangian route of Lee, Welborn, Manby & Miller, J. Chem. Phys. 151,
074104 (2019), specialised to what :class:`nbed.emb_scf.EmbedSCF` computes::

    E_emb = E_sub[D; h + v_emb] + E[gamma] - E[gamma_A] - tr(gamma_A v_emb)
    v_emb = G[gamma] - G[gamma_A] + (mu / 2) S gamma_B S

where ``gamma = gamma_A + gamma_B`` is the global Kohn-Sham density split into the
fragment and environment, ``G`` is the global KS two-electron potential and ``D`` the
converged embedded density.

The embedded SCF is variational, so ``D`` only brings an energy-weighted density. The
global orbitals are not variational for ``E_emb``, so two sets of multipliers are
needed:

* ``z`` (virtual-occupied), from the stationarity of the global KS energy. It is solved
  as a Z-vector (CPKS) problem.
* ``x`` (fragment-environment occupied pairs), from the condition that picks the
  partition. For canonical orbitals this is ``F_ab = 0``, and ``x`` follows in closed
  form from ``eps_a - eps_b``.

The nuclear-derivative integrals come from PySCF: ``grad.*.get_veff`` and
``hcore_generator`` for energies at fixed density (with grid-weight response), and
either :func:`fock_deriv_contract` or ``hessian.*.make_h1`` for Fock derivatives at
fixed density.

Scope of this prototype:

* closed-shell global RKS (``xc="hf"`` is allowed) partitioned by
  ``select_act_env_space`` on **canonical** orbitals;
* the **mu** projector;
* an embedded RHF (HF-in-DFT) or RKS (DFT-in-DFT) solution;
* CPU PySCF objects only.

Out of scope: Huzinaga (its projector depends on the embedded Fock, which adds another
response), open shells, localised (e.g. Pipek-Mezey) partitions, which need the
localisation Hessian, and correlated wavefunctions in the fragment.
"""

from dataclasses import dataclass

import numpy as np
from pyscf.grad.rhf import grad_nuc
from pyscf.scf import cphf, hf


def _sym(a):
    return 0.5 * (a + a.T)


def elec_energy_deriv(mf_grad, dm):
    """Nuclear derivative of the electronic energy at a *fixed* AO density.

    Covers ``tr(dm h) + E_2e[dm]`` (Coulomb, exchange and xc) as they depend on the
    AO integrals, plus the grid-weight response when ``mf_grad.grid_response`` is
    set. The ``-tr(W S^x)`` term that a normal gradient carries is *not* included.

    Args:
        mf_grad: Gradients object of the mean field whose functional is wanted.
        dm: ``(nao, nao)`` spin-summed density matrix.

    Returns:
        ``(natm, 3)`` array.
    """
    mol = mf_grad.mol
    hcore_deriv = mf_grad.hcore_generator(mol)
    vhf = mf_grad.get_veff(mol, dm)
    exc1_grid = getattr(vhf, "exc1_grid", None)
    de = np.zeros((mol.natm, 3))
    for k, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        de[k] = np.einsum("xij,ij->x", hcore_deriv(k), dm)
        de[k] += 2 * np.einsum("xij,ij->x", vhf[:, p0:p1], dm[p0:p1])
        if getattr(mf_grad, "grid_response", False) and exc1_grid is not None:
            de[k] += exc1_grid[k]
    return de


def fock_deriv_contract(mf_grad, dm, t, eps=1e-4):
    """``tr(t dF[dm]/dR)`` at fixed ``dm`` and ``t``, including grid response.

    Uses ``dF/d(dm) = F``: the Fock derivative contracted with ``t`` is the
    directional derivative, along ``t``, of the fixed-density energy gradient. It
    is taken by a central difference in *density* space, which is exact for the
    one- and two-electron integrals and O(eps^2) for the xc energy. No geometry is
    displaced. Unlike ``hessian.*.make_h1`` it includes the DFT grid-weight
    response, which is large for the partial fragment density.

    Args:
        mf_grad: Gradients object of the mean field whose functional is wanted.
        dm: ``(nao, nao)`` density at which the Fock matrix is taken.
        t: ``(nao, nao)`` symmetric matrix to contract with.
        eps: Step, relative to the largest element of ``t``.

    Returns:
        ``(natm, 3)`` array.
    """
    scale = np.abs(t).max()
    if scale == 0:
        return np.zeros((mf_grad.mol.natm, 3))
    step = eps / scale
    plus = elec_energy_deriv(mf_grad, dm + step * t)
    minus = elec_energy_deriv(mf_grad, dm - step * t)
    return (plus - minus) / (2 * step)


def ovlp_deriv_contract(mol, w):
    """``tr(S^x w)`` for every nuclear coordinate, for a symmetric AO matrix ``w``."""
    s1 = -mol.intor("int1e_ipovlp", comp=3)
    de = np.zeros((mol.natm, 3))
    for k, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        de[k] = 2 * np.einsum("xij,ij->x", s1[:, p0:p1], w[p0:p1])
    return de


def fock_deriv(mf, mo_coeff, mo_occ):
    """AO Fock derivatives ``dF[rho]/dR`` at fixed ``rho = sum occ c c^T``.

    Args:
        mf: Mean field supplying the functional and grids.
        mo_coeff: Orbitals defining the fixed density.
        mo_occ: Their occupations.

    Returns:
        ``(natm, 3, nao, nao)`` array.
    """
    h1 = mf.Hessian().make_h1(mo_coeff, mo_occ)
    return np.asarray(list(h1))


def _contract_atoms(h1ao, a):
    return np.einsum("kxij,ij->kx", h1ao, a)


def _canonical_blocks(emb, mf_global, tol):
    """Fragment/environment/virtual orbitals of the global mean field, and energies."""
    if not isinstance(mf_global, hf.RHF) or emb.SCF_type != "closed-shell":
        raise NotImplementedError("gradients are only implemented for closed shells")
    C = emb.C_full_reidx
    occ_A = np.asarray(emb.mo_occ_act).nonzero()[0]
    occ_B = np.asarray(emb.env_idx_occ)
    vir = np.asarray(emb.mo_occ_full_reidx == 0).nonzero()[0]
    C_A, C_B, C_v = C[:, occ_A], C[:, occ_B], C[:, vir]
    C_o = np.hstack([C_A, C_B])

    fock = mf_global.get_fock(dm=emb.dm_full)
    f_oo = C_o.T @ fock @ C_o
    if np.abs(f_oo - np.diag(np.diag(f_oo))).max() > tol:
        raise NotImplementedError(
            "the partitioned occupied orbitals are not canonical; gradients for "
            "localised partitions need the localisation response"
        )
    f_vv = C_v.T @ fock @ C_v
    return C_A, C_B, C_v, np.diag(f_oo), np.diag(f_vv), fock


@dataclass
class FragmentResponse:
    """What the fragment solver contributes to the embedding gradient.

    Attributes:
        grad_fixed_vemb: ``(natm, 3)`` nuclear gradient of the fragment energy with the
            AO embedding potential held fixed, nuclear repulsion and ``S^x`` terms
            included.
        dm_relaxed: ``(nao, nao)`` relaxed fragment density, i.e. the derivative of the
            fragment energy with respect to its one-electron Hamiltonian.
        env_orbital_grad: Optional ``(nao, n_env_occ)`` derivative of the fragment
            energy with respect to the environment orbital coefficients at fixed
            density (from the huzinaga orthogonality constraint).
        w_extra: Optional symmetric AO matrix contracted with ``S^x``.
    """

    grad_fixed_vemb: np.ndarray
    dm_relaxed: np.ndarray
    env_orbital_grad: np.ndarray = None
    w_extra: np.ndarray = None


def _grid(grad_obj, on):
    if hasattr(grad_obj, "grid_response"):
        grad_obj.grid_response = on
    return grad_obj


def _huzinaga_constraint(emb, fock, c_occ, tol):
    """Multipliers of the huzinaga orthogonality constraint ``c_occ^T S C_B = 0``.

    At convergence the huzinaga equations are exactly the stationarity conditions of
    the projector-free energy under that constraint, with multipliers
    ``Lambda = 4 c_occ^T F C_B``. The constraint then contributes
    ``-Lambda`` through the environment orbitals and through ``S^x``.
    """
    C_B = emb.C_full_reidx[:, emb.env_idx_occ]
    S = emb.Sao
    overlap = np.abs(c_occ.T @ S @ C_B).max() if C_B.size else 0.0
    if overlap > tol:
        raise ValueError(
            f"embedded occupied orbitals overlap the environment by {overlap:.1e}; the "
            "huzinaga gradient needs a converged, uncontaminated embedding"
        )
    lam = 4 * c_occ.T @ fock @ C_B
    env_orbital_grad = -S @ c_occ @ lam
    w_extra = -_sym(c_occ @ lam @ C_B.T)
    return env_orbital_grad, w_extra


def mean_field_response(emb, mf_emb, proj_type="mu", xc_grid_response=True,
                        orthogonality_tol=1e-7):
    """:class:`FragmentResponse` of an embedded HF or KS solution.

    Args:
        emb: The :class:`~nbed.emb_scf.EmbedSCF` that produced ``mf_emb``.
        mf_emb: Converged embedded RHF or RKS from ``build_emb_*``.
        proj_type: ``"mu"`` or ``"huz"``, as passed to ``build_emb_*``.
        xc_grid_response: Include the DFT grid-weight response.
        orthogonality_tol: Largest tolerated overlap with the environment (huzinaga).

    Returns:
        A :class:`FragmentResponse`.
    """
    mol = mf_emb.mol
    D = mf_emb.make_rdm1()
    occ = mf_emb.mo_occ > 0
    c_occ = mf_emb.mo_coeff[:, occ]
    w_emb = 2 * (c_occ * mf_emb.mo_energy[occ]) @ c_occ.T
    grad = (elec_energy_deriv(_grid(mf_emb.Gradients(), xc_grid_response), D)
            + grad_nuc(mol) - ovlp_deriv_contract(mol, w_emb))
    frag = FragmentResponse(grad, D)
    if proj_type == "huz":
        # the huzinaga get_fock is overridden; the energy uses the plain one
        fock = mf_emb.get_hcore() + mf_emb.get_veff(dm=D)
        frag.env_orbital_grad, frag.w_extra = _huzinaga_constraint(
            emb, fock, c_occ, orthogonality_tol)
    return frag


def global_embedding_gradient(emb, frag, mf_global=None, proj_type="mu",
                              canonical_tol=1e-6, degeneracy_tol=1e-4, cphf_tol=1e-10,
                              xc_grid_response=True, fd_eps=1e-4, return_terms=False):
    """Embedding gradient for any fragment solver described by a FragmentResponse.

    Adds the global Kohn-Sham response, the canonical-partition multipliers and the
    embedding-potential derivatives to ``frag``. See :func:`embedding_gradient` for
    the arguments.
    """
    if proj_type not in ("mu", "huz"):
        raise NotImplementedError(f"unknown projector {proj_type!r}")
    mf_global = emb.global_scf_obj if mf_global is None else mf_global
    mol = mf_global.mol
    # huzinaga keeps its projector out of the energy, so no mu terms
    mu = emb.mu_val if proj_type == "mu" else 0.0
    S = emb.Sao

    C_A, C_B, C_v, e_o, e_v, fock = _canonical_blocks(emb, mf_global, canonical_tol)
    nA, nB, nv = C_A.shape[1], C_B.shape[1], C_v.shape[1]
    no = nA + nB
    C_o = np.hstack([C_A, C_B])
    C = np.hstack([C_o, C_v])
    e_A, e_B = e_o[:nA], e_o[nA:]

    gamma = emb.dm_full
    gamma_A = emb.dm_act
    gamma_B = emb.dm_env
    dD = frag.dm_relaxed - gamma_A

    # fixed-density orbitals for the global functional at gamma and at gamma_A
    mo_occ = np.array([2.0] * no + [0.0] * nv)
    mo_occ_A = np.array([2.0] * nA + [0.0] * (nB + nv))
    resp = mf_global.gen_response(mo_coeff=C, mo_occ=mo_occ, hermi=1)
    resp_A = mf_global.gen_response(mo_coeff=C, mo_occ=mo_occ_A, hermi=1)

    # dE/dgamma_A and dE/dgamma_B as symmetric AO matrices; the J/K parts of the two
    # response kernels cancel, which leaves the xc kernel difference only
    resp_dD = resp(dD)
    M_A = -0.5 * mu * S @ gamma_B @ S + resp_dD - resp_A(dD)
    M_B = fock + resp_dD + 0.5 * mu * S @ dD @ S

    # orbital gradient Y_pi = dE/dU_pi, columns in [A | B] order
    Y = np.hstack([4 * C.T @ M_A @ C_A, 4 * C.T @ M_B @ C_B])
    if frag.env_orbital_grad is not None:
        Y[:, nA:] += C.T @ frag.env_orbital_grad

    # fragment-environment multipliers from the canonical condition F_ab = 0
    gap = e_A[:, None] - e_B[None, :]
    if gap.size and np.abs(gap).min() < degeneracy_tol:
        raise ValueError(
            f"a fragment and an environment orbital are within {np.abs(gap).min():.1e} "
            "Ha of each other, so the canonical partition is not differentiable"
        )
    Y_ab = Y[:nA, nA:no]  # row a (fragment), column b (environment)
    Y_ba = Y[nA:no, :nA]  # row b, column a
    x = (Y_ab - Y_ba.T) / gap
    X_ao = _sym(C_A @ x @ C_B.T)

    # Z-vector for the global KS stationarity
    resp_X = resp(X_ao)
    L = Y[no:, :] - 4 * C_v.T @ resp_X @ C_o

    def fvind(z):
        z = z.reshape(nv, no)
        dm = C_v @ z @ C_o.T
        return 2 * (C_v.T @ resp(dm + dm.T) @ C_o).ravel()

    z = cphf.solve(fvind, np.concatenate([e_o, e_v]), mo_occ, -L,
                   max_cycle=100, tol=cphf_tol)[0].reshape(nv, no)
    Z_ao = _sym(C_v @ z @ C_o.T)

    # everything that multiplies S^x, in the MO basis [A | B | v]
    nmo = no + nv
    omega = np.zeros((nmo, nmo))
    omega[:nA, :nA] = -0.5 * Y[:nA, :nA]
    omega[nA:no, nA:no] = -0.5 * Y[nA:no, nA:no]
    omega[:nA, nA:no] = x * e_B[None, :] - Y_ba.T
    omega[no:, :no] = z * e_o[None, :]
    omega[:no, :no] += 2 * C_o.T @ (resp_X + resp(Z_ao)) @ C_o
    w_ao = _sym(C @ omega @ C.T)

    # the mu projector's explicit S dependence
    w_ao += mu * _sym(gamma_B @ S @ dD)
    if frag.w_extra is not None:
        w_ao += frag.w_extra

    t_fock = -X_ao - Z_ao
    g_glob = _grid(mf_global.Gradients(), xc_grid_response)

    terms = {
        "E[gamma]": elec_energy_deriv(g_glob, gamma),
        "-E[gamma_A]": -elec_energy_deriv(g_glob, gamma_A),
        "fragment": frag.grad_fixed_vemb,
        "S^x global": ovlp_deriv_contract(mol, w_ao),
    }
    if xc_grid_response:
        terms["tr(dD v_emb^x)"] = (fock_deriv_contract(g_glob, gamma, dD, fd_eps)
                                   - fock_deriv_contract(g_glob, gamma_A, dD, fd_eps))
        terms["F^x response"] = fock_deriv_contract(g_glob, gamma, t_fock, fd_eps)
    else:
        h1_full = fock_deriv(mf_global, C, mo_occ)
        h1_A = fock_deriv(mf_global, C, mo_occ_A)
        terms["tr(dD v_emb^x)"] = _contract_atoms(h1_full - h1_A, dD)
        terms["F^x response"] = _contract_atoms(h1_full, t_fock)
    de = sum(terms.values())
    if return_terms:
        return de, terms
    return de


def embedding_gradient(emb, mf_emb, mf_global=None, proj_type="mu",
                       canonical_tol=1e-6, degeneracy_tol=1e-4, cphf_tol=1e-10,
                       xc_grid_response=True, fd_eps=1e-4, return_terms=False):
    """Nuclear gradient of the energy returned by ``build_emb_hf`` / ``build_emb_dft``.

    Args:
        emb: The :class:`~nbed.emb_scf.EmbedSCF` that produced ``mf_emb``. It must have
            been built from the canonical orbitals of ``mf_global``.
        mf_emb: Converged embedded mean field (RHF or RKS) from ``build_emb_*``.
        mf_global: Converged global RKS. Defaults to ``emb.global_scf_obj``.
        proj_type: ``"mu"`` or ``"huz"``, matching the embedded calculation.
        canonical_tol: Largest off-diagonal occupied Fock element accepted as
            canonical.
        degeneracy_tol: Smallest fragment-environment orbital energy gap. Near
            degeneracy makes the partition itself non-differentiable.
        cphf_tol: Convergence threshold of the Z-vector solve.
        xc_grid_response: Include the DFT grid-weight response. When ``False``,
            Fock derivatives come from ``hessian.*.make_h1`` (fully analytic, but
            with no grid response, which gives errors of 1e-4 to 1e-2 Ha/bohr for
            GGA and hybrid functionals because of the fragment density).
        fd_eps: Density-space step for :func:`fock_deriv_contract`.
        return_terms: Also return a dict of the separate contributions.

    Returns:
        ``(natm, 3)`` gradient in hartree/bohr.

    Raises:
        NotImplementedError: For open shells, non-canonical partitions or unknown
            projectors.
        ValueError: If a fragment and an environment orbital are (near) degenerate,
            or a huzinaga embedding is not orthogonal to the environment.
    """
    if proj_type not in ("mu", "huz"):
        raise NotImplementedError(f"unknown projector {proj_type!r}")
    if emb.SCF_type != "closed-shell":
        raise NotImplementedError("gradients are only implemented for closed shells")
    frag = mean_field_response(emb, mf_emb, proj_type, xc_grid_response)
    return global_embedding_gradient(
        emb, frag, mf_global, proj_type, canonical_tol, degeneracy_tol, cphf_tol,
        xc_grid_response, fd_eps, return_terms)
