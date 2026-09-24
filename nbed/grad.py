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
from pyscf import ao2mo
from pyscf.grad.rhf import grad_nuc
from pyscf.scf import hf
from scipy.sparse.linalg import LinearOperator, cg


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


def fock_deriv_contract(mf_grad, dm, t, eps=1e-5):
    """``tr(t dF[dm]/dR)`` at fixed ``dm`` and ``t``, including grid response.

    Uses ``dF/d(dm) = F``: the Fock derivative contracted with ``t`` is the
    directional derivative, along ``t``, of the fixed-density energy gradient. It
    is taken by a central difference in *density* space, which is exact for the
    one- and two-electron integrals and O(eps^4) for the xc energy. No geometry is
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

    def central(h):
        return (elec_energy_deriv(mf_grad, dm + h * t)
                - elec_energy_deriv(mf_grad, dm - h * t))

    # fourth-order stencil: the xc energy of the fragment density is far from
    # quadratic along correlated density directions
    return (8 * central(step) - central(2 * step)) / (12 * step)


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


def _solve_zvector(fvind, e_occ, e_vir, rhs, tol):
    """Solve ``(e_a - e_i) z_ai + fvind(z)_ai = rhs_ai`` for the Z-vector.

    Jacobi-preconditioned conjugate gradients on an absolute residual.
    ``pyscf.scf.cphf.solve`` stalls when the orbital energy differences span many
    orders of magnitude, as they do once a mu projector pushes the environment to
    ``~mu`` hartree, and its z then carries errors of 1e-6 Ha/bohr in the gradient.
    """
    diag = (e_vir[:, None] - e_occ[None, :]).ravel()
    b = np.asarray(rhs).ravel()
    op = LinearOperator((b.size, b.size), dtype=float,
                        matvec=lambda z: diag * z + np.asarray(fvind(z)).ravel())
    precond = LinearOperator((b.size, b.size), dtype=float, matvec=lambda r: r / diag)
    atol = tol * max(1.0, np.abs(b).max())
    z, info = cg(op, b, M=precond, rtol=0.0, atol=atol, maxiter=10 * b.size + 100)
    if info != 0:
        raise RuntimeError("the Z-vector equations did not converge")
    return z.reshape(len(e_vir), len(e_occ))


def global_embedding_gradient(emb, frag, mf_global=None, proj_type="mu",
                              canonical_tol=1e-6, degeneracy_tol=1e-4, cphf_tol=1e-10,
                              xc_grid_response=True, fd_eps=1e-5, return_terms=False):
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

    z = _solve_zvector(fvind, e_o, e_v, L, cphf_tol)
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
                       xc_grid_response=True, fd_eps=1e-5, return_terms=False):
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


##################################################################################
############# WF-in-DFT: gradients from externally supplied RDMs #################
##################################################################################


def wf_in_dft_energy(e_core, h1, eri, casdm1, casdm2, env_plus_corrections):
    """WF-in-DFT energy from active-space integrals and RDMs.

    ``(e_core, h1, eri)`` are the output of ``EmbedSCF.get_mo_integrals``, and
    ``env_plus_corrections`` that of ``build_emb_hf``. The RDMs follow PySCF's
    ``make_rdm12`` convention, ``E = e_core + h1 . d + 1/2 (pq|rs) D_pqrs``, and can
    come from any solver, e.g. a VQE measuring the qubit Hamiltonian.
    """
    norb = h1.shape[0]
    g = ao2mo.restore(1, eri, norb)
    return float(e_core + np.einsum("pq,pq", h1, casdm1)
                 + 0.5 * np.einsum("pqrs,pqrs", g, casdm2) + env_plus_corrections)


def _symmetrize_dm2(dm2):
    dm2 = 0.25 * (dm2 + dm2.transpose(1, 0, 2, 3) + dm2.transpose(0, 1, 3, 2)
                  + dm2.transpose(1, 0, 3, 2))
    return 0.5 * (dm2 + dm2.transpose(2, 3, 0, 1))


def _hcore_contract(mf_grad, a):
    hcore_deriv = mf_grad.hcore_generator(mf_grad.mol)
    return np.array([np.einsum("xij,ij->x", hcore_deriv(k), a)
                     for k in range(mf_grad.mol.natm)])


def _dm2_eri_deriv(mol, dm2_ao):
    """``1/2 sum (mn|ls)^x Gamma_mnls`` for a fully symmetric AO 2-RDM."""
    eri1 = mol.intor("int2e_ip1", comp=3).reshape((3,) + (mol.nao,) * 4)
    de = np.zeros((mol.natm, 3))
    for k, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        de[k] = -2 * np.einsum("xmnls,mnls->x", eri1[:, p0:p1], dm2_ao[p0:p1])
    return de


def _env_columns(emb, mf_emb):
    """Embedded columns carrying the occupied environment (weight > 0.5)."""
    C_B = emb.C_full_reidx[:, emb.env_idx_occ]
    proj = mf_emb.mo_coeff.T @ emb.Sao @ C_B
    return np.where(np.einsum("pb,pb->p", proj, proj) > 0.5)[0]


def rdm_fragment_response(emb, mf_emb, mo_cas_idxs, casdm1, casdm2, proj_type="mu",
                          env_cols=None, canonical_tol=1e-6, degeneracy_tol=1e-4,
                          cphf_tol=1e-10, orthogonality_tol=1e-7, fd_eps=1e-5):
    """:class:`FragmentResponse` of an active-space wavefunction given by its RDMs.

    The fragment energy is the CASCI-form energy of the integrals that
    ``EmbedSCF.get_mo_integrals(mf_emb, mf_emb.mo_coeff, ncas, nelecas, mo_cas_idxs)``
    returns, evaluated with the supplied RDMs. The embedded HF orbitals are not
    variational for it, so their response is included: a Z-vector over the
    occupied-virtual rotations and closed-form multipliers for the rotations
    between core and active occupied, and between active virtual and external,
    orbitals, which the canonical condition fixes. With huzinaga, the environment
    columns are excluded from both, since the orthogonality constraint slaves
    their rotations to the global environment orbitals.

    The result is exact when the RDMs come from a state that is stationary within
    the active space (FCI, a converged VQE). Rotations *inside* the active space are
    not given a response.

    Args:
        emb: The :class:`~nbed.emb_scf.EmbedSCF` that produced ``mf_emb``.
        mf_emb: Converged embedded RHF from ``build_emb_hf``.
        mo_cas_idxs: Columns of ``mf_emb.mo_coeff`` forming the active space.
        casdm1: ``(ncas, ncas)`` spin-summed 1-RDM.
        casdm2: ``(ncas,) * 4`` spin-summed 2-RDM, PySCF convention.
        proj_type: ``"mu"`` or ``"huz"``.
        env_cols: Environment columns of ``mf_emb``. Detected when ``None``.
        canonical_tol: Largest off-diagonal Fock element accepted as canonical.
        degeneracy_tol: Smallest orbital energy gap across the space boundaries.
        cphf_tol: Convergence threshold of the Z-vector solve.
        orthogonality_tol: Largest tolerated environment overlap (huzinaga).
        fd_eps: Density-space step for :func:`fock_deriv_contract`.

    Returns:
        A :class:`FragmentResponse`.
    """
    if not isinstance(mf_emb, hf.RHF) or isinstance(mf_emb, hf.KohnShamDFT):
        raise NotImplementedError("WF-in-DFT gradients need an embedded RHF reference")
    mol = mf_emb.mol
    S = emb.Sao
    C_all = mf_emb.mo_coeff
    nmo = C_all.shape[1]
    mo_cas_idxs = np.asarray(mo_cas_idxs, dtype=int)
    casdm2 = _symmetrize_dm2(np.asarray(casdm2))
    casdm1 = 0.5 * (casdm1 + casdm1.T)

    env = np.array([], dtype=int)
    if proj_type == "huz":
        env = _env_columns(emb, mf_emb) if env_cols is None else np.asarray(env_cols)
        if len(np.intersect1d(env, mo_cas_idxs)):
            raise ValueError("the active space contains environment columns")
    comp = np.setdiff1d(np.arange(nmo), env)
    occ_all = np.where(mf_emb.mo_occ > 0)[0]
    if len(np.intersect1d(env, occ_all)):
        raise ValueError("an environment column is occupied in the embedded reference")

    core = np.setdiff1d(occ_all, mo_cas_idxs)
    act_o = np.intersect1d(occ_all, mo_cas_idxs)
    vir = np.setdiff1d(comp, occ_all)
    act_v = np.intersect1d(vir, mo_cas_idxs)
    ext = np.setdiff1d(vir, mo_cas_idxs)
    # active columns in the order the RDMs use
    C_a = C_all[:, mo_cas_idxs]
    C_c = C_all[:, core]

    # projector-free embedded Fock and the orbital energies within the fragment space
    h_emb = mf_emb.get_hcore()
    D_hf = mf_emb.make_rdm1()
    fock = h_emb + mf_emb.get_veff(dm=D_hf)
    f_mo = C_all.T @ fock @ C_all
    f_cc = f_mo[np.ix_(comp, comp)]
    if np.abs(f_cc - np.diag(np.diag(f_cc))).max() > canonical_tol:
        raise NotImplementedError(
            "the embedded orbitals are not canonical; rotated active spaces (e.g. "
            "MP2 natural orbitals) need their own response")
    eps = np.diag(f_mo).copy()
    if proj_type == "huz" and len(env):
        overlap = np.abs(C_all[:, occ_all].T @ S @ emb.C_full_reidx[:, emb.env_idx_occ])
        if overlap.max() > orthogonality_tol:
            raise ValueError(f"embedded occupied orbitals overlap the environment by "
                             f"{overlap.max():.1e}")

    # fragment densities and the derivative of the energy w.r.t. the orbitals
    D_c = 2 * C_c @ C_c.T
    D_a = C_a @ casdm1 @ C_a.T
    D1 = D_c + D_a
    vj, vk = mf_emb.get_jk(mol, np.array([D_c, D_a]), hermi=1)
    G_c, G_a = vj[0] - 0.5 * vk[0], vj[1] - 0.5 * vk[1]
    dE_dC = np.zeros_like(C_all)
    dE_dC[:, core] = 4 * (h_emb + G_c + G_a) @ C_c
    eri = mol.intor("int2e").reshape((mol.nao,) * 4)
    half = np.einsum("mnls,nq,lr,st->mqrt", eri, C_a, C_a, C_a, optimize=True)
    dE_dC[:, mo_cas_idxs] = (2 * (h_emb + G_c) @ C_a @ casdm1
                             + 2 * np.einsum("mqrs,tqrs->mt", half, casdm2))
    G = C_all.T @ dE_dC  # G[p, q] = c_p . dE/dc_q

    resp = mf_emb.gen_response(mo_coeff=C_all, mo_occ=mf_emb.mo_occ, hermi=1)
    omega = np.zeros((nmo, nmo))
    mult = np.zeros((nmo, nmo))  # multipliers of the fragment orbital conditions

    # rotations inside one space leave the energy unchanged: symmetric part only
    for space in (core, act_o, act_v, ext):
        omega[np.ix_(space, space)] += -0.5 * G[np.ix_(space, space)]
    # the active occupied-virtual pairs are handled by the Z-vector below; only
    # their symmetric (overlap) part is not

    # pairs across a space boundary within one HF block: fixed by F_pq = 0
    for rows, cols in ((core, act_o), (act_v, ext)):
        if not len(rows) or not len(cols):
            continue
        gap = eps[rows][:, None] - eps[cols][None, :]
        if np.abs(gap).min() < degeneracy_tol:
            raise ValueError("near-degenerate orbitals across an active-space boundary")
        g_pq = G[np.ix_(rows, cols)]
        g_qp = G[np.ix_(cols, rows)]
        x = (g_pq - g_qp.T) / gap
        mult[np.ix_(rows, cols)] = x
        omega[np.ix_(rows, cols)] += x * eps[cols][None, :] - g_qp.T
    X_ao = _sym(C_all @ mult @ C_all.T)

    # occupied-virtual rotations of the embedded HF: Z-vector
    o, v = occ_all, vir
    C_o, C_v = C_all[:, o], C_all[:, v]
    resp_X = resp(X_ao)
    L = G[np.ix_(v, o)] - G[np.ix_(o, v)].T - 4 * C_v.T @ resp_X @ C_o

    def fvind(z):
        z = z.reshape(len(v), len(o))
        dm = C_v @ z @ C_o.T
        return 2 * (C_v.T @ resp(dm + dm.T) @ C_o).ravel()

    z = _solve_zvector(fvind, eps[o], eps[v], L, cphf_tol)
    mult[np.ix_(v, o)] = z
    Z_ao = _sym(C_v @ z @ C_o.T)
    omega[np.ix_(v, o)] += z * eps[o][None, :]
    omega[np.ix_(o, v)] += -G[np.ix_(o, v)]
    Q = resp_X + resp(Z_ao)
    omega[np.ix_(o, o)] += 2 * C_o.T @ Q @ C_o

    frag_env_grad = None
    if len(env):
        # environment rows: slaved to the global C_B by the orthogonality constraint
        C_E = C_all[:, env]
        g_eff = G[np.ix_(env, comp)].copy()
        g_eff[:, np.searchsorted(comp, o)] -= 4 * C_E.T @ Q @ C_o
        m_c = mult[np.ix_(comp, comp)]
        g_eff -= f_mo[np.ix_(env, comp)] @ (m_c + m_c.T)
        omega[np.ix_(comp, env)] += -g_eff.T
        C_B = emb.C_full_reidx[:, emb.env_idx_occ]
        R = C_B.T @ S @ C_E
        frag_env_grad = (-S @ C_all[:, comp] @ g_eff.T) @ R.T

    T = -(X_ao + Z_ao)
    g_hf = mf_emb.Gradients()
    grad = (elec_energy_deriv(g_hf, D1) - elec_energy_deriv(g_hf, D_a)
            + _hcore_contract(g_hf, D_a) + grad_nuc(mol))
    dm2_ao = np.einsum("tuvw,mt,nu,lv,sw->mnls", casdm2, C_a, C_a, C_a, C_a,
                       optimize=True)
    grad += _dm2_eri_deriv(mol, dm2_ao)
    grad += fock_deriv_contract(g_hf, D_hf, T, fd_eps)
    grad += ovlp_deriv_contract(mol, _sym(C_all @ omega @ C_all.T))
    return FragmentResponse(grad, D1 + T, env_orbital_grad=frag_env_grad)


def rdm_embedding_gradient(emb, mf_emb, mo_cas_idxs, casdm1, casdm2, proj_type="mu",
                           mf_global=None, env_cols=None, xc_grid_response=True,
                           fd_eps=1e-5, **kwargs):
    """Nuclear gradient of the WF-in-DFT energy of :func:`wf_in_dft_energy`.

    Args:
        emb: The :class:`~nbed.emb_scf.EmbedSCF` that produced ``mf_emb``.
        mf_emb: Converged embedded RHF from ``build_emb_hf`` with the same projector.
        mo_cas_idxs: Active columns of ``mf_emb.mo_coeff``, as passed to
            ``get_mo_integrals``.
        casdm1: Spin-summed active-space 1-RDM from the downstream solver.
        casdm2: Spin-summed active-space 2-RDM, PySCF ``make_rdm12`` convention.
        proj_type: ``"mu"`` or ``"huz"``.
        mf_global: Converged global RKS. Defaults to ``emb.global_scf_obj``.
        env_cols: Environment columns of ``mf_emb`` (huzinaga). Detected if ``None``.
        xc_grid_response: Include the DFT grid-weight response.
        fd_eps: Density-space step for :func:`fock_deriv_contract`.
        **kwargs: Passed to :func:`global_embedding_gradient`.

    Returns:
        ``(natm, 3)`` gradient in hartree/bohr.
    """
    if emb.SCF_type != "closed-shell":
        raise NotImplementedError("gradients are only implemented for closed shells")
    frag = rdm_fragment_response(emb, mf_emb, mo_cas_idxs, casdm1, casdm2, proj_type,
                                 env_cols=env_cols, fd_eps=fd_eps)
    return global_embedding_gradient(emb, frag, mf_global, proj_type,
                                     xc_grid_response=xc_grid_response, fd_eps=fd_eps,
                                     **kwargs)
