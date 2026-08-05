"""Choosing a strongly correlated CAS *inside* an embedded fragment.

This is a different job from :mod:`nbed.act_env_space`, which partitions a global
mean field into a fragment and an environment and whose output feeds
``EmbedSCF(...)``. Here the embedding has already run, and the question is which of
the embedded fragment's orbitals are worth handing to a correlated solver. The
output feeds ``EmbedSCF.get_mo_integrals(..., mo_cas_idxs=...)``.

The failure this exists to avoid: ranking the embedded virtuals by their Lowdin
population on the target atoms picks whatever is *most* localised there, and in a
polarised basis that is a bare d shell on a single atom, sitting one to two hartree
above the frontier. Such a CAS is a single determinant to three or four decimal
places, so a correlated solver has nothing to do with it. The criterion is not
broken, it is answering a different question: sitting on the target atoms and being
correlated are unrelated properties.

The selection is in two stages, and never allocates anything over the full virtual
space:

  1. project the embedded virtual space onto the *minimal valence* AO space of the
     target atoms and keep a bounded pool of the best-covered directions. Those are
     the antibonding partners of the occupied fragment orbitals, on the target atoms
     by construction;
  2. run MP2 inside [fragment occupied + pool] and keep the virtual natural orbitals
     with the largest occupation. This mixes the polarisation character back in where
     correlation actually wants it.

The pool is what caps the cost. Stage 1 rotates it out of the virtual space and
semicanonicalises it, which leaves it as a plain set of columns, so stage 2 is just
``mp.MP2(mf, frozen=everything_else)`` and pyscf sizes the integral transform by the
pool for us. Around three times the number of CAS virtuals is the useful setting. On
the stretched methanol of ``notebooks/3-active-space.ipynb`` a pool of 3x lands within
9 mHa of an unrestricted pool over every virtual, against 40 mHa at 1x, and a 2x pool
is small enough to select a qualitatively different active space.

Stage 1 is not decoration: choosing the pool by orbital energy instead, which is what
``frozen`` would do unaided, breaks down as soon as the basis carries diffuse
functions. See :func:`valence_virtual_pool`.

Having chosen a CAS, the second half of this module answers whether it was worth
choosing: :func:`diagnose_cas` reports natural occupations, the number of
effectively unpaired electrons, per-orbital entanglement entropies and the weight of
the reference determinant.

Orthogonality to the environment
--------------------------------

Selection cannot break it. Every step here is an *orthogonal rotation inside* a span of
columns of ``mf.mo_coeff`` from which the environment columns have been excluded: the
valence rotation of stage 1, the semicanonicalisation, the natural-orbital rotations of
stage 2, and the occupied reordering. A rotation within a subspace cannot generate
overlap with anything outside it, and the columns of a converged ``mf.mo_coeff`` are
S-orthonormal, so the CAS is orthogonal to the environment to round-off rather than
approximately. :func:`select_cas_by_mp2_no` asserts the rebuilt set is orthonormal.

Two things that guarantee does *not* cover:

  * **The partition itself.** Only ``env_cols`` says which columns are environment, and
    nothing downstream can check it, because the embedded Fock is block diagonal between
    fragment and environment by construction. See :func:`select_cas_by_mp2_no`.
  * **Anything that optimises orbitals.** Rotations that mix the pool with the excluded
    columns are exactly what an orbital optimiser explores, and with huzinaga the
    projector lives in ``get_fock`` where ``mcscf`` never looks. :func:`run_casscf`
    documents the failure and :func:`environment_overlap` detects it.

Whether the *embedded* orbitals still span the originally frozen environment is a
property of the projector, not of this module, and the two differ: huzinaga preserves
that subspace to round-off, while a mu-shift leaks at the order of ``1 / mu_val``.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field

from pyscf import gto, mcscf

import numpy
import nbed.backend as backend



from .act_env_space import describe_orbital, lowdin_populations, orbital_spread

##################################################################################
############### stage 1: where the fragment's valence space lives ################
##################################################################################


def fragment_valence_projector(mol, atom_indices, ref_basis="minao", drop_core=True):
    """Projector onto the minimal valence AO space of the target atoms.

    Ranking virtuals by target-atom population picks whatever is *most* localised,
    which in a polarised basis is a bare d shell on one atom: those orbitals sit tens
    of eV above the frontier and correlate nothing. The minimal valence space of the
    same atoms instead contains exactly the antibonding partners of the occupied
    fragment orbitals, so projecting onto it selects sigma*/pi* rather than
    polarisation functions.

    Args:
        mol: Molecule whose AO basis is being projected.
        atom_indices: 0-based indices of the target atoms.
        ref_basis: Minimal basis defining "valence", e.g. ``"minao"``.
        drop_core: Leave the core shells out of the reference space, so the projector
            only spans valence directions.

    Returns:
        Tuple ``(p_ref, n_ref)``: the ``(nao, nao)`` AO-basis projector, in the sense
        that ``C.T @ p_ref @ C`` has eigenvalues in ``[0, 1]`` for orthonormal ``C``,
        and the dimension of the reference space.

    Raises:
        ValueError: If the reference space comes out empty.
    """
    coords = mol.atom_coords(unit="Bohr")
    ref_mol = gto.Mole()
    ref_mol.atom = [(mol.atom_pure_symbol(i), coords[i]) for i in atom_indices]
    ref_mol.unit = "Bohr"
    ref_mol.basis = ref_basis
    # let pyscf guess the spin, so an odd-electron fragment does not raise
    ref_mol.spin = None
    ref_mol.charge = 0
    ref_mol.build(parse_arg=False, verbose=0)

    # minao holds one shell per occupied atomic shell, so the core AOs of an atom are
    # the first few of its block: none up to He, the 1s up to Ne, the 1s2s2p up to Ar
    keep = backend.xp.ones(ref_mol.nao, dtype=bool)
    if drop_core:
        ao_slices = ref_mol.aoslice_by_atom()
        for offset, atom in enumerate(atom_indices):
            charge = mol.atom_charge(atom)
            n_core_ao = 0 if charge <= 2 else (1 if charge <= 10 else 5)
            start = ao_slices[offset, 2]
            keep[start:start + n_core_ao] = False
    if not keep.any():
        raise ValueError("the reference valence space is empty")

    if backend.USING_GPU:
        s_cross = backend.xp.asarray(gto.intor_cross("int1e_ovlp", mol, ref_mol))[:, keep]
        s_ref = backend.xp.asarray(ref_mol.intor("int1e_ovlp"))[backend.xp.ix_(keep, keep)]
    else:
        s_cross = gto.intor_cross("int1e_ovlp", mol, ref_mol)[:, keep]
        s_ref = ref_mol.intor("int1e_ovlp")[backend.xp.ix_(keep, keep)]


    evals, evecs = backend.xp.linalg.eigh(s_ref)
    ok = evals > 1e-10
    s_ref_inv = (evecs[:, ok] / evals[ok]) @ evecs[:, ok].T
    return s_cross @ s_ref_inv @ s_cross.T, int(keep.sum())


def valence_virtual_rotation(mol, c_vir, atom_indices, ref_basis="minao", fock_mo=None):
    """Rotate an orbital block so its leading columns are fragment valence.

    Diagonalises the fragment valence projector inside the block. Each eigenvalue is
    the fraction of that rotated orbital lying in the minimal valence space of the
    target atoms, so it falls from near one for the antibonding orbitals to near zero
    for polarisation and diffuse functions, and the drop says how many are worth
    taking. Named for its usual use on the virtuals, but any orthonormal block works,
    including the occupied one.

    Args:
        mol: Molecule supplying the integrals.
        c_vir: ``(nao, nvir)`` orthonormal orbitals.
        atom_indices: 0-based target atom indices.
        ref_basis: Minimal basis defining the valence space.
        fock_mo: Optional ``(nvir,)`` diagonal or ``(nvir, nvir)`` Fock matrix in the
            ``c_vir`` basis. When given, the returned orbitals are reported with their
            expectation-value energies; the rotation itself is unchanged.

    Returns:
        Tuple ``(c_rot, weights, energies)``: rotated orbitals ordered by decreasing
        valence weight, their weights, and their diagonal Fock values (``None`` if
        ``fock_mo`` was not given).
    """
    p_ref, _ = fragment_valence_projector(mol, atom_indices, ref_basis)
    metric = c_vir.T @ p_ref @ c_vir
    weights, rot = backend.xp.linalg.eigh(metric)
    order = backend.xp.argsort(-weights)
    weights, rot = weights[order], rot[:, order]
    c_rot = c_vir @ rot

    energies = None
    if fock_mo is not None:
        fock_mo = backend.xp.asarray(fock_mo)
        if fock_mo.ndim == 1:
            fock_mo = backend.xp.diag(fock_mo)
        energies = backend.xp.einsum("ip,ij,jp->p", rot, fock_mo, rot)
    return c_rot, weights, energies


##################################################################################
################### keeping the reference determinant intact #####################
##################################################################################


def semicanonicalize(c_block, c_ref, mo_energy_ref, ovlp):
    """Diagonalise the reference Fock matrix inside a block of orbitals.

    A rotation inside the virtual block leaves the reference determinant untouched, so
    this costs nothing and buys meaningful orbital energies, which MP2 needs: its
    denominators are only orbital energy differences if the Fock matrix is diagonal.
    Without this step the valence rotation of stage 1 would leave the pool
    non-canonical and the MP2 amplitudes would be wrong. The Fock matrix is read off
    the reference orbitals, which are canonical, instead of being rebuilt.

    Args:
        c_block: ``(nao, k)`` orbitals spanning the block.
        c_ref: ``(nao, nmo)`` canonical reference orbitals, spanning the whole space.
        mo_energy_ref: Their orbital energies.
        ovlp: AO overlap matrix.

    Returns:
        Tuple ``(c_new, energies)``, energies ascending.
    """
    rot = c_ref.T @ ovlp @ c_block
    fock = backend.xp.einsum("ip,i,iq->pq", rot, mo_energy_ref, rot)
    energies, vecs = backend.xp.linalg.eigh(fock)
    return c_block @ vecs, energies


##################################################################################
################## stage 1 continued: the bounded orbital pool ###################
##################################################################################


def valence_virtual_pool(mol, c_vir, atom_indices, n_pool, c_ref=None,
                         mo_energy_ref=None, ovlp=None, ref_basis="minao"):
    """Bounded pool of virtuals covering the fragment's minimal valence space.

    This is where the cost is capped: MP2 then correlates only the pool, so its
    integral transform is sized by ``n_pool`` rather than by the basis.

    Choosing the pool by valence character rather than by orbital energy is what makes
    the truncation safe. Simply keeping the lowest-energy virtuals is the obvious
    alternative and it collapses as soon as the basis carries diffuse functions,
    because the lowest virtuals are then Rydberg-like and correlate nothing: on the
    methanol of ``notebooks/3-active-space.ipynb`` in aug-cc-pVDZ, a six-orbital pool
    chosen by energy gives an active space with *zero* unpaired electrons, while the
    same-sized valence pool recovers half the untruncated value.

    Args:
        mol: Molecule of the embedded subsystem.
        c_vir: ``(nao, nvir)`` embedded virtual orbitals.
        atom_indices: 0-based target atom indices.
        n_pool: How many virtuals to keep.
        c_ref: Canonical reference orbitals, for semicanonicalising the pool. Required
            if the pool is going to be correlated, since MP2 needs a diagonal Fock.
        mo_energy_ref: Their energies.
        ovlp: AO overlap matrix.
        ref_basis: Minimal basis defining the valence space.

    Returns:
        Tuple ``(c_pool, weights, energies)``: the pool orbitals, the fragment valence
        weight of each of the ``nvir`` rotated virtuals (so the tail shows what was
        left out), and the pool orbital energies or ``None``.

    Raises:
        ValueError: If ``n_pool`` exceeds the number of virtuals available.
    """
    if n_pool > c_vir.shape[1]:
        raise ValueError(
            f"asked for a pool of {n_pool} but only {c_vir.shape[1]} virtuals are "
            "available; shrink n_pool or enlarge the basis"
        )
    c_rot, weights, _ = valence_virtual_rotation(
        mol, c_vir, atom_indices, ref_basis=ref_basis)
    c_pool = c_rot[:, :n_pool]
    energies = None
    if c_ref is not None:
        c_pool, energies = semicanonicalize(c_pool, c_ref, mo_energy_ref, ovlp)
    return c_pool, weights, energies


##################################################################################
################ stage 2: MP2 natural orbitals inside the pool ###################
##################################################################################


@dataclass
class PoolMP2:
    """MP2 natural orbitals of both blocks of the correlated space.

    Attributes:
        rot_vir: ``(npool, npool)`` expansion of the virtual natural orbitals in the
            pool columns, ordered by *decreasing* occupation, so the leading ones are
            the virtuals correlation actually uses.
        occ_vir: Their occupations.
        rot_occ: ``(nocc, nocc)`` expansion of the occupied natural orbitals in the
            occupied columns, ordered by *increasing* occupation, so the leading ones
            are the most depleted and therefore the most correlated.
        occ_occ: Their occupations.
        e_corr: MP2 correlation energy inside [occupied + pool].
    """

    rot_vir: backend.xp.ndarray = field(repr=False)
    occ_vir: backend.xp.ndarray
    rot_occ: backend.xp.ndarray = field(repr=False)
    occ_occ: backend.xp.ndarray = None
    e_corr: float = 0.0


def pool_mp2_natural_orbitals(mf, occ_cols, pool_cols, verify_fock=True, fock_tol=1e-4,
                              max_orbital_energy=1e3):
    """Natural orbitals of an MP2 that correlates only [occupied + pool].

    Everything outside those columns is handed to pyscf's ``frozen`` keyword, so the
    integral transform and the amplitudes are sized by the pool rather than by the
    basis, and pyscf does the truncation instead of this module rebuilding a
    subsystem mean field by hand.

    The embedding projector plays no part. It is built to vanish on any space
    orthogonal to the occupied environment, and canonical MP2 needs only orbital
    energies and two-electron integrals, so the embedded Fock enters solely through
    ``mf.mo_energy``.

    Note what this means for environment leakage: the embedded Fock is block diagonal
    between the fragment and the environment *by construction*, that being the whole
    point of the projector, so no Fock-based test can notice an environment orbital in
    the pool. Nothing here can infer which columns are environment, and the caller must
    say so via ``env_cols``. What is detectable is the fingerprint the recommended
    ``huz_level_shift`` workflow leaves behind, an orbital parked at ``lambda``, and
    ``max_orbital_energy`` catches that.

    Args:
        mf: Converged embedded *Hartree-Fock* object, whose ``mo_coeff`` already holds
            the semicanonicalised pool in ``pool_cols``. MP2 perturbs an HF
            determinant, so an embedded DFT object is not a valid reference.
        occ_cols: Columns of the occupied orbitals to correlate.
        pool_cols: Columns of the pool virtuals to correlate.
        verify_fock: Check the embedded Fock matrix is block diagonal between the
            occupied orbitals and the pool, and diagonal within the correlated space
            with ``mf.mo_energy`` on it. The second fails if the pool was not
            semicanonicalised, which would silently invalidate the MP2 amplitudes.
        fock_tol: Tolerance for those checks, in hartree.
        max_orbital_energy: Reject the pool if any of its orbital energies exceeds this,
            in hartree. No physical virtual sits there, so it means a projected
            environment orbital was left in, whether shifted by ``huz_level_shift`` or
            by ``mu_val``. Set to ``None`` to skip.

    Returns:
        A :class:`PoolMP2`.

    Raises:
        AssertionError: If ``verify_fock`` and the reference is not a valid one.
        ValueError: If the pool contains a projected environment orbital.
    """
    occ_cols = backend.xp.asarray(occ_cols, dtype=int)
    pool_cols = backend.xp.asarray(pool_cols, dtype=int)
    active = backend.xp.concatenate([occ_cols, pool_cols])
    frozen = backend.xp.setdiff1d(backend.xp.arange(backend.xp.shape(mf.mo_coeff)[1]), active)

    if max_orbital_energy is not None:
        e_pool = backend.xp.asarray(mf.mo_energy)[pool_cols]
        stray = pool_cols[backend.xp.abs(e_pool) > max_orbital_energy]
        if len(stray):
            raise ValueError(
                f"pool columns {stray} have orbital energies "
                f"{backend.xp.round(e_pool[backend.xp.abs(e_pool) > max_orbital_energy], 1)} Ha, which "
                "means projected environment orbitals were left in the pool; pass the "
                "env_cols returned by build_emb_hf / build_emb_dft so they are excluded"
            )

    if verify_fock:
        rdm1ao = mf.make_rdm1()
        vhf = mf.get_veff(mol=mf.mol, dm=rdm1ao)
        fock_ao = mf.get_fock(vhf=vhf, dm=rdm1ao)
        fock_mo = backend.xp.asarray(mf.mo_coeff).T @ backend.xp.asarray(fock_ao) @ backend.xp.asarray(mf.mo_coeff)
        off = backend.xp.abs(fock_mo[backend.xp.ix_(occ_cols, pool_cols)]).max()
        assert off < fock_tol, (
            f"the occupied-pool Fock block is {off:.1e}, so the pool is not a valid MP2 "
            "reference; either the mean field is not converged or the pool orbitals did "
            "not come from its virtual space"
        )
        diag = backend.xp.abs(backend.xp.diag(fock_mo)[active] - backend.xp.asarray(mf.mo_energy)[active]).max()
        assert diag < fock_tol, (
            f"orbital energies disagree with the Fock diagonal by {diag:.1e}: "
            "semicanonicalise the pool before correlating it"
        )

    pt = backend.pyscf.mp.MP2(mf, frozen=[int(i) for i in frozen])
    pt.verbose = 0
    pt.kernel()
    dm1 = pt.make_rdm1()

    occs_v, vecs_v = backend.xp.linalg.eigh(dm1[backend.xp.ix_(pool_cols, pool_cols)])
    order_v = backend.xp.argsort(-occs_v)
    occs_o, vecs_o = backend.xp.linalg.eigh(dm1[backend.xp.ix_(occ_cols, occ_cols)])
    order_o = backend.xp.argsort(occs_o)

    return PoolMP2(
        rot_vir=vecs_v[:, order_v],
        occ_vir=occs_v[order_v],
        rot_occ=vecs_o[:, order_o],
        occ_occ=occs_o[order_o],
        e_corr=float(pt.e_corr),
    )


##################################################################################
############### picking CAS columns without tripping over indices ################
##################################################################################


def cas_columns_excluding_env(mo_occ, env_cols, n_occ, n_vir, nelectron=None):
    """CAS columns chosen by role, never by position.

    ``mcscf.CASCI`` takes its active space as a contiguous window ``range(ncore, ncore +
    ncas)`` around the frontier. After huzinaga embedding the environment orbitals are
    shifted by only ``|eps_env|`` each, so they land scattered among the active virtuals
    and that window silently swallows one. The result is a plausible-looking energy that
    is wrong by tens of millihartree, with no warning: excluding the environment is the
    only thing that prevents it, since the projector itself contributes nothing inside
    the CAS.

    Args:
        mo_occ: Occupations of the *embedded* mean field.
        env_cols: Columns of that mean field carrying the environment, as returned by
            ``build_emb_dft`` / ``build_emb_hf`` / ``check_embedding``.
        n_occ: How many occupied orbitals to make active, taken from the frontier down.
        n_vir: How many virtuals to make active, taken from the frontier up.
        nelectron: Electron count of the embedded subsystem, used only to report the
            contiguous window that would have been used instead.

    Returns:
        Tuple ``(mo_cas_idxs, nelecas, naive_window, collision)``: the safe columns, the
        ``(na, nb)`` active electron count, the contiguous window ``CASCI`` would have
        taken, and the environment columns that window would have swallowed.

    Raises:
        ValueError: If too few orbitals survive the environment exclusion.
    """
    mo_occ = backend.xp.asarray(mo_occ)
    env_cols = backend.xp.asarray(env_cols, dtype=int)
    occ_cols = backend.xp.where(mo_occ > 0)[0]
    vir_safe = backend.xp.setdiff1d(backend.xp.where(mo_occ == 0)[0], env_cols)

    if len(occ_cols) < n_occ or len(vir_safe) < n_vir:
        raise ValueError(
            f"asked for {n_occ} occupied and {n_vir} virtual orbitals but only "
            f"{len(occ_cols)} occupied and {len(vir_safe)} environment-free virtuals "
            "exist"
        )

    cas_occ = occ_cols[-n_occ:]
    cas_vir = vir_safe[:n_vir]
    mo_cas_idxs = backend.xp.concatenate([cas_occ, cas_vir])

    occ_in_cas = mo_occ[cas_occ]
    nelecas = (int((occ_in_cas > 0).sum()), int((occ_in_cas > 1).sum()))

    naive_window = backend.xp.array([], dtype=int)
    if nelectron is not None:
        ncore_naive = (int(nelectron) - 2 * n_occ) // 2
        naive_window = backend.xp.arange(ncore_naive, ncore_naive + n_occ + n_vir)
    collision = backend.xp.intersect1d(naive_window, env_cols)

    return mo_cas_idxs, nelecas, naive_window, collision


@dataclass
class CASSelection:
    """A chosen CAS, in a form ``EmbedSCF.get_mo_integrals`` can consume directly.

    Attributes:
        c_full: ``(nao, nmo)`` orbital set spanning exactly what the embedded mean field
            spanned, with the pool replaced by MP2 natural orbitals. Pass this as
            ``act_emb_C``.
        mo_cas_idxs: Columns of :attr:`c_full` forming the CAS. Pass this as
            ``mo_cas_idxs``.
        ncas: Size of the CAS.
        nelecas: ``(na, nb)`` active electrons.
        cas_occ_cols: The occupied part of :attr:`mo_cas_idxs`.
        cas_vir_cols: The virtual part, ordered by decreasing natural occupation.
        cas_natural_occ: MP2 natural occupations of the CAS virtuals, in the same order.
        pool_natural_occ: The same for every pool virtual, so the tail shows what was
            left out and whether the pool was big enough.
        occupied_natural_occ: MP2 natural occupations of the active occupied orbitals,
            ``None`` when every occupied orbital was kept active. Values still at 2 to
            several decimals mean that orbital contributes nothing.
        fragment_natural_occ: The same for *every* fragment occupied orbital, most
            depleted first, so the tail shows what freezing gave up. Compare its head
            against :attr:`occupied_natural_occ` to judge where ``n_cas_occ`` cut.
        valence_weights: Fragment valence weight of every environment-free virtual,
            descending. The drop says how many virtuals were worth pooling.
        n_pool: How many virtuals entered the pool.
        e_corr_pool: MP2 correlation energy inside [occupied + pool], the yardstick for
            pool-size convergence.
        atom_indices: The target atoms.
    """

    c_full: backend.xp.ndarray = field(repr=False)
    mo_cas_idxs: backend.xp.ndarray
    ncas: int
    nelecas: tuple[int, int]
    cas_occ_cols: backend.xp.ndarray
    cas_vir_cols: backend.xp.ndarray
    cas_natural_occ: backend.xp.ndarray
    pool_natural_occ: backend.xp.ndarray = field(repr=False)
    occupied_natural_occ: backend.xp.ndarray = field(default=None, repr=False)
    fragment_natural_occ: backend.xp.ndarray = field(default=None, repr=False)
    valence_weights: backend.xp.ndarray = field(default=None, repr=False)
    n_pool: int = 0
    e_corr_pool: float = 0.0
    atom_indices: backend.xp.ndarray = field(default=None, repr=False)


def select_cas_by_mp2_no(mf, atom_indices, n_cas_vir, n_cas_occ=None, n_pool=None,
                         env_cols=(), ovlp=None, ref_basis="minao", pool_factor=3,
                         valence_tol=1e-6, verify_fock=True, fock_tol=1e-4):
    """Choose a CAS from an embedded mean field by MP2 natural occupation.

    Runs the two-stage pipeline: a bounded pool of fragment-valence virtuals, then MP2
    inside [fragment occupied + pool], keeping the virtual natural orbitals with the
    largest occupation. Cost is set by ``n_pool``, not by the size of the virtual space.

    Args:
        mf: Converged embedded *Hartree-Fock* object, as returned by
            ``EmbedSCF.build_emb_hf``. MP2 perturbs an HF determinant, so an embedded
            DFT object is not a valid reference here.
        atom_indices: 0-based atoms whose valence space defines "interesting". These
            index the embedded molecule, which shares its geometry with the global one.
        n_cas_vir: How many virtuals to put in the CAS.
        n_cas_occ: How many occupied orbitals to make active, ranked by how far MP2 has
            depleted them below 2. ``None`` keeps every embedded occupied orbital active,
            which is the natural choice for a small fragment.
        n_pool: Pool size. Defaults to ``pool_factor * n_cas_vir``, capped by how many
            environment-free virtuals exist.
        env_cols: Columns of ``mf`` carrying the environment, as returned by
            ``build_emb_hf``. Excluding them is the *only* guard against environment
            character reaching the CAS: the embedded Fock is block diagonal between
            fragment and environment by construction, so nothing downstream can infer
            them, and the projector contributes nothing inside a CAS and so cannot
            rescue one. Getting this wrong costs tens of millihartree with no warning.
        ovlp: AO overlap matrix, defaulting to ``mf.get_ovlp()``.
        ref_basis: Minimal basis defining the valence space.
        pool_factor: Multiplier setting the default pool size. The default of 3 suits a
            fragment whose valence space is comfortably larger than the CAS; a small
            fragment can run out of valence virtuals first, which ``valence_tol`` warns
            about.
        valence_tol: Warn when the pool reaches into virtuals whose fragment valence
            weight is below this. The projector has the rank of the fragment's minimal
            valence space, so past that rank the weights are *exactly* zero and there is
            nothing left to rank by: which of those orbitals the pool takes is decided by
            round-off in a degenerate eigenvector, and the answer stops being reproducible
            at the 1e-4 level between otherwise equivalent embeddings. Enlarging the pool
            still helps on average, so this is a reproducibility warning rather than an
            accuracy one, and it is skipped when the pool spans every environment-free
            virtual, since then no choice is being made. Set to ``None`` to silence.
        verify_fock: Check that the pool really is a valid MP2 reference.
        fock_tol: Tolerance for that check, in hartree.

    Returns:
        A :class:`CASSelection`.

    Raises:
        NotImplementedError: If the embedded fragment is open shell.
        ValueError: If the requested sizes do not fit.
    """
    mol = mf.mol
    ovlp = mf.get_ovlp() if ovlp is None else ovlp
    mo_occ = backend.xp.rint(mf.mo_occ).astype(int)

    if backend.xp.any(mo_occ == 1):
        raise NotImplementedError(
            "MP2 natural-orbital selection assumes a closed-shell embedded fragment; "
            "the pool reference is built as a doubly occupied determinant"
        )

    occ_cols = backend.xp.where(mo_occ > 0)[0]
    vir_safe = backend.xp.setdiff1d(backend.xp.where(mo_occ == 0)[0], backend.xp.asarray(env_cols, dtype=int))

    n_pool = min(len(vir_safe), pool_factor * n_cas_vir) if n_pool is None else n_pool
    if not n_cas_vir <= n_pool <= len(vir_safe):
        raise ValueError(
            f"need n_cas_vir ({n_cas_vir}) <= n_pool ({n_pool}) <= environment-free "
            f"virtuals ({len(vir_safe)})"
        )
    if n_cas_occ is not None and n_cas_occ > len(occ_cols):
        raise ValueError(
            f"asked for {n_cas_occ} active occupied orbitals but the fragment only has "
            f"{len(occ_cols)}"
        )

    c_occ = mf.mo_coeff[:, occ_cols]
    c_vir = mf.mo_coeff[:, vir_safe]
    pool_cols = vir_safe[:n_pool]

    # stage 1: rank the environment-free virtuals by fragment valence character, then
    # semicanonicalise the pool so that MP2 has a diagonal Fock to work with
    c_rot, valence_weights, _ = valence_virtual_rotation(
        mol, c_vir, atom_indices, ref_basis=ref_basis)

    # a pool spanning every environment-free virtual makes no choice, so nothing can be
    # arbitrary about it however small the trailing weights are
    if (valence_tol is not None and n_pool < len(vir_safe)
            and valence_weights[n_pool - 1] < valence_tol):
        n_useful = int((valence_weights > valence_tol).sum())
        warnings.warn(
            f"the pool of {n_pool} reaches past this fragment's valence space, which only "
            f"spans {n_useful} virtuals: orbitals {n_useful} onwards have valence weight "
            f"below {valence_tol:g}, so which of them the pool takes is set by round-off "
            f"and the result is not reproducible to better than ~1e-4 Ha. Either drop "
            f"n_pool to {n_useful} or pass the whole environment-free virtual space "
            f"({len(vir_safe)}), where the ordering stops mattering.",
            stacklevel=2,
        )

    c_pool, e_pool = semicanonicalize(
        c_rot[:, :n_pool], mf.mo_coeff, mf.mo_energy, ovlp)

    # write the rotated orbitals back into a copy of the mean field, so that the pool is
    # a plain set of columns and pyscf's frozen-orbital MP2 can do the truncation
    mf_pool = copy.copy(mf)
    mf_pool.mo_coeff = backend.xp.asarray(mf.mo_coeff).copy()
    mf_pool.mo_energy = backend.xp.asarray(mf.mo_energy).copy()
    mf_pool.mo_coeff[:, pool_cols] = c_pool
    mf_pool.mo_energy[pool_cols] = e_pool
    mf_pool.mo_coeff[:, vir_safe[n_pool:]] = c_rot[:, n_pool:]

    # stage 2: MP2 over [occupied + pool] only. The occupied block stays canonical, so
    # the reference is stationary and the MP2 denominators are honest.
    pool_mp2 = pool_mp2_natural_orbitals(
        mf_pool, occ_cols, pool_cols, verify_fock=verify_fock, fock_tol=fock_tol)

    # the natural orbitals are kept in occupation order rather than semicanonicalised,
    # so that every CAS virtual can still be reported with the occupation that chose it
    c_full = mf_pool.mo_coeff.copy()
    c_full[:, pool_cols] = c_pool @ pool_mp2.rot_vir
    cas_vir_cols = pool_cols[:n_cas_vir]

    occupied_natural_occ = None
    if n_cas_occ is None:
        cas_occ_cols = occ_cols
    else:
        # Rank the occupied orbitals the same way as the virtuals, by how far MP2 has
        # pushed them off 2. Geometric criteria cannot do this job: a lone pair and a
        # stretched sigma bond can carry the same fragment valence weight while only the
        # latter is correlated. Rotating inside the occupied block leaves the determinant
        # untouched, so this only changes which orbitals CASCI freezes as core.
        c_full[:, occ_cols] = c_occ @ pool_mp2.rot_occ
        cas_occ_cols = occ_cols[:n_cas_occ]
        occupied_natural_occ = pool_mp2.occ_occ[:n_cas_occ]

    assert backend.xp.allclose(c_full.T @ ovlp @ c_full, backend.xp.eye(c_full.shape[1]), atol=1e-8), \
        "the rebuilt orbital set is not orthonormal"

    mo_cas_idxs = backend.xp.concatenate([cas_occ_cols, cas_vir_cols])
    occ_in_cas = mo_occ[cas_occ_cols]
    nelecas = (int((occ_in_cas > 0).sum()), int((occ_in_cas > 1).sum()))

    return CASSelection(
        c_full=c_full,
        mo_cas_idxs=mo_cas_idxs,
        ncas=len(mo_cas_idxs),
        nelecas=nelecas,
        cas_occ_cols=cas_occ_cols,
        cas_vir_cols=cas_vir_cols,
        cas_natural_occ=pool_mp2.occ_vir[:n_cas_vir],
        pool_natural_occ=pool_mp2.occ_vir,
        occupied_natural_occ=occupied_natural_occ,
        fragment_natural_occ=pool_mp2.occ_occ,
        valence_weights=valence_weights,
        n_pool=n_pool,
        e_corr_pool=pool_mp2.e_corr,
        atom_indices=backend.xp.asarray(atom_indices),
    )


def report_cas_orbitals(mol, c_cas, atom_indices, n_occ, mo_energy=None, p_ref=None,
                        occ_no=None):
    """Readable table of what ended up in the CAS.

    Args:
        mol: Molecule of the embedded subsystem.
        c_cas: ``(nao, ncas)`` CAS orbitals, occupied first.
        atom_indices: 0-based target atom indices.
        n_occ: How many leading columns are occupied.
        mo_energy: Optional ``(ncas,)`` orbital energies.
        p_ref: Optional fragment valence projector, for a valence-weight column.
        occ_no: Optional MP2 natural occupations of the CAS virtuals.

    Returns:
        A multi-line string.
    """
    per_atom, pop = lowdin_populations(mol, c_cas, atom_indices)
    spread = orbital_spread(mol, c_cas)
    if p_ref is None:
        p_ref, _ = fragment_valence_projector(mol, atom_indices)
    valence = backend.xp.einsum("ip,ij,jp->p", c_cas, p_ref, c_cas)

    lines = ["    idx  occ    energy  fragpop  valence  spread   MP2 nat occ"
             "   composition"]
    for column in range(c_cas.shape[1]):
        occ = 2.0 if column < n_occ else 0.0
        energy = "        " if mo_energy is None else f"{mo_energy[column]:8.4f}"
        nat = "           "
        if occ_no is not None and column >= n_occ:
            nat = f"{occ_no[column - n_occ]:11.5f}"
        lines.append(
            f"    {column:3d}  {occ:3.1f}  {energy}  {pop[column]:7.3f}  "
            f"{valence[column]:7.3f}  {spread[column]:6.2f}  {nat}   "
            f"{describe_orbital(mol, atom_indices, per_atom, column)}")
    lines.append(
        f"    fragment population: occupied min {pop[:n_occ].min():.3f}, "
        f"virtual min {pop[n_occ:].min():.3f} mean {pop[n_occ:].mean():.3f}; "
        f"virtual valence weight {valence[n_occ:].sum():.2f}")
    return "\n".join(lines)


##################################################################################
################## was the CAS worth choosing? strong-correlation ################
##################################################################################


def unpaired_electrons(occ):
    """Number of effectively unpaired electrons from natural occupations.

    Zero for a closed-shell single determinant and near two for a perfect diradical, so
    it is the single most direct answer to "is this active space interesting". The
    nonlinear form suppresses the contribution of the many weakly correlated orbitals
    that dynamic correlation sprinkles across the whole virtual space, so it isolates the
    genuinely open-shell ones.

    Args:
        occ: Natural occupations of the spatial orbitals, in ``[0, 2]``.

    Returns:
        Tuple ``(n_u, n_u_nl)``: Head-Gordon's ``sum min(n, 2 - n)`` and the nonlinear
        ``sum n^2 (2 - n)^2``.
    """
    occ = backend.xp.clip(backend.xp.asarray(occ, dtype=float), 0.0, 2.0)
    return float(backend.xp.sum(backend.xp.minimum(occ, 2.0 - occ))), \
        float(backend.xp.sum(occ ** 2 * (2.0 - occ) ** 2))


def von_neumann_entropy(occ):
    """Entropy of a set of natural occupations, treating orbitals as independent.

    Needs only a 1-RDM, so unlike :func:`single_orbital_entropies` it can be applied to
    MP2 natural occupations, where there is no CI vector to interrogate. The
    independence assumption makes it an approximation to the true orbital entropy, but it
    tracks the same trend and is available before any CASCI has been run.

    Args:
        occ: Natural occupations of the spatial orbitals, in ``[0, 2]``.

    Returns:
        The entropy in nats. Zero for occupations of exactly 0 or 2.
    """
    p = backend.xp.clip(backend.xp.asarray(occ, dtype=float) / 2.0, 0.0, 1.0)
    terms = backend.xp.zeros_like(p)
    for q in (p, 1.0 - p):
        ok = q > 1e-14
        terms[ok] -= q[ok] * backend.xp.log(q[ok])
    return float(terms.sum())


def single_orbital_entropies(casci, ci=None):
    """Per-orbital entanglement entropy from the CASCI reduced density matrices.

    The standard strong-correlation measure from DMRG orbital-entanglement analysis. Each
    spatial orbital has four possible local states, empty, up, down and doubly occupied,
    and their probabilities follow from the spin-resolved 1- and 2-RDMs. An orbital that
    is cleanly empty or cleanly doubly occupied contributes nothing; one that is
    genuinely entangled with the rest of the active space contributes up to ``ln 4``.
    Reading the double occupancy straight off the alpha-beta 2-RDM avoids any ambiguity
    over spin-summed conventions.

    Args:
        casci: A solved ``mcscf.CASCI`` (or ``CASSCF``) object.
        ci: CI vector to use, defaulting to ``casci.ci``.

    Returns:
        Tuple ``(s, s_total)``: the ``(ncas,)`` per-orbital entropies in nats and their
        sum, which measures how much of the active space is genuinely correlated.

    Raises:
        AttributeError: If the FCI solver cannot produce spin-resolved RDMs.
    """
    ci = casci.ci if ci is None else ci
    ncas, nelecas = casci.ncas, casci.nelecas
    if not hasattr(casci.fcisolver, "make_rdm12s"):
        raise AttributeError(
            "the FCI solver does not provide make_rdm12s, so the spin-resolved double "
            "occupancy is unavailable; use von_neumann_entropy instead"
        )
    (dm1a, dm1b), (_, dm2ab, _) = casci.fcisolver.make_rdm12s(ci, ncas, nelecas)

    diag = backend.xp.arange(ncas)
    w_both = dm2ab[diag, diag, diag, diag]          # <n_up n_down> on the same orbital
    w_up = backend.xp.diag(dm1a) - w_both
    w_down = backend.xp.diag(dm1b) - w_both
    w_empty = 1.0 - w_up - w_down - w_both

    entropies = backend.xp.zeros(ncas)
    for weights in (w_empty, w_up, w_down, w_both):
        ok = weights > 1e-14
        entropies[ok] -= weights[ok] * backend.xp.log(weights[ok])
    return entropies, float(entropies.sum())


def reference_weight(ci):
    """Weight of the aufbau determinant in a CI vector.

    Address zero of a pyscf CI vector is the determinant filling the lowest active
    orbitals, so for a CAS built on a mean-field reference this is the weight of that
    reference. Far from one means the single-determinant picture has broken down.

    Args:
        ci: CI vector, or a list of them, in which case the first root is used.

    Returns:
        Tuple ``(c0_squared, dominant_squared)``: the aufbau determinant's weight and
        the largest weight of any determinant. They differ once the reference stops being
        the dominant configuration.
    """
    ci = backend.xp.asarray(ci if not isinstance(ci, (list, tuple)) else ci[0])
    if ci.ndim > 2:
        ci = ci[0]
    return float(ci.flat[0] ** 2), float(backend.xp.max(ci ** 2))


@dataclass
class CASDiagnostics:
    """How strongly correlated a CAS turned out to be.

    Attributes:
        natural_occ: CAS natural occupations, descending.
        n_unpaired: Head-Gordon effectively unpaired electron count.
        n_unpaired_nl: Its nonlinear form, which suppresses dynamic correlation.
        entropy_occ: Entropy of the natural occupations alone.
        orbital_entropies: Per-orbital entanglement entropy, ``None`` if the solver
            could not supply spin-resolved RDMs.
        entropy_total: Their sum.
        reference_weight: Weight of the aufbau determinant.
        dominant_weight: Weight of the largest determinant.
        e_tot: Total CASCI energy of the embedded subsystem.
    """

    natural_occ: backend.xp.ndarray
    n_unpaired: float
    n_unpaired_nl: float
    entropy_occ: float
    orbital_entropies: backend.xp.ndarray = field(default=None, repr=False)
    entropy_total: float = 0.0
    reference_weight: float = 0.0
    dominant_weight: float = 0.0
    e_tot: float = 0.0

    def report(self):
        """Render the diagnostics as a printable multi-line string."""
        lines = [
            f"  CASCI energy of the subsystem : {self.e_tot:.8f} Ha",
            f"  natural occupations           : "
            f"{backend.xp.array2string(self.natural_occ, precision=4, suppress_small=True)}",
            f"  effectively unpaired electrons: {self.n_unpaired:.4f}"
            f"   (nonlinear {self.n_unpaired_nl:.4f})",
            f"  entropy of the occupations    : {self.entropy_occ:.4f} nats",
            f"  aufbau determinant weight     : {self.reference_weight:.4f}"
            f"   so 1 - c0^2 = {1 - self.reference_weight:.4f}",
            f"  largest determinant weight    : {self.dominant_weight:.4f}",
        ]
        if self.orbital_entropies is not None:
            lines += [
                f"  summed orbital entropy        : {self.entropy_total:.4f} nats",
                "  per-orbital entropy           : "
                + backend.xp.array2string(self.orbital_entropies, precision=4,
                                  suppress_small=True),
            ]
        verdict = ("strongly correlated" if self.n_unpaired > 0.5 else
                   "moderately correlated" if self.n_unpaired > 0.1 else
                   "essentially a single determinant")
        lines.append(f"  verdict                       : {verdict}")
        return "\n".join(lines)


def diagnose_cas(casci, verbose=True):
    """Measure how strongly correlated a solved CAS is.

    Args:
        casci: A solved ``mcscf.CASCI`` (or ``CASSCF``) object.
        verbose: Print the report as well as returning it.

    Returns:
        A :class:`CASDiagnostics`.
    """
    ncas, nelecas = casci.ncas, casci.nelecas
    dm1 = casci.fcisolver.make_rdm1(casci.ci, ncas, nelecas)
    occ = backend.xp.linalg.eigvalsh(dm1)[::-1]

    n_u, n_u_nl = unpaired_electrons(occ)
    try:
        orbital_entropies, entropy_total = single_orbital_entropies(casci)
    except AttributeError:
        orbital_entropies, entropy_total = None, 0.0
    c0_sq, dom_sq = reference_weight(casci.ci)

    diagnostics = CASDiagnostics(
        natural_occ=occ,
        n_unpaired=n_u,
        n_unpaired_nl=n_u_nl,
        entropy_occ=von_neumann_entropy(occ),
        orbital_entropies=orbital_entropies,
        entropy_total=entropy_total,
        reference_weight=c0_sq,
        dominant_weight=dom_sq,
        e_tot=float(casci.e_tot),
    )
    if verbose:
        print(diagnostics.report())
    return diagnostics


def environment_overlap(c_orbitals, c_env, ovlp):
    """Largest overlap between a set of orbitals and the environment.

    The number to check whenever orbitals have been *optimised* rather than merely
    selected. Selection cannot break environment orthogonality, so this returns
    round-off for anything coming out of :func:`select_cas_by_mp2_no` or
    :func:`run_casci`; optimisation can, and then this is the only thing that notices.

    Args:
        c_orbitals: ``(nao, k)`` orbitals to test, typically the active space.
        c_env: ``(nao, nenv)`` environment orbitals. Use the ones originally frozen,
            ``EmbedSCF.C_full_reidx[:, EmbedSCF.env_idx_occ]``, since those define the
            partition; the environment columns of the embedded mean field are trivially
            orthogonal to anything built by rotating its other columns.
        ovlp: AO overlap matrix.

    Returns:
        The largest absolute overlap. Round-off means the spaces are disjoint, order one
        means the active space has eaten the environment.
    """
    return float(backend.xp.abs(
        backend.xp.asarray(c_orbitals).T @ backend.xp.asarray(ovlp) @ backend.xp.asarray(c_env)).max())


def run_casci(mf, ncas, nelecas, mo_coeff=None, mo_cas_idxs=None, hcore=None):
    """Solve a CASCI on an embedded mean field, choosing orbitals by index not position.

    ``mcscf.CASCI`` would otherwise take a contiguous window around the frontier, which
    after huzinaga embedding can quietly contain an environment orbital. ``hcore`` is the
    embedded one-electron Hamiltonian *without* the projector, matching
    ``EmbedSCF.get_mo_integrals``: the projector vanishes on an environment-free CAS
    anyway, and freezing the density-dependent huzinaga operator into it would be wrong.

    Fixed orbitals are what makes that safe. The CI expansion lives entirely inside the
    active space, so it cannot reach the environment however the Hamiltonian is built,
    and the environment orbitals are unoccupied in the embedded reference so they do not
    enter the core potential either. :func:`run_casscf`, which relaxes the orbitals, has
    no such protection.

    Args:
        mf: Converged embedded mean field.
        ncas: Number of active orbitals.
        nelecas: ``(na, nb)`` active electrons.
        mo_coeff: Orbitals to choose from, defaulting to ``mf.mo_coeff``.
        mo_cas_idxs: Columns to make active. ``None`` uses the contiguous window.
        hcore: Override for the one-electron Hamiltonian, in the AO basis.

    Returns:
        The solved ``mcscf.CASCI`` object.
    """
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    try:     
        mf_cpu      = mf.to_cpu()   
        mo_coeff    = numpy.asarray(mo_coeff.get())
        if hcore is None:
            hcore_emb = numpy.asarray(mf.get_hcore().get())
        else:
            hcore_emb = numpy.asarray(hcore.get())
        if mo_cas_idxs is not None:
            mo_cas_idxs = numpy.asarray(mo_cas_idxs.get())
    except:
        mf_cpu = mf
        mo_coeff = numpy.asarray(mo_coeff)
        if hcore is None:
            hcore_emb = numpy.asarray(mf.get_hcore())
        else:
            hcore_emb = numpy.asarray(hcore)
        if mo_cas_idxs is not None:
            mo_cas_idxs = numpy.asarray(mo_cas_idxs)


    casci = mcscf.CASCI(mf_cpu, ncas, nelecas)
    casci.verbose = 0
    casci.get_hcore = lambda *args, **kwargs: hcore_emb
    if mo_cas_idxs is not None:
        mo_coeff = mcscf.addons.sort_mo(casci, mo_coeff, mo_cas_idxs, base=0)
    casci.kernel(mo_coeff)
    return casci


def run_casscf(mf, ncas, nelecas, mo_coeff=None, mo_cas_idxs=None, hcore=None,
               c_env=None, ovlp=None, env_tol=1e-6):
    """Solve a CASSCF on an embedded mean field, refusing to return a collapsed answer.

    Relaxing the orbitals removes the protection :func:`run_casci` enjoys, and after
    huzinaga embedding the default behaviour is catastrophic: the projector is density
    dependent, so ``EmbedSCF`` keeps it in ``get_fock``, while ``mcscf`` builds its own
    Hamiltonian from ``get_hcore``. The optimiser therefore never sees the projector,
    rotates the environment into the active space, and converges happily to a meaningless
    energy with believable natural occupations. On the stretched methanol of
    ``notebooks/3-active-space.ipynb`` that is a 47 hartree error reported as converged.

    So ``hcore`` must carry a projector here, unlike in :func:`run_casci`. Freezing the
    converged huzinaga operator into it is enough, since after convergence it is just a
    matrix::

        dm = mf.make_rdm1()
        fock = mf.get_hcore() + mf.get_veff(dm=dm)
        hcore = mf.get_hcore() + emb.get_huz_operator(fock, level_shift=huz_level_shift)

    A mu-shift embedding needs none of this: its projector is density independent and
    already sits in ``get_hcore``, so ``mcscf`` inherits it. Both routes agree, and both
    leak at the order of the shift rather than exactly, so pass ``c_env`` and check.

    Args:
        mf: Converged embedded mean field.
        ncas: Number of active orbitals.
        nelecas: ``(na, nb)`` active electrons.
        mo_coeff: Starting orbitals, defaulting to ``mf.mo_coeff``.
        mo_cas_idxs: Columns to make active. ``None`` uses the contiguous window.
        hcore: One-electron Hamiltonian in the AO basis, which must include a projector
            for a huzinaga embedding. Defaults to ``mf.get_hcore()``, correct only for
            mu-shift.
        c_env: Environment orbitals to check the optimised active space against. Strongly
            recommended: without it nothing here can tell a collapse from a solution.
        ovlp: AO overlap matrix, defaulting to ``mf.get_ovlp()``.
        env_tol: Largest tolerated overlap with the environment.

    Returns:
        The solved ``mcscf.CASSCF`` object.

    Raises:
        RuntimeError: If the optimised active space overlaps the environment.
    """
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    try:     
        mf_cpu      = mf.to_cpu()   
        mo_coeff    = numpy.asarray(mo_coeff.get())
        if hcore is None:
            hcore_emb = numpy.asarray(mf.get_hcore().get())
        else:
            hcore_emb = numpy.asarray(hcore.get())
        if mo_cas_idxs is not None:
            mo_cas_idxs = numpy.asarray(mo_cas_idxs.get())
        if c_env is not None:
            c_env = numpy.asarray(c_env.get())
    except:
        mf_cpu = mf
        mo_coeff = numpy.asarray(mo_coeff)
        if hcore is None:
            hcore_emb = numpy.asarray(mf.get_hcore())
        else:
            hcore_emb = numpy.asarray(hcore)
        if mo_cas_idxs is not None:
            mo_cas_idxs = numpy.asarray(mo_cas_idxs)
        if c_env is not None:
            c_env = numpy.asarray(c_env)

    
    casscf = mcscf.CASSCF(mf_cpu, ncas, nelecas)
    casscf.verbose = 0
    casscf.get_hcore = lambda *args, **kwargs: hcore_emb
    if mo_cas_idxs is not None:
        mo_coeff = mcscf.addons.sort_mo(casscf, mo_coeff, mo_cas_idxs, base=0)
    casscf.kernel(mo_coeff)

    if c_env is not None:
        ovlp = mf.get_ovlp() if ovlp is None else ovlp
        c_act = casscf.mo_coeff[:, casscf.ncore:casscf.ncore + casscf.ncas]
        leak = environment_overlap(c_act, c_env, ovlp)
        if leak > env_tol:
            raise RuntimeError(
                f"the optimised active space overlaps the environment by {leak:.2e}, so "
                f"E = {casscf.e_tot:.6f} is meaningless however converged it looks; the "
                "usual cause is a huzinaga embedding whose hcore carries no projector"
            )
    return casscf
