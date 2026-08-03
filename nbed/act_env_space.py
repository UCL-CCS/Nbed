
"""
helper functions for selecting which MO indices should be
1. ACTIVE
2. ENVIRONMENT
to defined embedded SCF objects.
"""
import numpy as np


##################################################################################
###### how to select active and environment orbitals (NOT an active space!) ######
###################################################p###############################

def lowdin_populations(mol, mo_coeff, atom_indices, drop_core_1s=True):
    """Fraction of every MO sitting on each target atom.

    Symmetrically orthogonalises the AO basis with S^(1/2) so the squared
    coefficients form a genuine partition of the orbital, then sums them over
    the AOs of each atom.

    Args:
        mol: Molecule supplying the overlap matrix and AO layout.
        mo_coeff: ``(nao, nmo)`` MO coefficients in the AO basis.
        atom_indices: 0-based indices of the target atoms.
        drop_core_1s: Exclude 1s AOs of elements beyond helium.

    Returns:
        Tuple of ``per_atom`` with shape ``(n_targets, nmo)`` and ``total``
        with shape ``(nmo,)``, the summed target-atom character of each MO.

    Raises:
        IndexError: If an atom index is outside the molecule.
        ValueError: If an atom has no AOs left after dropping cores.
    """
    ovlp = mol.intor("int1e_ovlp")
    evals, evecs = np.linalg.eigh(ovlp)
    s_half = (evecs * np.sqrt(np.clip(evals, 0.0, None))) @ evecs.T
    c_orth = s_half @ mo_coeff

    ao_slices = mol.aoslice_by_atom()
    ao_labels = mol.ao_labels()

    per_atom = []
    for index in atom_indices:
        if not 0 <= index < mol.natm:
            raise IndexError(
                f"atom index {index} is outside the molecule, "
                f"which has {mol.natm} atoms"
            )
        # a 1s is only core beyond helium; for H and He it is the valence shell
        has_core = mol.atom_charge(index) > 2
        aos = [
            mu
            for mu in range(ao_slices[index, 2], ao_slices[index, 3])
            if not (drop_core_1s and has_core and " 1s" in ao_labels[mu])
        ]
        if not aos:
            raise ValueError(
                f"atom {index} has no AOs left after dropping cores; "
                "pass drop_core_1s=False"
            )
        per_atom.append(np.einsum("mi,mi->i", c_orth[aos], c_orth[aos]))

    per_atom = np.array(per_atom)
    return per_atom, per_atom.sum(axis=0)


def orbital_spread(mol, mo_coeff):
    """Spatial extent sqrt(<r^2> - <r>^2) of each MO, in Bohr.

    Tells a compact valence antibonding orbital apart from a diffuse
    Rydberg-like one, which matters because both can score highly on the target
    atoms. In a large basis, valence orbitals come out several times tighter
    than the median virtual.

    Args:
        mol: Molecule supplying the dipole and r^2 integrals.
        mo_coeff: ``(nao, nmo)`` MO coefficients.

    Returns:
        Array of RMS spreads in Bohr; larger means more diffuse.
    """
    dip = mol.intor("int1e_r").reshape(3, mol.nao, mol.nao)
    r2 = mol.intor("int1e_r2")
    r_exp = np.einsum("mi,xmn,ni->xi", mo_coeff, dip, mo_coeff)
    r2_exp = np.einsum("mi,mn,ni->i", mo_coeff, r2, mo_coeff)
    return np.sqrt(np.maximum(r2_exp - np.einsum("xi,xi->i", r_exp, r_exp), 0.0))


def describe_orbital(mol, atom_indices, per_atom, index, cutoff=0.05):
    """Render one orbital's atom composition as a readable string.

    Args:
        mol: Molecule used to look up element symbols.
        atom_indices: 0-based target atom indices, matching ``per_atom`` rows.
        per_atom: ``(n_targets, nmo)`` populations.
        index: 0-based MO index.
        cutoff: Only report atoms above this population.

    Returns:
        A string like ``"O0:0.52 C2:0.08"``, or an empty string if no atom
        clears the cutoff.
    """
    return " ".join(
        f"{mol.atom_symbol(atom)}{atom}:{per_atom[row, index]:.2f}"
        for row, atom in enumerate(atom_indices)
        if per_atom[row, index] > cutoff
    )


def select_act_env_space(mf, atom_indices, n_occ_active, n_vir_active=None,
                        mo_coeff=None, drop_core_1s=True, max_spread=None):
    """Partition a mean field into a fragment on given atoms and its environment.

    Occupied and virtual orbitals are ranked separately by target-atom
    population, so a strongly localised high virtual beats a weakly localised
    one nearer the frontier.

    Args:
        mf: Converged mean-field object with ``mo_coeff`` and ``mo_occ``.
        atom_indices: 0-based indices of the atoms defining the fragment.
        n_occ_active: How many occupied orbitals to assign to the fragment.
            This fixes the subsystem electron count.
        n_vir_active: How many virtuals to add to the fragment block, which
            fixes the CAS size for a post-embedding solver. Defaults to
            ``n_occ_active``.
        mo_coeff: Orbitals to partition, defaulting to ``mf.mo_coeff``. Pass
            localised orbitals here to get a cleaner fragment/environment
            split; the rotation must stay within the occupied and virtual
            blocks separately, so that ``mf.mo_occ`` still describes column
            ``i``, and it must be unitary, so that the total density is
            unchanged.
        drop_core_1s: Exclude core 1s AOs from the population.
        max_spread: Reject orbitals more diffuse than this, in Bohr.

    Returns:
        TODO

    Raises:
        ValueError: If the occupations are fractional, or if either half has
            too few eligible orbitals.
    """
    mol = mf.mol
    mo_coeff = np.asarray(mf.mo_coeff if mo_coeff is None else mo_coeff)
    if mo_coeff.ndim == 3:
        raise ValueError(
            "unrestricted orbitals are not supported; "
            "pass a restricted (RHF/ROHF/RKS) mean field"
        )

    mo_occ = np.asarray(mf.mo_occ)
    if not np.allclose(mo_occ, np.rint(mo_occ)):
        raise ValueError(
            "fractional occupations: embedding needs an integer number of "
            "electrons on each subsystem"
        )
    mo_occ = np.rint(mo_occ).astype(int)

    n_vir_active = n_occ_active if n_vir_active is None else n_vir_active
    
    _, population = lowdin_populations(mol, mo_coeff, atom_indices, drop_core_1s)
    spread = orbital_spread(mol, mo_coeff)

    eligible = np.ones_like(population, dtype=bool)
    if max_spread is not None:
        eligible &= spread <= max_spread

    occ_pool = np.where((mo_occ > 0) & eligible)[0]
    vir_pool = np.where((mo_occ == 0) & eligible)[0]
    if len(occ_pool) < n_occ_active or len(vir_pool) < n_vir_active:
        raise ValueError(
            f"asked for {n_occ_active} occupied and {n_vir_active} virtual "
            f"orbitals but only {len(occ_pool)} and {len(vir_pool)} are "
            "eligible; relax max_spread or shrink the fragment"
        )
    
    pick_occ = np.sort(occ_pool[np.argsort(-population[occ_pool])[:n_occ_active]])
    pick_vir = np.sort(vir_pool[np.argsort(-population[vir_pool])[:n_vir_active]])
    active_idxs = np.concatenate([pick_occ, pick_vir])
    env_idxs = np.setdiff1d(np.arange(mol.nao), active_idxs)
    return active_idxs, env_idxs, population, spread