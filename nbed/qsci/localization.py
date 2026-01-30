import numpy as np

####################################################################################
######################     Methods to choose localized subsystems      #############
####################################################################################

# Please note that active_ao_idx from PySCF is obtained via:

# ao_slice = mol.aoslice_by_atom()
## AO slice info: [shl_start, shl_stop, ao_start, ao_stop]
## active AO indices
# active_ao_idx = np.hstack([
#     np.arange(ao_slice[i, 2], ao_slice[i, 3])
#     for i in range(n_active_atoms)
# ])
### this could be modified to make i go over active atom indices (rather than assuming hte first n are active!)


def mulliken_per_orbital(active_ao_idx: np.array, S: np.array, C: np.array) -> np.array:
    """
    Instead of the total density, compute per-orbital Mulliken population.
    In full this would be:

    q_{i, mu} = Tr(S |MO_{i}> <MO_{i}|) = Tr(S @ C[:,i] @ C[:,i].conj().T)
        where i and mu are the MO idxs and mu are the active AO idxs.

    given the full output matrix, then we need to sum over the active AO idxs of the [Q]_{i,mu} (where AO is the 2nd idx)
    The resulting matrix is a vector of length i (number of MOs). In different basis, the sum of this matrix should be the same, but have different values!
    These values can then be thresholded on.

    ### manual mode:
    Q = np.array([np.diag(S @ np.outer(C[:, i], C[:, i]))for i in range(C.shape[1])])
    OR
    Q = np.einsum('vm,vi,mi->im', S, C, C, optimize=True) 

    then do:
    mull_per_orb = Q[:, active_ao_idx].sum(axis=1)

    The code below does this, but in a more efficient way (doesn't require calculating all these steps and instead does it more directly)
    per_orbital_active[i] = Σ_{μ ∈ active_ao_idx} (S·C)_{μi} · C_{μi}
    This is the Mulliken population of orbital i summed over the active AOs.

    Args:
        active_ao_idx (np.array): list of active AO indices
        S (np.array): AO overlap matrix
        C (np.array): molecular orbital cofficient matrix (columns give MOs)
    Returns:
        per_orbital_active (np.array): Mulliken population of each orbital i summed over active AOs
    """
    # Compute: diag_part = (S @ C) * C, then sum over active AOs
    SC = S @ C  # (nao_total, n_mo)
    per_orbital_active = (SC[active_ao_idx, :] * C[active_ao_idx, :]).sum(axis=0)
    return per_orbital_active 


def direct_projector(active_ao_idx: np.array, S: np.array, C: np.array):
    """
    A direct way of projecting MOs onto only the active AOs (allowing importance to be determined)

    PLEASE NOTE:
    C_proj = P @ C
    Rows corresponding to inactive AOs are not necessarily zero
    --> because active AOs overlap with inactive ones (due to non-orthogonal basis!)
    But physically, C_proj represents something that lies entirely in the active AO span
    If you want coordinates strictly zero outside active AOs, you must switch to an orthogonal AO basis first (e.g. Löwdin)... but this changes C which we do NOT want to do!

    C_proj is a matrix whose columns are the projected MOs.
    where are each the S-orthogonal projection of the MO onto the span of active AOs.
    This means:
    1. the projected MOs lives entirely in the active AO subspace (physically).
    2. It is the closest possible vector (in the AO-overlap norm) to the original MO that uses only active-atom AOs.

    P @ C gives you the projected orbital itself
    C_A = C[active_ao_idx, :]
    here C_A gives you how much of the orbital lives on active AOs

    # Please note that:
    C_proj does not give an orthonormal MO basis
    different columns of C_proj are not orthogonal to each other
    C_proj does not diagonalize the Fock matrix


    Args:
        active_ao_idx (np.array): list of active AO indices
        S (np.array): AO overlap matrix
        C (np.array): molecular orbital cofficient matrix (columns give MOs)
    Returns:
        overlap_psi_and_psi_act_ao (np.array): Gives overlap bween [MO-i and MO-i-projected-onto-active-AO-sites] (value between 0 and 1)!
    """
    S_inv = np.linalg.pinv(S)
    S_allA = S[:, active_ao_idx]        # (nao, n_active_aos)
    S_Aall = S[active_ao_idx, :]        # (n_active, nao)
    S_AA   = S[np.ix_(active_ao_idx, active_ao_idx)]
    P = S_inv @ S_allA @ np.linalg.pinv(S_AA) @ S_Aall

    ### debugging checks!
    # assert np.allclose(P @ P, P), "projector is NOT idempotent"
    # For an orthogonal projector in an S-metric, you need: P^{†}S=SP:
    # assert np.allclose(P.conj().T @ S, S @ P, atol=1e-10), "P and I-P do not define orthogonal subspaces."

    # Q = np.eye(P.shape[0]) - P
    overlap_psi_and_psi_act_ao = []
    for i in range(C.shape[0]):

        psi = C[:, i]

        psi_proj  = P @ psi # = P @ psi
        # psi_ortho = Q @ psi # = (I-P) @ psi
        
        # ### debugging checks!
        # # Check: ψ = Pψ + (I-P)ψ
        # psi_reconstructed = psi_proj + psi_ortho
        # assert np.allclose(psi, psi_reconstructed), "projection WRONG "

        # # check sum of overlaps is 1
        overlap_act = psi.conj().T @ S @ psi_proj
        # overlap_env = psi.conj().T @ S @ psi_ortho
        # assert np.isclose(overlap_act+overlap_env, 1), "overlap should be 1 for valid C!"

        overlap_psi_and_psi_act_ao.append(overlap_act)

    return np.array(overlap_psi_and_psi_act_ao)


# for direct_projector can do the following checks:
# If ϕ is any linear combination of active AOs, then Pϕ = ϕ
## we can check as:
# rng = np.random.default_rng(1) # random
# # arbitrary vector
# v = rng.standard_normal(nao)

# # decompose
# vA = P @ v
# v_perp = v - vA

## 1) v = vA + v_perp
# print(np.allclose(v, vA + v_perp))

## 2) (I-P)v = v_perp
# assert np.allclose(v_perp, (np.eye(P.shape[0]) - P) @ v)

## 3) v_perp is S-orthogonal to all active AOs
# print(np.allclose(S_Aall @ v_perp, np.zeros(S_Aall.shape[0])))

## 4) vA is unchanged by P (Pϕ = ϕ) for active AOs!
# print(np.allclose(P @ vA, vA))