
import numpy as np
from pyscf import scf

def get_embedding_potential(global_scf_cheap, dm_full, dm_act, ovlp_ao):

    # v_eff_emb = global_scf_cheap.get_veff(dm=dm_full) - global_scf_cheap.get_veff(dm=dm_act)

    ## note in get_roothaan_fock we are not including the core Hamiltonian... we just want the 2e- terms
    veff_glob = global_scf_cheap.get_veff(dm=dm_full) 
    G_glob = scf.rohf.get_roothaan_fock((veff_glob[0],veff_glob[1]), dm_full, ovlp_ao)

    veff_act = global_scf_cheap.get_veff(dm=dm_act) 
    G_act = scf.rohf.get_roothaan_fock((veff_act[0],veff_act[1]), dm_act, ovlp_ao)
    G_EMB = G_glob - G_act

    return G_EMB


def get_mu_projector(C_env_occ:np.array, S:np.array):
    """
    mu value NOT included!

    This can be used to do DFT-in-DFT iterations (C_matrix can be updated)

    """
    P_env_mu = (S @ C_env_occ @ C_env_occ.T @ S)
    return P_env_mu


def get_huzinaga_projector(global_scf_expensive, C_env_occ:np.array, S:np.array, dm_act:np.array, v_eff_emb:np.array):
    """
    This CANNOT be used to do DFT-in-DFT iterations (C_matrix must reamin fixed [cannot be updated once chosen])
    As otherwise block is not zeroed out during scf loop. 
    """

    F_emb = global_scf_expensive.get_fock(dm=dm_act) + v_eff_emb #<--- adding veff here (it is needed in projector def, otherwise not all values are zeroed out!)

    # projector of F onto enviro positions (this is then subtracted from Fock matrix to zero the env positions out)
    huz_proj  = F_emb @ C_env_occ @ C_env_occ.T @ S


    ## note this is the embedded fock matrix
    # F_projected_emb = F_emb - proj_Femb_S # <--- this is the embedded fock matrix to use!

    ## F_projected_emb should be block diagonal w.r.t: C_act and C_env
    ## while the orbitals may overlap... the block diagonal structure makes the eigenstates (MOs) distinct!
    return F_emb, huz_proj

