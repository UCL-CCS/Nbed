
import numpy as np


def get_embedding_potential(global_scf_cheap, dm_full, dm_act):
    v_eff_emb = global_scf_cheap.get_veff(dm=dm_full) - global_scf_cheap.get_veff(dm=dm_act)
    return v_eff_emb


def get_mu_projector(C_env_occ:np.array, Sao:np.array):
    """
    mu value NOT included!

    This can be used to do DFT-in-DFT iterations (C_matrix can be updated)

    """
    P_env_mu = (Sao @ C_env_occ @ C_env_occ.T @ Sao)
    return P_env_mu


def get_huzinaga_projector(C_env_occ:np.array, Sao:np.array):
    """
    """
    # Build env projector (this form is idempotent!)
    P_env_huz = C_env_occ @ C_env_occ.T @ Sao

    return P_env_huz
