def test_non_ortho_env_projector():
    """Return the non-ortho environment projector"""
    # 3. Define projector in standard (non-orthogonal basis)
    # projector = s_half @ ortho_env_projector @ s_half

    # logger.info(
    #     f"""Are subsystem B (env) projected onto themselves in ORTHO basis: {
    #         np.allclose(projector @ c_loc_occ_and_virt[:, enviro_MO_inds],
    #         c_loc_occ_and_virt[:, enviro_MO_inds])}"""
    # )

    # logger.info(
    #     f"""Is subsystem A traced out  in ORTHO basis?: {
    #         np.allclose(projector@c_loc_occ_and_virt[:, active_MO_inds],
    #         np.zeros_like(c_loc_occ_and_virt[:, active_MO_inds]))}"""
    # )
    return projector
