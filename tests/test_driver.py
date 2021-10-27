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
    pass


def test_basis_change()
    """Construct operator to change basis.

    Get operator that changes from standard canonical orbitals (C_matrix standard) to
    localized orbitals (C_matrix_localized)

    Args:
        pyscf_scf (StreamObject): PySCF molecule object
        c_all_localized_and_virt (np.array): C_matrix of localized orbitals (includes occupied and virtual)
        sanity_check (bool): optional flag to check if change of basis is working properly

    Returns:
        matrix_std_to_loc (np.array): Matrix that maps from standard (canonical) MOs to localized MOs
    """
    # s_mat = pyscf_scf.get_ovlp()
    # s_half = sp.linalg.fractional_matrix_power(s_mat, 0.5)

    # # find orthogonal orbitals
    # ortho_std = s_half @ pyscf_scf.mo_coeff
    # ortho_loc = s_half @ c_all_localized_and_virt

    # # Build change of basis operator (maps between orthonormal basis (canonical and localized)
    # unitary_ORTHO_std_onto_loc = np.einsum("ik,jk->ij", ortho_std, ortho_loc)

    # s_neg_half = sp.linalg.fractional_matrix_power(s_mat, -0.5)

    # # move back into non orthogonal basis
    # matrix_std_to_loc = s_neg_half @ unitary_ORTHO_std_onto_loc @ s_half

    # if sanity_check:
    #     if np.allclose(unitary_ORTHO_std_onto_loc @ ortho_loc, ortho_std) is not True:
    #         raise ValueError(
    #             "Change of basis incorrect... U_ORTHO_std_onto_loc*C_ortho_loc !=  C_ortho_STD"
    #         )

    #     if (
    #         np.allclose(
    #             unitary_ORTHO_std_onto_loc.conj().T @ unitary_ORTHO_std_onto_loc,
    #             np.eye(unitary_ORTHO_std_onto_loc.shape[0]),
    #         )
    #         is not True
    #     ):
    #         raise ValueError("Change of basis (U_ORTHO_std_onto_loc) is not Unitary!")

    # if sanity_check:
    #     if (
    #         np.allclose(
    #             matrix_std_to_loc @ c_all_localized_and_virt, pyscf_scf.mo_coeff
    #         )
    #         is not True
    #     ):
    #         raise ValueError(
    #             "Change of basis incorrect... U_std*C_std !=  C_loc_occ_and_virt"
    #         )

    pass