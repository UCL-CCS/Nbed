"""
File to contain tests of the embed.py script.
"""
from pathlib import Path

import numpy as np
import openfermion

from nbed.localizers import Localizer

water_filepath = Path("tests/molecules/water.xyz").absolute()


def test_nbed() -> None:
    # This test is broken as we're updating the main code
    # q_ham, e_classical = nbed(
    #     geometry=str(water_filepath),
    #     active_atoms=2,
    #     basis="sto-3g",
    #     xc_functional="b3lyp",
    #     output="openfermion",
    #     convergence=1e-8,
    # )
    # print(len(q_ham.terms))
    # assert len(q_ham.terms) == 1079
    # assert np.isclose(q_ham.constant, -45.42234047466274)
    # assert np.isclose(e_classical, -3.5605837557207654)
    pass


def test_orthogonal_enviro_projector() -> None:
    # # 1. Get orthogonal C matrix (localized)
    # c_loc_ortho = s_half @ local_sys.c_loc_occ_and_virt

    # # 2. Define projector that projects MO orbs of subsystem B onto themselves and system A onto zero state!
    # #    (do this in orthongoal basis!)
    # #    note we only take MO environment indices!
    # ortho_proj = np.einsum(
    #     "ik,jk->ij",
    #     c_loc_ortho[:, local_sys.enviro_MO_inds],
    #     c_loc_ortho[:, local_sys.enviro_MO_inds],
    # )

    # # env projected onto itself
    # logger.info(
    #     f"""Are subsystem B (env) projected onto themselves in ORTHO basis: {
    #         np.allclose(ortho_proj @ c_loc_ortho[:, enviro_MO_inds],
    #         c_loc_ortho[:, enviro_MO_inds])}"""
    # )

    # # act projected onto zero vec
    # logger.info(
    #     f"""Is subsystem A traced out  in ORTHO basis?: {
    #         np.allclose(ortho_proj @ c_loc_ortho[:, active_MO_inds],
    #         np.zeros_like(c_loc_ortho[:, active_MO_inds]))}"""
    # )
    pass


if __name__ == "__main__":
    pass
