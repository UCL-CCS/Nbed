"""File generate H4 structures."""

import numpy as np
from nbed.utils import save_ordered_xyz_file
import os
def generate_H4_structure_dict(beta_angle, diagonal_radius=1.735) -> str:
    """Get raw xyz string of molecular geometry.

    #----------#
    | .       .|
    |   .   .  |
    |     . >B |
    |   . v .  |
    | .   a   .|
    #----------#

    B = beta_angle
    . . . = diagonal_radius

    """

    beta_rad = np.deg2rad(beta_angle)
    half_a = (np.pi-beta_rad)/2
    bottom_length = 2 * (diagonal_radius*np.sin(half_a))
    top_lenght = 2* (diagonal_radius*np.sin(beta_rad/2))

    xyz_postions =[
        [0.0, 0.0, 0.0],                  # bottom left
        [bottom_length, 0.0, 0.0],        # bottom right
        [bottom_length, top_lenght, 0.0], # top right
        [0.0, top_lenght, 0.0]            # top left

    ]
    struct_dict = {}
    for ind, H_xyz in enumerate(xyz_postions):
        struct_dict[ind] = ('H', tuple(H_xyz))

    return struct_dict


if __name__ == '__main__':
    cwd = os.getcwd()
    beta_angle_list= list(range(85,95+1))
    for b_angle in beta_angle_list:
        H4_struct_dict = generate_H4_structure_dict(b_angle, diagonal_radius=1.735)

        file_name = f'H4_beta_{b_angle}_bottom_bottom_top_top_order'
        save_ordered_xyz_file(file_name,
                              struct_dict=H4_struct_dict,
                              active_atom_inds=[0,1,2,3],
                              save_location=cwd
                              )

        file_name = f'H4_beta_{b_angle}_bottom_top_bottom_top_order'
        save_ordered_xyz_file(file_name,
                              struct_dict=H4_struct_dict,
                              active_atom_inds=[0,2,1,3],
                              save_location=cwd
                              )