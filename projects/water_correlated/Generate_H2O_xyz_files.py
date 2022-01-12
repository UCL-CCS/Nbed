
"""File to generate H2O" structures."""

import numpy as np
from nbed.utils import save_ordered_xyz_file
import os
def generate_H2O_from_reference(R_length=0.8) -> str:
    """Build raw xyz string of water for a single changine O----H bond length.
    Note this changing H-bond length is kept at the TOP of xyz file.
    """

    water_xyz_list = [
        ('H', [0.7493682, 0.0000000, 0.2770822]),
        ('O', [0.0000000, 0.0000000, 0.0000000]),
        ('H', [-0.7493682, 0.0000000, 0.2770822])
    ]
    Hyp = np.linalg.norm(water_xyz_list[0][1])
    Opp = water_xyz_list[0][1][2]
    angle = np.arcsin(Opp / Hyp)

    x_pos = np.around(R_length*np.cos(angle), 7)
    z_pos = np.around(R_length*np.sin(angle), 7)

    water_xyz_list = [
        ('H', [x_pos, 0.0000000, z_pos]),
        ('O', [0.0000000, 0.0000000, 0.0000000]),
        ('H', [-0.7493682, 0.0000000, 0.2770822])
    ]
    struct_dict = dict(zip(range(3), water_xyz_list))

    return struct_dict


if __name__ == '__main__':
    cwd = os.getcwd()
    # output_dir = os.path.join(cwd, 'H2O_structures')
    # # Create target Directory if it doesn't exist
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)

    R_length_list = [0.5, 0.8, 1, 1.5, 2, 3, 4, 5]
    for R in R_length_list:
        H2O_struct_dict = generate_H2O_from_reference(R_length=R)

        file_name = f'H2O_{int(R*10)}'
        save_ordered_xyz_file(file_name,
                              struct_dict=H2O_struct_dict,
                              active_atom_inds=[0,1,2],
                              save_location=cwd
                              )