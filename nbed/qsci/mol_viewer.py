import py3Dmol
from pyscf import gto, mcscf, ao2mo
from typing import List, Optional 

def mol_viewer(pyscf_mol:gto.Mole, active_atm_idxs: List[int], width:Optional[int]=800, height:Optional[int]=600,
               show_atm_indices:Optional[bool]=False, active_color:Optional[str]="purple", active_opacity:Optional[float]=0.7, 
               active_color_radius:Optional[float]=0.55) -> py3Dmol.view:
    """
    Draw molecule and highligh active region in selected colour.

    Args:
        pyscf_mol (gto.Mole): PySCF molecule object
        active_atm_idxs (list): list of atom indices that are active (matches up with xyz order)
        width (int): pixel width of viewer box
        height (int):pixel height of viewer box
        show_atm_indices (bool): whether to show atom indices
        active_color (str): what colour to shade active region
        active_opacity (float): how dense the active region shading is
        active_color_radius (float): how large the radius of the spheres colouring the active region are
    Returns
        view (py3Dmol.view): py3Dmol view object. Use .show() method to view system

    """
    # Create a 3Dmol view object
    view = py3Dmol.view(width=width, height=height)

    # Set background color
    view.setBackgroundColor('white')

    # Add the XYZ data to the viewer
    view.addModel(pyscf_mol.tostring(format="xyz"), 'xyz')

    # Apply stick representation
    view.setStyle({'stick': {'colorscheme':'cyanCarbon'}, 
                    })

    view.addStyle({'serial':active_atm_idxs},{"sphere": {"color":   active_color,
                                                        "radius":  active_color_radius,
                                                        'opacity': active_opacity}} 
                                                        )
    if show_atm_indices is True:
        for atm_idx in range(pyscf_mol.natm):
            atom_coord = pyscf_mol.atom_coord(atm_idx, unit=pyscf_mol.unit) # unit gives Bohr/Angstrom
            view.addLabel(
                str(atm_idx),
                {
                    'position': {'x': atom_coord[0], 'y': atom_coord[1], 'z': atom_coord[2]},
                    'fontSize': 14,
                    'fontColor': 'white',
                    'backgroundColor': 'grey',
                    'inFront': True,         # keep labels readable
                    'showBackground': True,  # draw a box behind text
                }
            )

    return view
