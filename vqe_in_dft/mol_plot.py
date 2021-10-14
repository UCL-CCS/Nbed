import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import os
from pyscf.tools import cubegen
import numpy as np
import py3Dmol
# rdkit import is weird and requires each part to be imported before going deeper
import rdkit
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from pyscf import gto

def Draw_molecule(xyz_string: str, width: int=400, height: int=400, jupyter_notebook: bool=False) -> py3Dmol.view:
	"""
	Draw molecule from xyz string

	Args:
		xyz_string (str): xyz string of molecule
		width (int): width of image
		height (int): Height of image
		jupyter_notebook (bool): Whether to allow plotting in Jupyter notebooks

	Returns:
		view (py3dmol.view object). Run view.show() method to print molecule.

	"""
	if jupyter_notebook is True:
		rdkit.Chem.Draw.IPythonConsole.ipython_3d = True  # enable py3Dmol inline visualization

	view = py3Dmol.view(width=width, height=height)
	view.addModel(xyz_string, "xyz")
	view.setStyle({'stick':{}})
	view.zoomTo()
	return view

def Draw_cube_orbital(PySCF_mol_obj: gto.Mole, xyz_string: str, C_matrix: np.ndarray, index_list: List[int], 
					  width: int=400, height: int=400, jupyter_notebook: bool=False) -> List:
	"""
	Draw orbials given a C_matrix (columns contain molecular orbs) and xyz string of molecule.
	This function writes orbitals to tempory cube files then deletes them.
	For standard use the C_matrix input should be C_matrix optimized by a self consistent field (SCF) run.

	Args:
		PySCF_mol_obj (pyscf.mol): PySCF mol object. Required for pyscf.tools.cubegen function
		xyz_string (str): xyz string of molecule
		C_matrix (np.array): Numpy array of molecular orbitals (columns are MO). 
		index_list (List): List of MO indices to plot
		width (int): width of image
		height (int): Height of image
		jupyter_notebook (bool): Whether to allow plotting in Jupyter notebooks

	Returns:
		plotted_orbitals (List): List of plotted orbitals (py3Dmol.view) ordered the same way as in index_list

	"""
	if jupyter_notebook is True:
		rdkit.Chem.Draw.IPythonConsole.ipython_3d = True  # enable py3Dmol inline visualization

	if not set(index_list).issubset(set(range(C_matrix.shape[1]))):
		raise ValueError('list of MO indices to plot is outside of C_matrix column indices')

	plotted_orbitals = []
	for index in index_list:
		File_name = f'temp_MO_orbital_index{index}.cube'
		cubegen.orbital(PySCF_mol_obj, File_name, C_matrix[:, index])
		
		view = py3Dmol.view(width=width, height=height)
		view.addModel(xyz_string, "xyz")
		view.setStyle({'stick':{}})
		
		with open(File_name, 'r') as f:
			view.addVolumetricData(f.read(), "cube", {'isoval': -0.02, 'color': "red", 'opacity': 0.75})
		with open(File_name, 'r') as f2:
			view.addVolumetricData(f2.read(), "cube", {'isoval': 0.02, 'color': "blue", 'opacity': 0.75})
		
		plotted_orbitals.append(view.zoomTo())
		os.remove(File_name) # delete file once orbital is drawn
	
	return plotted_orbitals

