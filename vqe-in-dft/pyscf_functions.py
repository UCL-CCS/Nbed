from pyscf import gto, dft, scf, lib, cc, fci
from copy import deepcopy as copy
import numpy as np
import scipy as sp


def Generate_supersystem_and_indices(mol_A, mol_B):
    """
    Concatenate TWO subsystems into supersystem pyscf object!
    (https://sunqm.github.io/pyscf/_modules/pyscf/gto/mole.html#conc_mol)

    Generate matrix for convecting between subsystems and the supersystem

    returns:
        supersystem (pyscf.mol): supersystem pyscf objet
        sub2sup (np.array): Numpy array of indices for subsystem
    """
    supersystem = gto.mole.conc_mol(mol_A, mol_B)

    list_of_subsystems = [mol_A, mol_B]
    nao = np.array([subsystem.mol.nao_nr() for subsystem in list_of_subsystems])
    nssl = [None for i in range(len(list_of_subsystems))]

    for i, sub in enumerate(list_of_subsystems):
        nssl[i] = np.zeros(sub.mol.natm, dtype=int)
        for j in range(sub.mol.natm):
            ib_t = np.where(sub.mol._bas.transpose()[0] == j)[0]
            i_b = ib_t.min()
            ie_t = np.where(sub.mol._bas.transpose()[0] == j)[0]
            i_e = ie_t.max()
            i_r = sub.mol.nao_nr_range(i_b, i_e + 1)
            i_r = i_r[1] - i_r[0]
            nssl[i][j] = i_r

        assert nssl[i].sum() == sub.mol.nao_nr(), "naos not equal!"

    nsl = np.zeros(supersystem.mol.natm, dtype=int)
    for i in range(supersystem.mol.natm):
        i_b = np.where(supersystem.mol._bas.transpose()[0] == i)[0].min()
        i_e = np.where(supersystem.mol._bas.transpose()[0] == i)[0].max()
        i_r = supersystem.mol.nao_nr_range(i_b, i_e + 1)
        i_r = i_r[1] - i_r[0]
        nsl[i] = i_r

    assert nsl.sum() == supersystem.mol.nao_nr(), "naos not equal!"

    sub2sup = [None for i in range(len(list_of_subsystems))]
    for i, sub in enumerate(list_of_subsystems):
        sub2sup[i] = np.zeros(nao[i], dtype=int)
        for j in range(sub.mol.natm):
            match = False
            c_1 = sub.mol.atom_coord(j)
            for k in range(supersystem.mol.natm):
                c_2 = supersystem.mol.atom_coord(k)
                dist = np.dot(c_1 - c_2, c_1 - c_2)
                if dist < 0.0001:
                    match = True
                    i_a = nssl[i][0:j].sum()
                    j_a = i_a + nssl[i][j]
                    # ja = ia + nsl[b]
                    i_b = nsl[0:k].sum()
                    j_b = i_b + nsl[k]
                    # jb = ib + nssl[i][a]
                    sub2sup[i][i_a:j_a] = range(i_b, j_b)

            assert match, 'no atom match!'

    return supersystem, sub2sup

