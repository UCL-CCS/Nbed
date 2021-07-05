"""
Simulator class to run embedding with fragments
"""

import numpy as np

from ..fragmentation.fragment_pair import FragmentPair

class Simulator:
    "Runs embedding simulations"

    def __init__(self, fragments: FragmentPair) -> None:
        self._frags = fragments

    def run(self) -> None:
        "Function to contain and run simulations."
        raise NotImplementedError

    def _absolute_localisation(self) -> None:
        "Perform simulation with absolute localisation."
        pass

    def ao_to_mo(self, mol, fragments: FragmentPair) -> np.ndarray:
        """
        Generate the translation matrix between subsystem to supersystem.

        Parameters
        ----------
        mol : Mole
            Full system Mole object.
        fragments : FragmentPair
            Pair of fragments which comprise the full system.

        Returns
        -------
        np.ndarray
            Array values for transformation matrix.
        """

        # Get the total number of atomic orbitals.
        nao = np.array([frag.mol.nao_nr() for frag in fragments])
        # Length two array as we have two fragments
        nssl = [None] * 2

        # Calculate values of nssl: 
        for i, sub in enumerate(fragments):
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

        nsl = np.zeros(mol.natm, dtype=int)
        for i in range(mol.natm):
            i_b = np.where(mol._bas.transpose()[0] == i)[0].min()
            i_e = np.where(mol._bas.transpose()[0] == i)[0].max()
            i_r = mol.nao_nr_range(i_b, i_e + 1)
            i_r = i_r[1] - i_r[0]
            nsl[i] = i_r

        assert nsl.sum() == mol.nao_nr(), "naos not equal!"

        ao_to_mo = [None] * len(fragments)
        for i, sub in enumerate(fragments):
            ao_to_mo[i] = np.zeros(nao[i], dtype=int)
            for j in range(sub.mol.natm):
                match = False
                c_1 = sub.mol.atom_coord(j)
                for k in range(mol.natm):
                    c_2 = mol.atom_coord(k)
                    dist = np.dot(c_1 - c_2, c_1 - c_2)
                    if dist < 0.0001:
                        match = True
                        i_a = nssl[i][0:j].sum()
                        j_a = i_a + nssl[i][j]
                        #ja = ia + nsl[b]
                        i_b = nsl[0:k].sum()
                        j_b = i_b + nsl[k]
                        #jb = ib + nssl[i][a]
                        ao_to_mo[i][i_a:j_a] = range(i_b, j_b)

                assert match, 'no atom match!'

        return ao_to_mo

    def update_fock(self, diis=True) -> bool:
        """Updates the full system fock matrix.

        Parameters
        ----------
        diis : bool
            Whether to use the diis method.
        """

        # Optimize: Rather than recalc. the full V, only calc. the V for changed densities.
        # get 2e matrix
        num_rank = self.mol.nao_nr()
        dmat = [np.zeros((num_rank, num_rank)), np.zeros((num_rank, num_rank))]
        sub_unrestricted = False
        a2m = self.ao_to_mo

        for i, sub in enumerate(self.fragments):
            if sub.unrestricted:
                sub_unrestricted = True

            dmat[0][np.ix_(a2m[i], a2m[i])] += (sub.env_dmat[0])
            dmat[1][np.ix_(a2m[i], a2m[i])] += (sub.env_dmat[1])

        if self.fs_unrestricted or sub_unrestricted:
            self.emb_vhf = self.os_scf.get_veff(self.mol, dmat)
            self.fock = self.os_scf.get_fock(h1e=self.hcore, vhf=self.emb_vhf, dm=dmat)
        elif self.mol.spin != 0:
            self.emb_vhf = self.fs_scf.get_veff(self.mol, dmat)
            temp_fock = self.fs_scf.get_fock(h1e=self.hcore, vhf=self.emb_vhf, dm=dmat)
            self.fock = [temp_fock, temp_fock]
        else:
            dmat = dmat[0] + dmat[1]
            self.emb_vhf = self.fs_scf.get_veff(self.mol, dmat)
            temp_fock = self.fs_scf.get_fock(h1e=self.hcore, vhf=self.emb_vhf, dm=dmat)
            self.fock = [temp_fock, temp_fock]

        if (not self.ft_diis is None) and diis:
            if self.fs_unrestricted or sub_unrestricted:
                new_fock = self.ft_diis.update(self.fock)
                self.fock[0] = new_fock[0]
                self.fock[1] = new_fock[1]
            else:
                new_fock = self.ft_diis.update(self.fock[0])
                self.fock[0] = new_fock
                self.fock[1] = new_fock

        #Add the external potential to each fock.
        self.fock[0] += self.ext_pot[0]
        self.fock[1] += self.ext_pot[1]
        for i, sub in enumerate(self.fragments):
            sub_fock_0 = self.fock[0][np.ix_(a2m[i], a2m[i])]
            sub_fock_1 = self.fock[1][np.ix_(a2m[i], a2m[i])]
            sub.emb_fock = [sub_fock_0, sub_fock_1]

        return True

    def update_proj_pot(self) -> bool:
        # currently updates all at once. Can mod. to update one subsystem; likely won't matter.
        """Updates the projection potential of the system.
        """
        a2m = self.ao_to_mo
        for i, sub_a in enumerate(self.fragments):
            num_rank_a = sub_a.mol.nao_nr()
            proj_op = [np.zeros((num_rank_a, num_rank_a)), np.zeros((num_rank_a, num_rank_a))]

            # cycle over all other fragments
            for j, sub_b in enumerate(self.fragments):
                if j == i:
                    continue

                sub_b_dmat = sub_b.env_dmat
                smat_ab = self.smat[np.ix_(a2m[i], a2m[j])]
                smat_ba = self.smat[np.ix_(a2m[j], a2m[i])]

                # get mu-parameter projection operator
                if isinstance(self.proj_oper, (int, float)):
                    proj_op[0] += self.proj_oper * np.dot(smat_ab, np.dot(sub_b_dmat[0], smat_ba))
                    proj_op[1] += self.proj_oper * np.dot(smat_ab, np.dot(sub_b_dmat[1], smat_ba))

                elif self.proj_oper in ('huzinaga', 'huz'):
                    fock_ab = [None, None]
                    fock_ab[0] = self.fock[0][np.ix_(a2m[i], a2m[j])]
                    fock_ab[1] = self.fock[1][np.ix_(a2m[i], a2m[j])]
                    fock_den_smat = [None, None]
                    fock_den_smat[0] = np.dot(fock_ab[0], np.dot(sub_b_dmat[0], smat_ba))
                    fock_den_smat[1] = np.dot(fock_ab[1], np.dot(sub_b_dmat[1], smat_ba))
                    proj_op[0] += -1. * (fock_den_smat[0] + fock_den_smat[0].transpose())
                    proj_op[1] += -1. * (fock_den_smat[1] + fock_den_smat[1].transpose())

                elif self.proj_oper in ('huzinagafermi', 'huzfermi'):
                    fock_ab = [None, None]
                    fock_ab[0] = self.fock[0][np.ix_(a2m[i], a2m[j])]
                    fock_ab[1] = self.fock[1][np.ix_(a2m[i], a2m[j])]
                    #The max of the fermi energy
                    efermi = [None, None]
                    if self.ft_setfermi is None:
                        efermi[0] = max([fermi[0] for fermi in self.ft_fermi])
                        efermi[1] = max([fermi[1] for fermi in self.ft_fermi])
                    else:
                        efermi[0] = self.ft_setfermi
                        efermi[1] = self.ft_setfermi

                    fock_ab[0] -= smat_ab * efermi[0]
                    fock_ab[1] -= smat_ab * efermi[1]

                    fock_den_smat = [None, None]
                    fock_den_smat[0] = np.dot(fock_ab[0], np.dot(sub_b_dmat[0], smat_ba))
                    fock_den_smat[1] = np.dot(fock_ab[1], np.dot(sub_b_dmat[1], smat_ba))
                    proj_op[0] += -1. * (fock_den_smat[0] + fock_den_smat[0].transpose())
                    proj_op[1] += -1. * (fock_den_smat[1] + fock_den_smat[1].transpose())

            self.proj_pot[i] = proj_op.copy()

        return True
    
    def update_fock_proj_diis(self, iter_num):
        """Updates the fock matrix and the projection potential together
           using a diis algorithm. Then subdivided into fragments for density
           relaxation. This only works in the absolutely localized basis."""

        fock = copy.copy(self.fock)
        fock = np.array(fock)
        num_rank = self.mol.nao_nr()
        dmat = [np.zeros((num_rank, num_rank)), np.zeros((num_rank, num_rank))]
        a2m = self.ao_to_mo
        sub_unrestricted = False
        diis_start_cycle = 10
        proj_energy = 0.
        for i, sub in enumerate(self.fragments):
            if sub.unrestricted:
                sub_unrestricted = True
            fock[0][np.ix_(a2m[i], a2m[i])] += self.proj_pot[i][0]
            fock[1][np.ix_(a2m[i], a2m[i])] += self.proj_pot[i][1]
            dmat[0][np.ix_(a2m[i], a2m[i])] += (sub.env_dmat[0])
            dmat[1][np.ix_(a2m[i], a2m[i])] += (sub.env_dmat[1])
            proj_energy += (np.einsum('ij,ji', self.proj_pot[i][0], sub.env_dmat[0]) +
                            np.einsum('ij,ji', self.proj_pot[i][1], sub.env_dmat[1]))
                            
        #remove off diagonal elements of fock matrix
        #new_fock = np.zeros_like(fock)
        #for i, sub in enumerate(self.fragments):
        #    new_fock[0][np.ix_(a2m[i], a2m[i])] = fock[0][np.ix_(a2m[i], a2m[i])]
        #    new_fock[1][np.ix_(a2m[i], a2m[i])] = fock[1][np.ix_(a2m[i], a2m[i])]

        #fock = new_fock

        if self.mol.spin == 0 and not(sub_unrestricted or self.fs_unrestricted):
            elec_dmat = dmat[0] + dmat[1]
        else:
            elec_dmat = copy.copy(dmat)

        
        elec_energy = self.fs_scf.energy_elec(dm=elec_dmat, h1e=self.hcore, vhf=self.emb_vhf)[0]
        elec_proj_energy = elec_energy + proj_energy

        if not(sub_unrestricted or self.fs_unrestricted):
            fock = (fock[0] + fock[1]) / 2.
            dmat = dmat[0] + dmat[1]

        if self.ft_diis_num == 2:
            fock = self.ft_diis.update(fock)
        elif self.ft_diis_num == 3:
            fock = self.ft_diis_2.update(fock)
            #temp_fock = self.ft_diis.update(self.smat, dmat, fock)
            #if iter_num > diis_start_cycle:
            #    fock = temp_fock
        #elif self.ft_diis_num > 3:
        #    temp_fock = self.ft_diis.update(self.smat, dmat, fock, elec_proj_energy)
        #    if iter_num > diis_start_cycle:
        #        fock = temp_fock

        if not(sub_unrestricted or self.fs_unrestricted):
            fock = [fock, fock]

        for i, sub in enumerate(self.fragments):
            sub_fock_0 = fock[0][np.ix_(a2m[i], a2m[i])]
            sub_fock_1 = fock[1][np.ix_(a2m[i], a2m[i])]
            sub.emb_proj_fock = [sub_fock_0, sub_fock_1]
