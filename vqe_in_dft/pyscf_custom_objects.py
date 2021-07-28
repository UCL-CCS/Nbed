from pyscf import gto, dft, scf, lib, cc, fci
from copy import deepcopy as copy
import numpy as np
import scipy as sp


class subsystem:
    """
    Base class used to define each subsystem object

    Args:
        mol (pyscf):
        env_method (str): exchange correlation (xc) functional for environement RKS calculation

    returns:
        blah
    Attributes:

    """

    def __init__(self, mol, env_method):
        self.mol = mol
        self.env_method = env_method

        # intialize different attributes
        self.env_scf = self._init_env_scf()
        self.env_hcore = self.env_scf.get_hcore()
        self.env_dmat = None
        self._init_density()

        self.emb_fock = None
        self.emb_PROJ_fock = None
        self.subsystem_fock = None
        self.proj_pot = np.zeros_like(
            self.env_hcore
        )  # gets updated by supersystem object (never in subsystem class!)
        self.H_A_in_B = None

        self.env_mo_occ = None
        self.env_mo_energy = None
        self.env_mo_coeff = None

    def _init_env_scf(self):
        """
        Initializes environement pyscf scf object

        Args:
            dl

        returns:
            env_scf (pyscf.RKS): returns restricted kohn-sham pyscf object of environment
        """
        if self.mol.spin != 0:
            raise ValueError("Code currently only runs restricted scf calcuations!")

        env_scf = scf.RKS(self.mol)
        env_scf.xc = self.env_method

        return env_scf

    def _init_density(self, dmat_guess_options=None):
        """
        Initializes subsystem density matrix

        Args:
            dl

        returns:
            env_scf (pyscf.RKS): returns restricted kohn-sham pyscf object of environment
        """

        scf_obj_ENV = self.env_scf

        if dmat_guess_options is None:
            env_dmat = scf_obj_ENV.get_init_guess()
        else:
            # this runs a calculation!
            scf_obj_ENV.kernal()
            env_dmat = scf_obj_ENV.make_rdm1()
            # TODO: other guess options allowed in pyscf

        self.env_dmat = env_dmat

        # once density initialized update subsystem Fock
        self.update_subsystem_fock()

    def update_subsystem_fock(self, hcore=None, dmat=None):
        """
        Update subsystem Fock matrix

        Args:
            hcore (numpy.array): core Hamiltonian of subsystem
            dmat (numpy.array): density matrix of core Hamiltonian

        returns:
            None
        """
        if hcore is None:
            hcore = self.env_hcore

        if dmat is None:
            dmat = self.env_dmat

        if self.mol.spin != 0:
            raise ValueError("Code currently only runs restricted scf calcuations!")

        self.subsystem_fock = self.env_scf.get_fock(h1e=hcore, dm=dmat)

        return None

    def update_emb_pot(self, emb_fock=None):
        """
        Update embedding potential for the subsystem

        args:
            emb_fock = can be used to define frozen potential!
        """

        if emb_fock is None:
            emb_fock = None
        else:
            emb_fock = self.emb_fock

        self.update_subsystem_fock()

        self.emb_fock = emb_fock - self.subsystem_fock

        return None

    def get_env_projection_energy(self):
        """
        Returns the projection operator energy!!

        Args:

        returns:
            None
        """
        dmat = copy(self.env_dmat)

        proj_pot = self.proj_pot

        # e_proj = np.trace(emb_pot.dot(dmat))
        e_proj = np.einsum("ij, ji", proj_pot, dmat).real
        return e_proj

    def get_env_electronic_energy(self):  # , emb_pot=None):
        """
        Returns total subsystem energy!

        Note: freeze and  thaw uses embedding fock (BUT not for energies)!

        Args:
            hcore (numpy.array): core Hamiltonian of subsystem
            dmat (numpy.array): density matrix of core Hamiltonian

        returns:
            None
        """
        dmat = copy(self.env_dmat)
        subsystem_electronic_energy = self.env_scf.energy_elec(dm=dmat)[
            0
        ]  # 0th index gives HF energy and [1] index gives Coloumb energy!

        e_projection = self.get_env_projection_energy()

        return subsystem_electronic_energy + e_projection

    def get_env_energy(self):  # , dmat=None, proj_pot=None):
        """
        Returns total subsystem energy!


        Args:
            hcore (numpy.array): core Hamiltonian of subsystem
            dmat (numpy.array): density matrix of core Hamiltonian

        returns:
            None
        """

        # dmat = copy(self.env_dmat)
        # e_emb = np.einsum('ij, ji' emb_pot, dmat).real
        # e_emb = np.trace(emb_pot.dot(dmat))

        self.env_energy = self.get_env_electronic_energy() + self.mol.energy_nuc()
        return self.env_energy

    def diagonalize_subsystem_dmat(self):
        """
        Get new subsystem density from updated Fock matrix! Return error between density matrices

        returns:
            error in difference between previous and update density matrix
        """

        old_subsys_dmat = copy(self.env_dmat)

        ## Diagonalize subsystem fock matrix and return new density
        if self.emb_PROJ_fock is None:
            emb_proj_Fock = (
                self.emb_fock + self.proj_pot
            )  # Note that self.proj_pot gets updated in supersystem freeze_thaw_calc! And thus is NOT array of zeros
        else:
            # given in update_fock_projector_diis method of supersystem!
            emb_proj_Fock = self.emb_PROJ_fock

        self.env_mo_energy, self.env_mo_coeff = self.env_scf.eig(
            emb_proj_Fock, self.env_scf.get_ovlp()
        )

        self.env_mo_occ = self.env_scf.get_occ(self.env_mo_energy, self.env_mo_coeff)

        # NEW density matrix from diagonalization!
        self.env_dmat = np.dot(
            (self.env_mo_coeff * self.env_mo_occ),
            self.env_mo_coeff.transpose().conjugate(),
        )

        ### END diagonalization routine

        # error - difference between new and old density matrices
        error = sp.linalg.norm(self.env_dmat - old_subsys_dmat)
        return error

    def Run_coupled_cluster_calc(self):
        # see __gen_hf_scf

        ### first need restricted Hartree fock
        high_level_scf = scf.RHF(self.mol)

        if self.proj_pot is None:
            raise ValueError(
                "No projection operator! Check supersystem calculations (look for huzinaga)"
            )

        def Get_embedded_Hamiltonian(mol, projection_op):  # , env_dmat=self.env_dmat):

            hcore = scf.hf.get_hcore(high_level_scf.mol)
            Hcore_with_embedding_pot = hcore + projection_op
            return Hcore_with_embedding_pot  # Fock_with_embedding_pot

        # Change object to use projection matrix!
        high_level_scf.get_hcore = lambda *args, **kwargs: Get_embedded_Hamiltonian(
            high_level_scf, self.proj_pot
        )  # lambda here means doesn't matter what arguements you have, will run function above (fairly hacky solution)
        # Run Hartree-Fock
        high_level_scf.run()

        # run coupled cluster on modified object!
        CC_high_levl_obj = cc.CCSD(high_level_scf)

        (
            ecc,
            t1,
            t2,
        ) = (
            CC_high_levl_obj.kernel()
        )  # Ecc = correlation energy, t1 and t2 are coupled cluster amplitudes

        CC_energy = high_level_scf.e_tot + ecc  # = HF + CC correction!

        print(f"High level energy: {CC_energy:>4.8f}")

        self.H_A_in_B = high_level_scf.get_hcore()

        return CC_energy


class Supersystem:
    """
    Class of global system, where subsystems define global system!

    Args:
        list_of_subsystems (list): list of pyscf mol objects
        env_method (str): exchange correlation (xc) functional for environement RKS calculation

    returns:
        blah
    Attributes:

    """

    def __init__(
        self,
        list_of_subsystems,
        full_sys_method,
        freeze_thaw_conv,
        freeze_thaw_max_iter,
    ):

        self.list_of_subsystems = list_of_subsystems
        self.full_sys_method = (
            full_sys_method  # XC correlation funciton for full system!
        )

        self.mol = None
        self._generate_supersystem()
        self.subsys2supersys = self._subsystem_two_supersystem_indices()

        self.full_sys_scf_obj = self._init_scf_supersystem()
        self.full_system_energy = None

        self.supersystem_emb_vhf = None
        self.supersystem_fock = None
        self.supersystem_hcore = self.full_sys_scf_obj.get_hcore()
        self.supersystem_Smat = self.full_sys_scf_obj.get_ovlp()
        self.supersystem_dmat = None

        self.subsystem_indexed_proj_OP_list = [
            None for _subsys in self.list_of_subsystems
        ]  # list of subsytem huzinaga operators!

        self.freeze_thaw_conv = freeze_thaw_conv  # convergence precision
        self.freeze_thaw_max_iter = (
            freeze_thaw_max_iter  # max number of iterations for each freeze thaw cycle
        )

        self.DFT_in_DFT_energy = None
        self.WF_in_DFT_energy = None
        self.WF_in_DFT_energy_corr = None

    def _generate_supersystem(self):
        """
        # https://sunqm.github.io/pyscf/_modules/pyscf/gto/mole.html#conc_mol
        # Concatenate subsystems into supersystem pyscf object!

        # TODO: Currently only works for 2 subsystems!

        returns:
            None
        """

        self.mol = gto.mole.conc_mol(
            self.list_of_subsystems[0].mol, self.list_of_subsystems[1].mol
        )

        return None

    def _subsystem_two_supersystem_indices(self):
        """
        Matrix for convecting between subsystems and the supersystem

        returns:
            env_scf (pyscf.RKS): returns restricted kohn-sham pyscf object of environment
        """

        nao = np.array(
            [subsystem.mol.nao_nr() for subsystem in self.list_of_subsystems]
        )
        nssl = [None for i in range(len(self.list_of_subsystems))]

        for i, sub in enumerate(self.list_of_subsystems):
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

        nsl = np.zeros(self.mol.natm, dtype=int)
        for i in range(self.mol.natm):
            i_b = np.where(self.mol._bas.transpose()[0] == i)[0].min()
            i_e = np.where(self.mol._bas.transpose()[0] == i)[0].max()
            i_r = self.mol.nao_nr_range(i_b, i_e + 1)
            i_r = i_r[1] - i_r[0]
            nsl[i] = i_r

        assert nsl.sum() == self.mol.nao_nr(), "naos not equal!"

        sub2sup = [None for i in range(len(self.list_of_subsystems))]
        for i, sub in enumerate(self.list_of_subsystems):
            sub2sup[i] = np.zeros(nao[i], dtype=int)
            for j in range(sub.mol.natm):
                match = False
                c_1 = sub.mol.atom_coord(j)
                for k in range(self.mol.natm):
                    c_2 = self.mol.atom_coord(k)
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

                assert match, "no atom match!"

        # self.subsys2supersys = sub2sup
        return sub2sup

    def _init_scf_supersystem(self):
        """
        ### TODO: can make options for different supersystem calculations!

        returns:
            None

        Attributes:

        """

        if self.mol.spin != 0:
            raise ValueError("Code currently only runs restricted scf calcuations!")

        full_sys_scf_obj = scf.RKS(self.mol)
        full_sys_scf_obj.xc = self.full_sys_method

        # grids are important for subsystem calcs (need to use this supersystem grid for those calcs)
        grids = dft.gen_grid.Grids(self.mol)
        grids.build()
        full_sys_scf_obj.grids = grids

        return full_sys_scf_obj

    def _init_subsystem_denisties_and_get_supersystem_density(self):
        """

        Initializes all subsystem densities and returns full system density

        returns:
            None

        Attributes:

        """

        if self.mol.spin != 0:
            raise ValueError("Code currently only runs restricted scf calcuations!")

        # run scf calc on supersystem!
        self.Get_supersystem_energy()

        # Initialize subsystem densities!
        # TODO could use supersystem calculated dmat and select parts using subsys2supersys matrix as initialization!
        for i, subsystem in enumerate(self.list_of_subsystems):

            # NEED GRIDS TO BE SAME FOR ALL OBJECTS!!!
            subsystem.env_scf.grids = self.full_sys_scf_obj.grids

            subsystem._init_density(
                dmat_guess_options=None
            )  # currently guess from isolated subsystems!
            # TODO: add supersystem slice here!

        return None

    def Get_embedded_dmat(self):
        """

        Initializes all subsystem densities and returns full system density

        returns:
            None

        Attributes:

        """
        dm_env = np.zeros((self.mol.nao_nr(), self.mol.nao_nr()))

        for i, subsystem in enumerate(self.list_of_subsystems):

            # add embedded subsystem density using correct slice!
            dm_env[
                np.ix_(self.subsys2supersys[i], self.subsys2supersys[i])
            ] += subsystem.env_dmat

        return dm_env

    def update_supersystem_fock(self):
        """

        Update supersystem Fock

        returns:
            None

        Attributes:

        """
        full_system_dmat = self.Get_embedded_dmat()

        self.supersystem_emb_vhf = self.full_sys_scf_obj.get_veff(
            self.mol, full_system_dmat
        )
        self.supersystem_fock = self.full_sys_scf_obj.get_fock(
            h1e=self.supersystem_hcore,
            vhf=self.supersystem_emb_vhf,
            dm=full_system_dmat,
        )

        for i, subsystem in enumerate(self.list_of_subsystems):
            # slice supersystem fock
            # and update subsystem Fock matrices
            subsystem.emb_fock = self.supersystem_fock[
                np.ix_(self.subsys2supersys[i], self.subsys2supersys[i])
            ]
        return None

    def update_proj_pot(self):
        """

        Update projection potential (does all subsystems at once)... This is the projection operator!

        only does Huzinaga #TODO: add other methods such as mu param

        returns:
            None

        Attributes:

        """
        for i, subsystem_A in enumerate(self.list_of_subsystems):

            # SubSys_projection_OP = np.zeros((subsystem_A.mol.nao_nr(), subsystem_A.mol.nao_nr()), dtype=float)
            SubSys_projection_OP = np.zeros_like(subsystem_A.env_hcore)
            # cycle over OTHER subsystems
            for j, subsystem_B in enumerate(self.list_of_subsystems):
                if j == i:
                    continue

                subsystem_B_dmat = subsystem_B.env_dmat

                # Smat_AB = self.supersystem_Smat[np.ix_(self.subsys2supersys[i], self.subsys2supersys[j])] # index i,j
                Smat_BA = self.supersystem_Smat[
                    np.ix_(self.subsys2supersys[j], self.subsys2supersys[i])
                ]  # index j, i

                # Huzinaga method
                fock_AB = self.supersystem_fock[
                    np.ix_(self.subsys2supersys[i], self.subsys2supersys[j])
                ]  # index i,j

                F_AB_yammaB_S_BA = np.dot(fock_AB, subsystem_B_dmat.dot(Smat_BA))

                SubSys_projection_OP += -1 * (
                    F_AB_yammaB_S_BA + F_AB_yammaB_S_BA.transpose()
                )

            self.subsystem_indexed_proj_OP_list[i] = SubSys_projection_OP.copy()

        return None

    def update_fock_projector_diis(self):
        """

        Update the Fock matrix and the projection potential (projection operators) together using DIIS alg.

        NOTE this only works in the absolutely localized basis

        TODO: currently doesn't apply DIIS, just updates fock matrices to have projection operator in it

        returns:
            None

        Attributes:

        """

        super_Fock = copy(self.supersystem_fock)

        # dmat = np.zeros((self.mol.nao_nr(), self.mol.nao_nr()))
        dmat = np.zeros_like(self.supersystem_hcore)
        projection_energy = 0

        for i, subsystem in enumerate(self.list_of_subsystems):

            # add projection operator to SUPERsystem Fock matrix
            super_Fock[
                np.ix_(self.subsys2supersys[i], self.subsys2supersys[i])
            ] += self.subsystem_indexed_proj_OP_list[i]
            dmat[
                np.ix_(self.subsys2supersys[i], self.subsys2supersys[i])
            ] += subsystem.env_dmat

            ## projection_energy += np.trace(self.subsystem_indexed_proj_OP_list[i].dot(subsystem.env_dmat))
            # projection_energy += np.einsum('ij, ji', self.subsystem_indexed_proj_OP_list[i], subsystem.env_dmat)

        # electronic_E = self.full_sys_scf_obj.energy_elec(dm=dmat, h1e=self.supersystem_hcore, vhf=self.supersystem_emb_vhf)
        # Energy_electronic_and_projected = electronic_E + projection_energy

        for i, subsystem in enumerate(self.list_of_subsystems):
            # update subsystem Fock matrices with  SUPERfock matrix that has huzinaga op in it!
            subsystem.emb_PROJ_fock = super_Fock[
                np.ix_(self.subsys2supersys[i], self.subsys2supersys[i])
            ]

        return None

    def Run_freeze_and_thaw_embedding_calc(self):  # , verbose=False):
        """

        Perform embedded freeze and thaw calcualtion. This does subsystem density optimization!

        returns:
            None

        Attributes:

        """

        freeze_thaw_error = 1  # needed for start of loop (overwritten)
        freeze_thaw_iter = 0
        while (freeze_thaw_error > self.freeze_thaw_conv) and (
            freeze_thaw_iter < self.freeze_thaw_max_iter
        ):

            freeze_thaw_error = 0  # <-- overwritten
            freeze_thaw_iter += 1

            self.update_supersystem_fock()

            self.update_proj_pot()  # calculates the huzinaga projection operators for each subsystem

            # self.update_fock_projector_diis() # adds huzinaga operator to each subsystem Fock matrix (subsystem.emb_PROJ_fock object)

            for i, subsystem in enumerate(self.list_of_subsystems):

                subsystem.proj_pot = self.subsystem_indexed_proj_OP_list[
                    i
                ]  # projection operator for subsystem!

                diff_in_dmats = (
                    subsystem.diagonalize_subsystem_dmat()
                )  # also updates subsystem density!
                freeze_thaw_error += diff_in_dmats

                ## optional start
                # if verbose is True:
                e_proj = subsystem.get_env_projection_energy()
                print(
                    f"iter:{freeze_thaw_iter:>3d}:   subsystem:{i:<2d}  |ddm|:{freeze_thaw_error:12.6e}   |Tr[DP]|:{e_proj:12.6e}"
                )

        # check convergence
        freeze_thaw_conv_flag = True
        if freeze_thaw_error > self.freeze_thaw_conv:
            freeze_thaw_conv_flag = False
            print("\n Freeze and thaw NOT converged!!! \n")
        else:
            print("\n Freeze and thaw has converged!!! \n ")

        self.update_supersystem_fock()  # final update using optimal subsystem densities
        self.update_proj_pot()

        ## print subsystem energies with optimized densities!!!
        for i, subsystem in enumerate(self.list_of_subsystems):
            subsystem.get_env_energy()  # <--subsystem energy (nuclear + electronic + projection)

            print(f"subsystem {i} Energy: {subsystem.env_energy:>4.8f}")

        return freeze_thaw_conv_flag

    def Get_supersystem_energy(self):
        """

        Get supersystem energy and store density matrix

        returns:
            None

        Attributes:

        """

        # check if calculation already done!
        if (self.full_system_energy is None) or not (self.full_sys_scf_obj.converged):

            full_system_method = (
                self.full_sys_method
            )  # TODO can add option to change what theory to use

            if self.freeze_thaw_conv:
                full_system_dmat = (
                    self.Get_embedded_dmat()
                )  # builds dmat from freeze thaw optimzed subsystem densities!
                self.full_system_energy = self.full_sys_scf_obj.kernel(
                    dm0=full_system_dmat
                )  # dm0 = initial guess for density matrix!  # <-- this runs scf calc!
            else:
                self.full_system_energy = (
                    self.full_sys_scf_obj.kernel()
                )  # <-- this runs scf calc!

            # convergence_flag, self.full_system_energy, self.supersystem_mo_energy, self.supersystem_mo_coeff, self.supersystem_mo_occ = self.full_sys_scf_obj.kernel() #< this runs scf calc!
            # self.full_system_energy = self.full_sys_scf_obj.kernel() #<-- this runs scf calc!

            self.supersystem_mo_coeff = self.full_sys_scf_obj.mo_coeff
            self.supersystem_mo_occ = self.full_sys_scf_obj.mo_occ
            self.supersystem_mo_energy = self.full_sys_scf_obj.mo_energy
            self.supersystem_dmat = self.full_sys_scf_obj.make_rdm1(
                self.supersystem_mo_coeff, self.supersystem_mo_occ
            )

            print(
                f"Supersystem KS-DFT calculation Energy: {self.full_system_energy:>4.8f}"
            )

            # TODO: Can run localization pyscf functions here to partition system!

    def Get_DFT_in_DFT_energy(self):
        """

        Calcualte DFt in DFT energy... NOT necessary for total embedding, but useful as diagnostic tool

        returns:
            DFT in DFT energy

        Attributes:

        """

        full_system_embedded_dmat = self.Get_embedded_dmat()

        DFT_in_DFT_energy = self.full_sys_scf_obj.energy_tot(
            h1e=self.supersystem_hcore,
            vhf=self.supersystem_emb_vhf,
            dm=full_system_embedded_dmat,
        )

        projection_e = 0
        for subsystem in self.list_of_subsystems:
            projection_e += subsystem.get_env_projection_energy()

        DFT_in_DFT_energy += projection_e
        print(f"DFT-in-DFT Energy: {DFT_in_DFT_energy:>4.8f}")
        self.DFT_in_DFT_energy = DFT_in_DFT_energy

    def Get_WF_in_DFT_energy(self):
        """

        Calcualte WF in DFT energy by ONIOM method.
            - Energy is calculated for the isolated subsystem A of interest using a expensive (high level) method
            - Energy is calculated for the isolated subsystem A using a low level method

        The full energy is then: E_total = E_{total}^{low-level} + E_{A}^{high-level} - E_{A}^{low-level}

        returns:
            WF in DFT energy

        Attributes:

        """
        supersystem_energy = self.full_system_energy

        print(f"DFT Supersystem calc: {supersystem_energy:>4.8f}")

        sub_system_A = self.list_of_subsystems[0]
        sub_system_A_environment_energy = sub_system_A.env_energy
        sub_system_A_WF_energy = sub_system_A.Run_coupled_cluster_calc()

        Energy_total = (
            supersystem_energy
            - sub_system_A_environment_energy
            + sub_system_A_WF_energy
        )

        self.WF_in_DFT_energy = Energy_total

        print(f"WF-in-DFT Energy: {Energy_total:>4.8f}")

    def Get_WF_in_DFT_energy_SUPERIOR(self):
        """

        Calcualte WF in DFT energy with corrections

        returns:
            WF in DFT energy

        Attributes:

        """

        supersystem_energy = self.full_system_energy
        print(f"DFT Supersystem calc: {supersystem_energy:>4.8f}")

        if self.DFT_in_DFT_energy is None:
            self.Get_DFT_in_DFT_energy()

        print(f"DFT-in-DFT calc: {self.DFT_in_DFT_energy:>4.8f}")

        sub_system_A = self.list_of_subsystems[0]
        sub_system_A_environment_energy = sub_system_A.env_energy
        sub_system_A_WF_energy = sub_system_A.Run_coupled_cluster_calc()

        ## WF in DFT
        approx_WF_in_DFT_energy = sub_system_A_WF_energy + self.DFT_in_DFT_energy

        subsystem_A_dmat = sub_system_A.env_dmat
        first_order_corr = np.trace(subsystem_A_dmat.dot(sub_system_A.H_A_in_B))
        approx_WF_in_DFT_energy += first_order_corr

        jcross_SubsysA = sub_system_A.get_jk(
            (sub_system_A, sub_system_A, sub_system_A, sub_system_A),
            subsystem_A_dmat,
            scripts="ijkl,lk->ij",
            aosym="s4",
        )
        ecoul_SubsysA = np.einsum("ij,ij", jcross_SubsysA, subsystem_A_dmat)

        kcross_SubsysA = sub_system_A.get_jk(
            (sub_system_A, sub_system_A, sub_system_A, sub_system_A),
            subsystem_A_dmat,
            scripts="ijkl,jk->il",
        )
        ex_SubsysA = np.einsum("ij,ji", kcross_SubsysA, subsystem_A_dmat)

        coloumb_exchange_E_subsys_A = np.einsum(
            "ij, ji",
            sub_system_A.env_scf.get_veff(dm=subsystem_A_dmat),
            subsystem_A_dmat,
        ).real
        approx_WF_in_DFT_energy += coloumb_exchange_E_subsys_A
        print(np.allclose(coloumb_exchange_E_subsys_A, ecoul_SubsysA + ex_SubsysA))
        print(
            f"first order correction: {first_order_corr + coloumb_exchange_E_subsys_A:>4.8f}"
        )
        print("subsystem A energy:", sub_system_A_environment_energy)
        print(
            "subsystem A energy via correction:",
            first_order_corr + coloumb_exchange_E_subsys_A,
        )

        self.WF_in_DFT_energy_corr = (
            supersystem_energy - self.DFT_in_DFT_energy + approx_WF_in_DFT_energy
        )

        print(f"WF-in-DFT /w first order corr: {self.WF_in_DFT_energy_corr:>4.8f}")

    def Supersystem_FCI(self):

        myhf = scf.RHF(self.mol)
        myhf.kernel()
        cisolver = fci.FCI(self.mol, myhf.mo_coeff)
        self.FCI_energy, self.FCI_WaveFn = cisolver.kernel()

        print(f"Supersystem FCI Energy: {self.FCI_energy:>4.8f}")

    def Run_full_embedded_WF_in_DFT_calc(self):

        # Freeze and Thaw
        print("".center(80, "*"), "\n")
        self.Run_freeze_and_thaw_embedding_calc()
        print("".center(80, "*"), "\n")

        # supersystem energy
        print("".center(80, "*"), "\n")
        self.Get_supersystem_energy()
        print("".center(80, "*"), "\n")

        # DFT in DFT energy
        print("".center(80, "*"), "\n")
        self.Get_DFT_in_DFT_energy()
        print("".center(80, "*"), "\n")

        # WF in DFT energy ONIOM
        print("".center(80, "*"), "\n")
        self.Get_WF_in_DFT_energy()
        print("".center(80, "*"), "\n")

        # WF in DFT energy w/ correction
        print("".center(80, "*"), "\n")
        self.Get_WF_in_DFT_energy_SUPERIOR()
        print("".center(80, "*"), "\n")

        # FCI energy
        print("".center(80, "*"), "\n")
        self.Supersystem_FCI()
        print("".center(80, "*"), "\n")

        print(
            "Error in Supersystem KS-DFT calc:",
            self.FCI_energy - self.full_system_energy,
        )
        print("Error in DFT-in-DFT calc:", self.FCI_energy - self.DFT_in_DFT_energy)
        print("Error in WF-in-DFT calc:", self.FCI_energy - self.WF_in_DFT_energy)
        print(
            "Error in WF-in-DFT /w correction:",
            self.FCI_energy - self.WF_in_DFT_energy_corr,
        )
