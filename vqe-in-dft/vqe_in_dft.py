import scipy as sp
from pyscf import gto, dft, lib, mp, cc, scf, tools, ci, fci, lo, ao2mo
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from openfermion.transforms import jordan_wigner
from openfermion.ops import FermionOperator
import py3Dmol
from pyscf.tools import cubegen
import os

# pip install py3Dmol
# conda install conda-forge rdkit

# may need to do:
# pip install protobuf==3.13.0

class standard_full_system_molecule():


    def __init__(self, geometry, N_active_atoms, low_level_scf_method='RKS', low_level_xc_functional= 'lda, vwn',
                       E_convergence_tol = 1e-6, basis = 'STO-3G', output_file_name='output.dat', unit= 'angstrom', pyscf_print_level=1, memory=4000, charge=0, spin=0, run_fci=True):
        """
        Initialize the full molecular system.

            Args:
                geometry (list): A list of tuples giving atom (str) and coordinates (tuple of floats).
                                 An example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
                                 IMPORTANT: First N atoms must be the active system (to be treated at higher level later)

                low_level_scf_method (str): String defining cheap SCF method.

                E_convergence_tol (float): Energy convergence threshold. 

                den_convergence_tol (float): Density convergence threshold. 

                basis (str): A string defining the basis set. An example is 'cc-pvtz'.

                output_file_name (str): string of output file name

                unit (str): Units for coordinates of input atoms (geometry). Should be either: "angstrom" or "bohr"

                pyscf_print_level (int): pyscf print level. 0 (no output - quiet) to 9 (lots of print information - noisy)

                memory (int): memory usage in MB

                charge (int): Charge of molecule, note this effects the number of electrons.

                spin (int): The number of unpaired electrons: 2S, i.e. the difference between the number of alpha and beta electrons.

                run_fci (bool): whether to run FCI calculation.

        """


        self.geometry = geometry
        self.basis = basis
        self.unit = unit
        self.memory = memory # RAM in MB
        self.charge = charge 
        self.spin = spin
        self.N_active_atoms = N_active_atoms

        # SCF info
        self.low_level_scf_method = low_level_scf_method
        self.low_level_xc_functional = None if low_level_scf_method=='RHF' else low_level_xc_functional


        self.E_convergence_tol = E_convergence_tol

        # file_system and print
        self.output_file_name = output_file_name
        self.pyscf_print_level =  pyscf_print_level

        # 
        self.run_fci = run_fci


        self._init_pyscf_system()

        self.E_FCI = self._run_fci() if run_fci else None

    def _init_pyscf_system(self):

        self.full_system_mol = gto.Mole(atom= self.geometry,
                      basis=self.basis,
                       charge=self.charge,
                       spin= self.spin,
                      )

        self.full_system_mol.unit = self.unit
        self.full_system_mol.build()


        if self.low_level_scf_method == 'RKS':
            self.full_system_scf = scf.RKS(self.full_system_mol)
            self.full_system_scf.verbose = self.pyscf_print_level
            self.full_system_scf.max_memory= self.memory
            self.full_system_scf.conv_tol = self.E_convergence_tol
            self.full_system_scf.xc = self.low_level_xc_functional
        else:
            raise ValueError(f'Unknown SCF method: {self.low_level_scf_method}')

        self.S_ovlp = self.full_system_scf.get_ovlp()
        self.standard_hcore = self.full_system_scf.get_hcore()
        self.full_system_scf.kernel()

        two_e_term_total = self.full_system_scf.get_veff()
        self.e_xc_total = two_e_term_total.exc
        self.v_xc_total = two_e_term_total - self.full_system_scf.get_j() 


    def _get_xyz_format(self):

        if isinstance(self.geometry, str):
            # if already xzy format
            return self.geometry
        
        n_atoms = len(self.geometry)
        xyz_str=f'{n_atoms}'
        xyz_str+='\n \n'
        for atom, xyz in self.geometry:
            xyz_str+= f'{atom}\t{xyz[0]}\t{xyz[1]}\t{xyz[2]}\n'
        
        return xyz_str

    def draw_molecule_3D(self, width=400, height=400, jupyter_notebook=False):

        if jupyter_notebook is True:
            import rdkit
            from rdkit.Chem import Draw
            from rdkit.Chem.Draw import IPythonConsole
            rdkit.Chem.Draw.IPythonConsole.ipython_3d = True  # enable py3Dmol inline visualization

        xyz_geom = self._get_xyz_format()
        view = py3Dmol.view(width=width, height=height)
        view.addModel(xyz_geom, "xyz")
        view.setStyle({'stick':{}})
        view.zoomTo()
        return(view.show())


    def _run_fci(self):

        HF_scf = scf.RHF(self.full_system_mol)
        HF_scf.verbose= self.pyscf_print_level
        HF_scf.max_memory= self.memory
        HF_scf.conv_tol = self.E_convergence_tol
        HF_scf.kernel()

        self.my_fci = fci.FCI(HF_scf).run()
        # print('E(UHF-FCI) = %.12f' % self.my_fci.e_tot)

        self.E_FCI = self.my_fci.e_tot

        # myci = ci.CISD(HF_scf).run() # this is UCISD
        # print('UCISD total energy = ', myci.e_tot)
        return self.my_fci.e_tot

    def Draw_cube_orbital(self, xyz_string, cube_file, width=400, height=400):

        view = py3Dmol.view(width=width, height=height)
        view.addModel(xyz_string, "xyz")
        view.setStyle({'stick':{}})
        
        with open(cube_file, 'r') as f:
            view.addVolumetricData(f.read(), "cube", {'isoval': -0.02, 'color': "red", 'opacity': 0.75})
        with open(cube_file, 'r') as f2:
            view.addVolumetricData(f2.read(), "cube", {'isoval': 0.02, 'color': "blue", 'opacity': 0.75})
        view.zoomTo()
        return view.show()

    def plot_orbital(self, C_matrix, index):
        

        xyz_geom = self._get_xyz_format()
        if not isinstance(index, int):
            raise ValueError(f'index: {index} required for slice is not an integar')
        if  C_matrix.shape[1]<=index:
            raise ValueError('index is outside of C_matrix shape')


        File_name = f'temp_MO_orbital_index{index}.cube'
        cubegen.orbital(self.full_system_mol, File_name, C_matrix[:, index])
        self.Draw_cube_orbital(xyz_geom, File_name)
        os.remove(File_name) # delete file once orbital is drawn

        return None

    def plot_all_orbitals(self, C_matrix):
        
        for MO_ind in range(C_matrix.shape[1]):
            self.plot_orbital(C_matrix, MO_ind)
        
        return None



class embedded_molecular_system(standard_full_system_molecule):

        def __init__(self, geometry, N_active_atoms, localization_method, projection_method, high_level_scf_method='CCSD', high_level_xc_functional = 'b3lyp', mu_level_shift=1e6,

                       low_level_scf_method='RKS', low_level_xc_functional= 'lda, vwn', E_convergence_tol = 1e-6, basis = 'STO-3G', output_file_name='output.dat',
                        unit= 'angstrom', pyscf_print_level=1, memory=4000, charge=0, spin=0, run_fci=False):


            super().__init__(geometry, N_active_atoms, low_level_scf_method, low_level_xc_functional,
                              E_convergence_tol, basis, output_file_name, unit, 
                             pyscf_print_level, memory, charge, spin, run_fci)

            self.localization_method = localization_method

            self.projection_method = projection_method
            self.mu_level_shift = mu_level_shift

            self.high_level_scf_method = high_level_scf_method
            self.high_level_xc_functional = None if high_level_scf_method=='RHF' else high_level_xc_functional

            self.v_emb = None
            self.dm_active = None
            self.density_emb_active = None


        def _Localize_orbitals(self, localization_method, sanity_check=False, save_localized_MO_orbs=False):

            AO_slice_matrix = self.full_system_mol.aoslice_by_atom()

            

            # run localization scheme
            if localization_method.lower() == 'spade':

                # Take occupied orbitals of global system calc
                occupied_orbs = self.full_system_scf.mo_coeff[:,self.full_system_scf.mo_occ>0]

                # Get active AO indices
                N_active_AO = AO_slice_matrix[self.N_active_atoms-1][3]  # find max AO index for active atoms (neg 1 as python indexs from 0)

                
                S_half = sp.linalg.fractional_matrix_power(self.S_ovlp, 0.5)
                orthogonal_orbitals = (S_half@occupied_orbs)[:N_active_AO, :] # Get rows (the active AO) of orthogonal orbs 

                # Comute singular vals
                u, singular_values, rotation_matrix = np.linalg.svd(orthogonal_orbitals, full_matrices=True)

                # find where largest step change 
                # delta_s = singular_values[1:] - singular_values[:-1]
                delta_s = [(i, (singular_values[i] - singular_values[i+1] )) for i in range(len(singular_values) - 1)] # contains (index, delta_s)

                self.n_act_mos = max(delta_s, key=lambda x: x[0])[0] + 1 # finds index where largest step change is! (adds 1 due to python indexing going from 0)
                self.n_env_mos = len(singular_values) - self.n_act_mos

                # define active and environment orbitals from localization
                self.act_orbitals = occupied_orbs @ rotation_matrix.T[:, :self.n_act_mos]
                self.env_orbitals = occupied_orbs @ rotation_matrix.T[:, self.n_act_mos:]

                self.C_matrix_all_localized_orbitals = occupied_orbs @ rotation_matrix.T

                self.active_MO_inds  = np.arange(self.n_act_mos)
                self.enviro_MO_inds = np.arange(self.n_act_mos, self.n_act_mos+self.n_env_mos)

            else:
                THRESHOLD = 0.9

                # Take C matrix from SCF calc
                opt_C = self.full_system_scf.mo_coeff

                # run localization scheme
                if localization_method.lower() == 'pipekmezey':
                    ### PipekMezey
                    PM = lo.PipekMezey(self.full_system_mol, opt_C)
                    PM.pop_method = 'mulliken' # 'meta-lowdin', 'iao', 'becke'
                    C_loc = PM.kernel() # includes virtual orbs too!
                    C_loc_occ = C_loc[:,self.full_system_scf.mo_occ>0]

                elif localization_method.lower() == 'boys':
                    ### Boys
                    boys_SCF = lo.boys.Boys(self.full_system_mol, opt_C)
                    C_loc  = boys_SCF.kernel()
                    C_loc_occ = C_loc[:,self.full_system_scf.mo_occ>0]

                elif localization_method.lower() == 'ibo':
                    ### intrinsic bonding orbs
                    C_loc = lo.ibo.ibo(self.full_system_mol, sum(self.full_system_scf.mo_occ)//2, locmethod='IBO', verbose=1)
                    C_loc_occ = C_loc[:,self.full_system_scf.mo_occ>0]
                else:
                    raise ValueError(f'unknown localization method {localization_method}')
                
                                
                # find indices of AO of active atoms
                ao_active_inds = np.arange(AO_slice_matrix[0,2], AO_slice_matrix[self.N_active_atoms-1,3])

                #### use einsum to be faster!

                # MO_active_inds = []
                # for mo_orb_loc_ind in range(C_loc_occ.shape[1]):
                    
                #     mo_overlap_with_active_ao = sum(C_loc_occ[ao_active_inds , mo_orb_loc_ind])
                #     print(mo_overlap_with_active_ao)
                #     if mo_overlap_with_active_ao>THRESHOLD:
                #         mo_overlap_with_active_ao.append(mo_orb_loc_ind)

                #### 

                # note only using OCCUPIED C_loc_occ
                
                # # active_AO_MO_overlap = np.einsum('ij->j', C_loc_occ[ao_active_inds, :]) # (take active rows (active AOs) and all columns) sum  down columns (to give MO contibution from active AO)
                # S_half = sp.linalg.fractional_matrix_power(self.S_ovlp, 0.5)
                # ortho_localized_orbs =  S_half@C_loc_occ # make sure localized MOs are orthogonal!
                # # sum(np.abs(ortho_localized_orbs[:,ind])**2) # should equal 1 for any ind as normalized!

                # # TODO: Check this code here!!!!!
                # # active_AO_MO_overlap = np.einsum('ij->j', ortho_localized_orbs[ao_active_inds, :])# (take active rows (active AOs) and all columns) sum  down columns (to give MO contibution from active AO)
                # # active_AO_MO_overlap = np.sqrt(np.einsum('ij->j', np.abs(ortho_localized_orbs[ao_active_inds, :])**2))
                # active_AO_MO_overlap = np.einsum('ij->j', C_loc_occ[ao_active_inds, :])
                # print(active_AO_MO_overlap)

                # # threshold to check which MOs have a high character from the active AOs 
                # # self.active_MO_inds = np.where(active_AO_MO_overlap>THRESHOLD)[0]
                # # IMPORTANT CHANGE HERE

                active_MO_ind_list=[]
                for mo_ind in range(C_loc_occ.shape[1]):
                    MO_orb = C_loc_occ[:, mo_ind] # MO coefficients (c_j)
                    MO_active_AO_overlap=0
                    for active_AO_index in ao_active_inds:
                        bra_aoI_ket_All_AOs = self.S_ovlp[active_AO_index, :] # [ < ϕ_AO_SELECTED | ϕ_AO_0> , < ϕ_AO_SELECTED | ϕ_AO_1>, ..., < ϕ_AO_SELECTED | ϕ_AO_M> ]
                        AO_i_overlap_MO = np.dot(bra_aoI_ket_All_AOs.T, MO_orb) # < ϕ_AO_i |ψ_MO> = Σ_j  (c_j < ϕ_AO_i | ϕ_AO_j >) # aka summation of ci and overlap
                        MO_active_AO_overlap+=AO_i_overlap_MO
                    if MO_active_AO_overlap>THRESHOLD:
                        active_MO_ind_list.append(mo_ind)

                self.active_MO_inds = np.array(active_MO_ind_list)

                # threshold to check which MOs have a high character from the active AOs 
                # self.active_MO_inds = np.where(active_AO_MO_overlap>THRESHOLD)[0]
                self.enviro_MO_inds = np.array([i for i in range(C_loc_occ.shape[1]) if i not in self.active_MO_inds]) # get all non active MOs

                # define active MO orbs and environment
                self.act_orbitals = C_loc_occ[:, self.active_MO_inds] # take MO (columns of C_matrix) that have high dependence from active AOs
                self.env_orbitals = C_loc_occ[:, self.enviro_MO_inds]
                
                self.C_matrix_all_localized_orbitals = C_loc_occ

                self.n_act_mos = len(self.active_MO_inds)
                self.n_env_mos = len(self.enviro_MO_inds)



            self.dm_active =  2 * self.act_orbitals @ self.act_orbitals.T
            self.dm_enviro =  2 * self.env_orbitals @ self.env_orbitals.T

            if sanity_check:
                ## check number of electrons is still the same after orbitals have been localized (change of basis)
                if not np.isclose((np.trace(self.dm_active@self.S_ovlp) + np.trace(self.dm_enviro@self.S_ovlp)), self.full_system_mol.nelectron):
                    raise ValueError('number of electrons in localized orbitals is incorrect')
                

                # checking denisty matrix parition makes sense:

                # gamma_localized_full_system = gamma_act + gamma_env
                dm_localised_full_system = 2* self.C_matrix_all_localized_orbitals@ self.C_matrix_all_localized_orbitals.conj().T
                if not np.allclose(dm_localised_full_system, self.dm_active + self.dm_enviro):
                    raise ValueError('gamma_full != gamma_active + gamma_enviro')
            

            if save_localized_MO_orbs:
                # Save localized orbitals as molden file
                with open('LOCALIZED_orbs.molden', 'w') as outfile:
                    tools.molden.header(self.full_system_mol,
                                        outfile)
                    tools.molden.orbital_coeff(
                                        self.full_system_mol,
                                         outfile, 
                                         self.act_orbitals, # <- active orbitals!
                                         ene=self.full_system_scf.mo_energy[self.active_MO_inds],
                                         occ=self.full_system_scf.mo_occ[self.active_MO_inds])

            return None

        def _Get_embedded_potential(self):

            if self.dm_active is None:
                self._Localize_orbitals(self.localization_method)

            if self.projection_method == 'huzinaga':
                Fock = self.standard_hcore + self.full_system_scf.get_veff(dm=self.dm_active+self.dm_enviro)
                F_gammaB_S = Fock @ self.dm_enviro @ self.S_ovlp
                projector = -0.5 * (F_gammaB_S + F_gammaB_S.T)
            elif self.projection_method == 'mu_shfit':
                projector = self.mu_level_shift * (self.S_ovlp @ self.dm_enviro  @ self.S_ovlp)
            else:
                raise ValueError(f'Unknown projection method {self.projection_method}')


            # define the embedded potential
            g_A_and_B = self.full_system_scf.get_veff(dm=self.dm_active+self.dm_enviro)

            g_A = self.full_system_scf.get_veff(dm=self.dm_active)

            # V_embed = G[𝛾_act + 𝛾_env] − G[𝛾_act] + Projector
            self.v_emb = g_A_and_B - g_A + projector

            return None

        def Get_energy_from_dm(self, dm_matrix, check_E_with_pyscf=True):
            """
            Get Energy from denisty matrix

            Note this uses the standard hcore (NO embedding potential here!)
            """

            # It seems that PySCF lumps J and K in the J array 
            J_mat = self.full_system_scf.get_j(dm = dm_matrix)
            K_mat = np.zeros_like(J_mat)
            two_e_term =  self.full_system_scf.get_veff(dm=dm_matrix)
            e_xc = two_e_term.exc
            v_xc = two_e_term - J_mat 

            Energy_elec = np.einsum('ij, ij', dm_matrix, self.standard_hcore + J_mat/2) + e_xc
            
            if check_E_with_pyscf:
                Energy_elec_pyscf = self.full_system_scf.energy_elec(dm=dm_matrix)[0]
                if not np.isclose(Energy_elec_pyscf, Energy_elec):
                    raise ValueError('Energy calculation incorrect')

            return Energy_elec, J_mat, K_mat, e_xc, v_xc

        def Get_optimized_embedded_dm(self):


            if self.dm_active is None:
                self._Localize_orbitals()

            # Define embedded system
            self.full_system_mol_EMBEDDED = gto.Mole(atom= self.geometry,
                      basis=self.basis,
                       charge=self.charge,
                       spin= self.spin,
                      )

            self.full_system_mol_EMBEDDED.build()

            # RE-DEFINE number of electrons in system
            self.full_system_mol_EMBEDDED.nelectron = 2*len(self.active_MO_inds)


            if (self.low_level_scf_method == 'RKS' and self.high_level_scf_method=='RKS'):
                self.full_system_EMBEDDED_scf = scf.RKS(self.full_system_mol_EMBEDDED) # <-- DFT calculation
                self.full_system_EMBEDDED_scf.verbose = self.pyscf_print_level
                self.full_system_EMBEDDED_scf.max_memory= self.memory
                self.full_system_EMBEDDED_scf.conv_tol = self.E_convergence_tol
                self.full_system_EMBEDDED_scf.xc = self.low_level_xc_functional # <-- LOW level calculation (TODO: could change to high level)

            elif (self.low_level_scf_method == 'RKS' and self.high_level_scf_method=='CCSD'):
                self.full_system_EMBEDDED_scf = scf.RHF(self.full_system_mol_EMBEDDED) # <-- HF calculation
                self.full_system_EMBEDDED_scf.verbose = self.pyscf_print_level
                self.full_system_EMBEDDED_scf.max_memory= self.memory
                self.full_system_EMBEDDED_scf.conv_tol = self.E_convergence_tol

            else:
                raise ValueError(f'Unknown SCF methods: {self.low_level_scf_method, self.high_level_scf_method}')


            if self.v_emb is None:
                self._Get_embedded_potential()

            # overwrite h_core to include embedding term!!!!
            self.full_system_EMBEDDED_scf.get_hcore = lambda *args: self.v_emb + self.standard_hcore 

            # run SCF calculation with embedding potential!
            self.full_system_EMBEDDED_scf.kernel()

            if self.full_system_EMBEDDED_scf.conv_check is False:
                raise ValueError('Embedded calculation has NOT converged')

            if self.high_level_scf_method=='CCSD':
                embedded_cc_obj = cc.CCSD(self.full_system_EMBEDDED_scf)
                embedded_cc_obj.verbose = self.pyscf_print_level

                # NEED to redefine embedded h_core again!
                embedded_cc_obj._scf.get_hcore = lambda *args: self.v_emb + self.standard_hcore 
                self.e_cc, self.t1, self.t2 = embedded_cc_obj.kernel()
                self.eris = embedded_cc_obj.ao2mo(mo_coeff=self.full_system_EMBEDDED_scf.mo_coeff)
                # ^ Note we do not use this RDM as energy comes from full_system_EMBEDDED_scf C matrix and t1, t2 terms!

                if not np.isclose(embedded_cc_obj.e_hf, self.full_system_EMBEDDED_scf.e_tot):
                    # check HF and CC_hf match!
                    raise ValueError('Error in HF calc of coupled cluster')


                # # undo hcore change!

                # embedded_cc_obj._scf.get_hcore = lambda *args: self.standard_hcore
                # eris = embedded_cc_obj.ao2mo(mo_coeff=self.full_system_EMBEDDED_scf.mo_coeff)
                # E_cc_standard_with_embedded_Density = embedded_cc_obj.energy(self.t1, self.t2, eris)
                # print(E_cc_standard_with_embedded_Density, E_cc_standard_with_embedded_Density) # check to see if same result with hcore back to normal

               # trygve helgaker pgs 20 - 23
                RDM1_CC = embedded_cc_obj.make_rdm1() # from coupled cluster calc
                # The diagonal elements of the **spin-orbital** one electron density matrix are the expectation values of the occupation-number operators!
                if not np.isclose(np.trace(RDM1_CC), 2*len(self.active_MO_inds)):
                    raise ValueError('number of electrons in CC gamma_active not correct')
                


                RDM2_CC = embedded_cc_obj.make_rdm2() # from coupled cluster calc
                RDM2_CC_transformed = np.transpose(RDM2_CC, (0, 2, 1, 3)).reshape([RDM2_CC.shape[0]**2, RDM2_CC.shape[0]**2])   # ijkl --> ij, kl matrix 
                N=2*len(self.active_MO_inds)    
                if not np.isclose(0.5*np.trace(RDM2_CC_transformed), 0.5*N*(N-1)):
                    raise ValueError('RDM 2e CC gamma_active not correct')        


            # Get gamma_active embedded
            EMBEDDED_occupied_orbs = self.full_system_EMBEDDED_scf.mo_coeff[:,self.full_system_EMBEDDED_scf.mo_occ>0]

            # optimized embedded denisty matrix
            self.density_emb_active = 2 * EMBEDDED_occupied_orbs @ EMBEDDED_occupied_orbs.conj().T

            ## check number of electrons makes sense:
            if not np.isclose(np.trace(self.density_emb_active@self.S_ovlp), 2*len(self.active_MO_inds)):
                raise ValueError('number of electrons in gamma_active not correct')

            return None

        def Get_embedded_energy(self, check_E_with_pyscf=True):

            if self.density_emb_active is None:
                self.Get_optimized_embedded_dm()

            self.E_act, J_act, K_act, e_xc_act, v_xc_act = self.Get_energy_from_dm(self.dm_active, 
                                                                              check_E_with_pyscf=check_E_with_pyscf)

            self.E_env, J_env, K_env, e_xc_env, v_xc_env = self.Get_energy_from_dm(self.dm_enviro, 
                                                                              check_E_with_pyscf=check_E_with_pyscf)

            
            j_cross = 0.5 * (np.einsum('ij,ij', self.dm_active, J_env) + np.einsum('ij,ij', self.dm_enviro, J_act))
            k_cross = 0.0 # included in j_cross for pyscf!

            xc_cross = self.e_xc_total - e_xc_act - e_xc_env
            self.two_e_cross = j_cross + k_cross + xc_cross


            # J_emb, K_emb =EMBEDDED_full_system_scf.get_jk(dm=self.density_emb_active) 
            # NOTE how normal H_core is being used!
            # E_act_emb = np.einsum('ij,ij', self.density_emb_active, self.standard_hcore + 0.5 * J_emb - 0.25 * K_emb)
            E_act_emb = self.full_system_scf.energy_elec(dm=self.density_emb_active)[0]

            self.E_embedding_correction = np.einsum('ij,ij', self.v_emb, self.density_emb_active - self.dm_active)
            
            Energy_embedding = E_act_emb + self.E_env + self.two_e_cross + self.full_system_scf.energy_nuc() + self.E_embedding_correction
            
            if self.high_level_scf_method == 'RKS':
                if not np.isclose(Energy_embedding, self.full_system_scf.e_tot):
                    # when same functional is used, this should be the same as standard global SCF calc using low level method
                    raise ValueError('Embedding energy not matching global calc')

            return Energy_embedding

        def High_level_SCF(self):


            if self.high_level_scf_method == 'RKS':

                pyscf_mol_input = deepcopy(self.full_system_mol)
                pyscf_mol_input.nelectron = 2*len(self.active_MO_inds) # re-define number of electrons!

                self.full_system_scf_HIGH_LEVEL = scf.RKS(pyscf_mol_input)
                self.full_system_scf_HIGH_LEVEL.verbose = self.pyscf_print_level
                self.full_system_scf_HIGH_LEVEL.max_memory= self.memory
                self.full_system_scf_HIGH_LEVEL.conv_tol = self.E_convergence_tol
                self.full_system_scf_HIGH_LEVEL.xc = self.high_level_xc_functional # <-- BETTER functional!

                # NOTE: do NOT run full_system_scf_HIGH_LEVEL.kernel(), instead use embedded density matrix
                E_act_emb_HIGH_LVL = self.full_system_scf_HIGH_LEVEL.energy_elec(dm=self.density_emb_active)[0]

            elif self.high_level_scf_method == 'CCSD':

                pyscf_mol_input = deepcopy(self.full_system_mol)
                pyscf_mol_input.nelectron = 2*len(self.active_MO_inds) # re-define number of electrons!

                self.full_system_scf_HIGH_LEVEL = scf.RHF(pyscf_mol_input) # <-- HF calc
                self.full_system_scf_HIGH_LEVEL.verbose = self.pyscf_print_level
                self.full_system_scf_HIGH_LEVEL.max_memory= self.memory
                self.full_system_scf_HIGH_LEVEL.conv_tol = self.E_convergence_tol

                E_HF_high_level = self.full_system_scf_HIGH_LEVEL.energy_elec(dm=self.density_emb_active)[0]

                cc_standard_obj = cc.CCSD(self.full_system_scf_HIGH_LEVEL)

                E_cc_standard_with_embedded_Density = cc_standard_obj.energy(self.t1, self.t2, self.eris) ## CC calc, with embedded gamma_active and standard CC obj (no embedding)

                # cc_standard_obj.nocc = self.full_system_mol_EMBEDDED.nelectron // 2
                # cc_standard_obj.nmo = self.full_system_EMBEDDED_scf.mo_energy.size
                # cc_standard_obj.mo_coeff = self.full_system_EMBEDDED_scf.mo_coeff
                # eris = cc_standard_obj.ao2mo(mo_coeff=self.full_system_EMBEDDED_scf.mo_coeff) # embedded gamma_active, with NORMAL CC obj (no embedding)
                # E_cc_standard_with_embedded_Density = cc_standard_obj.energy(self.t1, self.t2, eris) ## CC calc, with embedded gamma_active

                E_act_emb_HIGH_LVL = E_HF_high_level + E_cc_standard_with_embedded_Density

            E_high_lvl = E_act_emb_HIGH_LVL + self.E_env + self.two_e_cross + self.full_system_scf.energy_nuc() + self.E_embedding_correction

            return E_high_lvl




class VQE_embedded():

    def __init__(self, C_matrix_optimized_embedded_OCC_and_VIRT, H_core_standard, nuclear_energy, PySCF_full_system_mol_obj, active_MO_inds):


        self.C_matrix_optimized_embedded_OCC_and_VIRT = C_matrix_optimized_embedded_OCC_and_VIRT
        self.H_core_standard = H_core_standard
        self.H_fermionic = None
        self.H_qubit = None
        self.nuclear_energy = nuclear_energy
        self.active_MO_inds =  active_MO_inds # in spatial (NOT spin) basis

        self.full_system_mol = PySCF_full_system_mol_obj


    def Get_one_and_two_body_integrals(self):
        #2D array for one-body Hamiltonian (H_core) in the MO representation
        # A 4-dimension array for electron repulsion integrals in the MO
        # representation.  The integrals are computed as
        # h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy

        canonical_orbitals  = self.C_matrix_optimized_embedded_OCC_and_VIRT # includes virtual MOs!


        # one_body_integrals
        one_body_integrals =  canonical_orbitals.conj().T @ self.H_core_standard @ canonical_orbitals # < psi_MO | H_core | psi)MO > # one_body_integrals

        # two_body_integrals
        eri = ao2mo.kernel(self.full_system_mol,
                            canonical_orbitals)

        n_orbitals = canonical_orbitals.shape[1]
        two_body_integrals = ao2mo.restore(1, # no permutation symmetry
                              eri, 
                            n_orbitals)

        # two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order='C') # change to physists ordering

        return one_body_integrals, two_body_integrals

    def get_active_space_integrals(self, one_body_integrals,
                                   two_body_integrals,
                                   occupied_indices=None,
                                   active_indices=None):
        """Restricts a molecule at a spatial orbital level to an active space
        This active space may be defined by a list of active indices and
            doubly occupied indices. Note that one_body_integrals and
            two_body_integrals must be defined
            n an orthonormal basis set.
        Args:
            one_body_integrals: One-body integrals of the target Hamiltonian
            two_body_integrals: Two-body integrals of the target Hamiltonian
            occupied_indices: A list of spatial orbital indices
                indicating which orbitals should be considered doubly occupied.
            active_indices: A list of spatial orbital indices indicating
                which orbitals should be considered active.
        Returns:
            tuple: Tuple with the following entries:
            **core_constant**: Adjustment to constant shift in Hamiltonian
            from integrating out core orbitals
            **one_body_integrals_new**: one-electron integrals over active
            space.
            **two_body_integrals_new**: two-electron integrals over active
            space.
        """
        # Fix data type for a few edge cases
        occupied_indices = [] if occupied_indices is None else occupied_indices
        if (len(active_indices) < 1):
            raise ValueError('Some active indices required for reduction.')

        # Determine core constant
        core_constant = 0.0
        for i in occupied_indices:
            core_constant += 2 * one_body_integrals[i, i]
            for j in occupied_indices:
                core_constant += (2 * two_body_integrals[i, j, j, i] -
                                  two_body_integrals[i, j, i, j])

        # Modified one electron integrals
        one_body_integrals_new = np.copy(one_body_integrals)
        for u in active_indices:
            for v in active_indices:
                for i in occupied_indices:
                    one_body_integrals_new[u, v] += (
                        2 * two_body_integrals[i, u, v, i] -
                        two_body_integrals[i, u, i, v])

        # Restrict integral ranges and change M appropriately
        return (core_constant,
                one_body_integrals_new[np.ix_(active_indices, active_indices)],
                two_body_integrals[np.ix_(active_indices, active_indices,
                                             active_indices, active_indices)])


    def spatial_to_spin(self, one_body_integrals, two_body_integrals):
        """
        Convert from spatial MOs to spin MOs
        """
        n_qubits = 2*one_body_integrals.shape[0]

        one_body_terms = np.zeros((n_qubits, n_qubits))
        two_body_terms = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

        for p in range(n_qubits//2):
            for q in range(n_qubits//2):
                
                one_body_terms[2*p, 2*q] = one_body_integrals[p,q] # spin UP
                one_body_terms[(2*p + 1), (2*q +1)] = one_body_integrals[p,q] # spin DOWN
                
                # continue 2-body terms
                for r in range(n_qubits//2):
                    for s in range(n_qubits//2):
                                        
                        ### SAME spin                
                        two_body_terms[2*p, 2*q , 2*r, 2*s] = two_body_integrals[p,q,r,s] # up up up up
                        two_body_terms[(2*p+1), (2*q +1) , (2*r + 1), (2*s +1)] = two_body_integrals[p,q,r,s] # down down down down
                        
                        ### mixed spin                
                        two_body_terms[2*p, 2*q , (2*r + 1), (2*s +1)] = two_body_integrals[p,q,r,s] # up up down down
                        two_body_terms[(2*p+1), (2*q +1) , 2*r, 2*s] = two_body_integrals[p,q,r,s] # down down up up            
                        
        ### remove vanishing terms
        EQ_Tolerance=1e-8
        one_body_terms[np.abs(one_body_terms)<EQ_Tolerance]=0
        two_body_terms[np.abs(two_body_terms)<EQ_Tolerance]=0

        return one_body_terms, two_body_terms

    def Get_H_fermionic(self):
        

        one_body_integrals_FULL, two_body_integrals_FULL = self.Get_one_and_two_body_integrals()

        active_space_const, one_body_integrals_active, two_body_integrals_active = self.get_active_space_integrals(one_body_integrals_FULL,
                                                                                                                   two_body_integrals_FULL,
                                                                                                                   occupied_indices=None,
                                                                                                                   active_indices=self.active_MO_inds) # only consider active MOs

        # convert to spin orbs
        one_body_terms, two_body_terms = self.spatial_to_spin(one_body_integrals_active, two_body_integrals_active)


        # build fermionic Hamiltonian
        H_fermionic = FermionOperator((),  self.nuclear_energy + active_space_const)
        # two_body_terms = two_body_terms.transpose(0,2,3,1) # for physist notation!

        # one body terms
        for p in range(one_body_terms.shape[0]):
            for q in range(one_body_terms.shape[0]):
                
                H_fermionic += one_body_terms[p,q] * FermionOperator(((p, 1), (q, 0)))
                
                # two body terms
                for r in range(two_body_terms.shape[0]):
                    for s in range(two_body_terms.shape[0]):
                        
                        ######## physist notation
                        ## (requires:
                        ##           two_body_terms transpose (0,2,3,1) before loop starts!
                        ##)
        #                 H_qubit += 0.5*two_body_terms[p,q,r,s] * FermionOperator(((p, 1), (q, 1), (r,0), (s, 0)))
                        
                        ######## chemist notation
                        H_fermionic += 0.5*two_body_terms[p,q,r,s] * FermionOperator(((p, 1), (r, 1), (s,0), (q, 0)))

        self.H_fermionic = H_fermionic
        return None

    def Get_H_qubit(self):
        
        if self.H_fermionic is None:
            self.Get_H_fermionic()

        self.H_qubit = jordan_wigner(self.H_fermionic)
        return None


    def Get_qubit_Energy(self):
        from openfermion.linalg import get_sparse_operator

        if self.H_qubit is None:
            self.Get_H_qubit()

        H_JW_mat = get_sparse_operator(self.H_qubit)
        eigvals_EMBED, eigvecs_EMBED = sp.sparse.linalg.eigsh(H_JW_mat, which='SA', k=1)

        # print(np.binary_repr(np.where(eigvecs_EMBED>1e-2)[0][0]).count('1') )
        # print(eigvals_EMBED)

        return eigvals_EMBED, eigvecs_EMBED

