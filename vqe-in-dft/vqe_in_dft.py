import scipy as sp
from pyscf import gto, dft, lib, mp, cc, scf, tools, ci, fci, lo, ao2mo
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import jordan_wigner

class human():
    def __init__(self, size):
        hello

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

    def plot_molecule_3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(self.N_active_atoms):
            ax.scatter(*self.geometry[i][1:][0], marker='o', color='green', label='systemA')

        for coord in self.geometry[self.N_active_atoms:]:
            ax.scatter(*coord[1:][0], marker='o', color='red', label='systemB')

        plt.legend()
        plt.show()


    def _run_fci(self):

        HF_scf = scf.RHF(self.full_system_mol)
        HF_scf.verbose= self.pyscf_print_level
        HF_scf.max_memory= self.memory
        HF_scf.conv_tol = self.E_convergence_tol
        HF_scf.kernel()

        self.my_fci = fci.FCI(HF_scf).run()
        # print('E(UHF-FCI) = %.12f' % self.my_fci.e_tot)

        self.E_FCI = self.my_fci.e_tot

        return self.my_fci.e_tot

# myci = ci.CISD(HF_scf).run() # this is UCISD
# print('UCISD total energy = ', myci.e_tot)

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


        def _Localize_orbitals(self, localization_method, sanity_check=False, save_localized_orbs=False):

            # Take occupied orbitals of global system calc
            occupied_orbs = self.full_system_scf.mo_coeff[:,self.full_system_scf.mo_occ>0]

            # run localization scheme
            if localization_method.lower() == 'spade':

                # PsiEmbed
                orthogonal_orbitals = occupied_orbs[:self.N_active_atoms, :] # take active rows and all columns

                #### Alexis
                # AO_slice_matrix = self.full_system_mol.aoslice_by_atom()
                # # active_ao_ind=[]
                # # for atm_i, atm_str in enumerate([atom_str for atom_str, _ in self.full_system_mol.atom][:self.N_active_atoms]):
                # #     slice_ind = list(range(AO_slice_matrix[atm_i, 2], AO_slice_matrix[atm_i, 3]))
                # #     active_ao_ind = [*active_ao_ind, *slice_ind]
                # # active_ao_ind = np.array(active_ao_ind)

                # active_ao_ind = np.arange(AO_slice_matrix[0,2], AO_slice_matrix[self.N_active_atoms,3]) # gets first and final ind 
                # orthogonal_orbitals = occupied_orbs[active_ao_ind, :]

                # Correct from here
                u, singular_values, rotation_matrix = np.linalg.svd(orthogonal_orbitals, full_matrices=True)

                # find where largest step change 
                # delta_s = singular_values[1:] - singular_values[:-1]
                delta_s = [(i, (singular_values[i] - singular_values[i+1] )) for i in range(len(singular_values) - 1)] # contains (index, delta_s)

                self.n_act_mos = max(delta_s, key=lambda x: x[0])[0] + 1 # finds index where largest step change is! (adds 1 for python indexing)
                self.n_env_mos = len(singular_values) - self.n_act_mos

                # define active and environment orbitals from localization
                self.act_orbitals = occupied_orbs @ rotation_matrix.T[:, :self.n_act_mos]
                self.env_orbitals = occupied_orbs @ rotation_matrix.T[:, self.n_act_mos:]

                self.C_matrix_all_localized_orbitals = occupied_orbs @ rotation_matrix.T

                self.active_ao_ind  = np.arange(self.n_act_mos)
                self.enviro_ao_ind = np.arange(self.n_act_mos, self.n_act_mos+self.n_env_mos)

            else:

                THRESHOLD = 1.5 # TODO: Remove this

                PM = lo.PipekMezey(self.full_system_mol, occupied_orbs)
                PM.pop_method = localization_method # 'mulliken', 'meta-lowdin', 'iao', 'becke'
                C_loc_occ = PM.kernel() # <--  NEW C matrix (where orbitals are localized to atoms)

                dm_localised = 2* C_loc_occ @ C_loc_occ.conj().T
                PS_matrix = dm_localised @ self.S_ovlp

                AO_slice_matrix = self.full_system_mol.aoslice_by_atom()
                active_ao_ind=[]
                atomic_charges_list = self.full_system_mol.atom_charges()
                for atm_i, atm_str in enumerate([atom_str for atom_str, _ in self.full_system_mol.atom][:self.N_active_atoms]):
                    slice_ind = list(range(AO_slice_matrix[atm_i, 2], AO_slice_matrix[atm_i, 3]))
                    
                    # print(atm_str)
                    qA = atomic_charges_list[atm_i] #atomic charge of atom_i
                    for k in slice_ind:
                        mulliken_charge = qA - PS_matrix[k,k]  # mulliken charge of ao centred on atom_i
                        mulliken_population = PS_matrix[k,k]  # mulliken pop of ao centred on atom_i
                #         print(mulliken_charge)
                        # print(mulliken_population)
                        if mulliken_population>THRESHOLD: ### <--- not sure about this THRESHOLD
                            active_ao_ind.append(k)

                self.active_ao_ind = active_ao_ind
                self.enviro_ao_ind = [ind for ind in range(C_loc_occ.shape[1]) if ind not in active_ao_ind]

                self.n_act_mos = len(self.active_ao_ind)
                self.n_env_mos = len(self.enviro_ao_ind)
                # define active and environment orbitals from localization
                self.act_orbitals = C_loc_occ[:, self.active_ao_ind]
                self.env_orbitals = C_loc_occ[:, self.enviro_ao_ind]

                self.C_matrix_all_localized_orbitals = C_loc_occ

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
            

            if save_localized_orbs:
                # Save localized orbitals as molden file
                with open('LOCALIZED_orbs.molden', 'w') as outfile:
                    tools.molden.header(self.full_system_mol,
                                        outfile)
                    tools.molden.orbital_coeff(
                                        self.full_system_mol,
                                         outfile, 
                                         self.act_orbitals, # <- active orbitals!
                                         ene=self.full_system_scf.mo_energy[np.array(self.active_ao_ind)],
                                         occ=self.full_system_scf.mo_occ[np.array(self.active_ao_ind)])

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

            # V_embed = G[𝛾_act + 𝛾_env] − G[𝛾_act]+Projector
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
                _Localize_orbitals()

            # Define embedded system
            self.full_system_mol_EMBEDDED = gto.Mole(atom= self.geometry,
                      basis=self.basis,
                       charge=self.charge,
                       spin= self.spin,
                      )

            self.full_system_mol_EMBEDDED.build()

            # RE-DEFINE number of electrons in system
            self.full_system_mol_EMBEDDED.nelectron = 2*len(self.active_ao_ind)


            if (self.low_level_scf_method == 'RKS' and self.high_level_scf_method=='RKS'):
                self.full_system_EMBEDDED_scf = scf.RKS(self.full_system_mol_EMBEDDED) # <-- DFT calculation
                self.full_system_EMBEDDED_scf.verbose = self.pyscf_print_level
                self.full_system_EMBEDDED_scf.max_memory= self.memory
                self.full_system_EMBEDDED_scf.conv_tol = self.E_convergence_tol
                self.full_system_EMBEDDED_scf.xc = self.low_level_xc_functional

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
                if not np.isclose(np.trace(RDM1_CC), 2*len(self.active_ao_ind)):
                    raise ValueError('number of electrons in CC gamma_active not correct')
                


                RDM2_CC = embedded_cc_obj.make_rdm2() # from coupled cluster calc
                RDM2_CC_transformed = np.transpose(RDM2_CC, (0, 2, 1, 3)).reshape([RDM2_CC.shape[0]**2, RDM2_CC.shape[0]**2])   # ijkl --> ij, kl matrix 
                N=2*len(self.active_ao_ind)    
                if not np.isclose(0.5*np.trace(RDM2_CC_transformed), 0.5*N*(N-1)):
                    raise ValueError('RDM 2e CC gamma_active not correct')        

            # Get gamma_active embedded
            EMBEDDED_occupied_orbs = self.full_system_EMBEDDED_scf.mo_coeff[:,self.full_system_EMBEDDED_scf.mo_occ>0]

            # optimized embedded denisty matrix
            self.density_emb_active = 2 * EMBEDDED_occupied_orbs @ EMBEDDED_occupied_orbs.conj().T

            ## check number of electrons makes sense:
            if not np.isclose(np.trace(self.density_emb_active@self.S_ovlp), 2*len(self.active_ao_ind)):
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
                pyscf_mol_input.nelectron = 2*len(self.active_ao_ind) # re-define number of electrons!

                self.full_system_scf_HIGH_LEVEL = scf.RKS(pyscf_mol_input)
                self.full_system_scf_HIGH_LEVEL.verbose = self.pyscf_print_level
                self.full_system_scf_HIGH_LEVEL.max_memory= self.memory
                self.full_system_scf_HIGH_LEVEL.conv_tol = self.E_convergence_tol
                self.full_system_scf_HIGH_LEVEL.xc = self.high_level_xc_functional # <-- BETTER functional!

                # NOTE: do NOT run full_system_scf_HIGH_LEVEL.kernel(), instead use embedded density matrix
                E_act_emb_HIGH_LVL = self.full_system_scf_HIGH_LEVEL.energy_elec(dm=self.density_emb_active)[0]

            elif self.high_level_scf_method == 'CCSD':

                pyscf_mol_input = deepcopy(self.full_system_mol)
                pyscf_mol_input.nelectron = 2*len(self.active_ao_ind) # re-define number of electrons!

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




class VQE_embedded(embedded_molecular_system):

    def __init__(self, geometry, N_active_atoms, localization_method, projection_method, mu_level_shift=1e6,

               low_level_scf_method='RKS', low_level_xc_functional= 'lda, vwn', E_convergence_tol = 1e-6, basis = 'STO-3G', output_file_name='output.dat',
                unit= 'angstrom', pyscf_print_level=1, memory=4000, charge=0, spin=0, run_fci=False):

        high_level_scf_method = 'CCSD'
        high_level_xc_functional= None
        super().__init__(geometry, N_active_atoms, localization_method, projection_method, high_level_scf_method, high_level_xc_functional, mu_level_shift,
                           low_level_scf_method, low_level_xc_functional, E_convergence_tol, basis, output_file_name,
                            unit, pyscf_print_level, memory, charge, spin, run_fci)



    def Get_one_and_two_body_integrals_embedded(self):
        #2D array for one-body Hamiltonian (H_core) in the MO representation
        # A 4-dimension array for electron repulsion integrals in the MO
        # representation.  The integrals are computed as
        # h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy
        if self.dm_active is None:
            self._Get_embedded_potential()


        canonical_orbitals  = self.C_matrix_all_localized_orbitals

        # one_body_integrals
        embedded_hcore = self.v_emb + self.standard_hcore 
        one_body_integrals = canonical_orbitals.conj().T @ embedded_hcore @ canonical_orbitals

        # two_body_integrals
        pyscf_mol_input = deepcopy(self.full_system_mol)
        pyscf_mol_input.nelectron = 2*len(self.active_ao_ind) # re-define number of electrons!

        eri = ao2mo.kernel(pyscf_mol_input,
                            canonical_orbitals)

        n_orbitals = canonical_orbitals.shape[1]
        eri = ao2mo.restore(1, # no permutation symmetry
                              eri, 
                            n_orbitals)

        two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order='C')

        return one_body_integrals, two_body_integrals

    def Get_one_and_two_body_integrals_STANDARD(self):
        #2D array for one-body Hamiltonian (H_core) in the MO representation
        # A 4-dimension array for electron repulsion integrals in the MO
        # representation.  The integrals are computed as
        # h[p,q,r,s]=\int \phi_p(x)* \phi_q(y)* V_{elec-elec} \phi_r(y) \phi_s(x) dxdy

        canonical_orbitals  = self.full_system_scf.mo_coeff[:,self.full_system_scf.mo_occ>0]
        

        # one_body_integrals
        one_body_integrals = canonical_orbitals.conj().T @ self.standard_hcore @ canonical_orbitals

        # two_body_integrals
        eri = ao2mo.kernel(self.full_system_mol,
                            canonical_orbitals)

        n_orbitals = canonical_orbitals.shape[1]
        eri = ao2mo.restore(1, # no permutation symmetry
                              eri, 
                            n_orbitals)

        two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order='C')

        return one_body_integrals, two_body_integrals

    def spinorb_from_spatial(self, one_body_integrals, two_body_integrals):
        EQ_TOLERANCE=1e-8
        n_qubits = 2 * one_body_integrals.shape[0]

        # Initialize Hamiltonian coefficients.
        one_body_coefficients = np.zeros((n_qubits, n_qubits))
        two_body_coefficients = np.zeros(
            (n_qubits, n_qubits, n_qubits, n_qubits))
        # Loop through integrals.
        for p in range(n_qubits // 2):
            for q in range(n_qubits // 2):

                # Populate 1-body coefficients. Require p and q have same spin.
                one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
                one_body_coefficients[2 * p + 1, 2 * q +
                                      1] = one_body_integrals[p, q]
                # Continue looping to prepare 2-body coefficients.
                for r in range(n_qubits // 2):
                    for s in range(n_qubits // 2):

                        # Mixed spin
                        two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 *
                                              s] = (two_body_integrals[p, q, r, s])
                        two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                                              1] = (two_body_integrals[p, q, r, s])

                        # Same spin
                        two_body_coefficients[2 * p, 2 * q, 2 * r, 2 *
                                              s] = (two_body_integrals[p, q, r, s])
                        two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * r +
                                              1, 2 * s +
                                              1] = (two_body_integrals[p, q, r, s])

        # Truncate.
        one_body_coefficients[
            np.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
        two_body_coefficients[
            np.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.

        return one_body_coefficients, two_body_coefficients



    def Get_H_embedded_for_VQE(self):
        

        constant = self.full_system_scf.energy_nuc() 

        one_body_integrals, two_body_integrals = self.Get_one_and_two_body_integrals_embedded()

        one_body_coefficients, two_body_coefficients = self.spinorb_from_spatial(one_body_integrals, two_body_integrals)

        molecular_hamiltonian = InteractionOperator(constant,
                                                    one_body_coefficients,
                                                    1 / 2 * two_body_coefficients)

        # n_qubits = two_body_coefficients.shape[0]

        
        Qubit_Hamiltonian = jordan_wigner(molecular_hamiltonian)

        return Qubit_Hamiltonian


    def Get_H_standard_for_VQE(self):
        

        constant = self.full_system_scf.energy_nuc() 

        one_body_integrals, two_body_integrals = self.Get_one_and_two_body_integrals_STANDARD()

        one_body_coefficients, two_body_coefficients = self.spinorb_from_spatial(one_body_integrals, two_body_integrals)

        molecular_hamiltonian = InteractionOperator(constant,
                                                    one_body_coefficients,
                                                    1 / 2 * two_body_coefficients)

        # n_qubits = two_body_coefficients.shape[0]

        
        Qubit_Hamiltonian = jordan_wigner(molecular_hamiltonian)

        return Qubit_Hamiltonian
