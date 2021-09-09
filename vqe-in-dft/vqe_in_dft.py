import scipy as sp
from pyscf import gto, dft, lib, mp, cc, scf, tools, ci, fci, lo, ao2mo
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from openfermion.transforms import jordan_wigner
from openfermion.ops import FermionOperator
from openfermion.linalg import get_sparse_operator
import py3Dmol
from pyscf.tools import cubegen
import os
from copy import deepcopy

# pip install py3Dmol
# conda install conda-forge rdkit

# may need to do:
# pip install protobuf==3.13.0

def Get_xyz_string(PySCF_mol_obj):
    if isinstance(PySCF_mol_obj.atom, str):
        # xyz_str = self.PySCF_scf_obj.atom
        raise ValueError('Currently not accepting string inputs of geometry of pyscf object')
    else:
        n_atoms = len(PySCF_mol_obj.atom)
        xyz_str=f'{n_atoms}'
        xyz_str+='\n \n'
        for atom, xyz in PySCF_mol_obj.atom:
            xyz_str+= f'{atom}\t{xyz[0]}\t{xyz[1]}\t{xyz[2]}\n'
    return xyz_str

def Draw_molecule(xyz_string, width=400, height=400, jupyter_notebook=False):

    if jupyter_notebook is True:
        import rdkit
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import IPythonConsole
        rdkit.Chem.Draw.IPythonConsole.ipython_3d = True  # enable py3Dmol inline visualization

    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_string, "xyz")
    view.setStyle({'stick':{}})
    view.zoomTo()
    return(view.show())
    
def Localize_orbitals(localization_method, PySCF_scf_obj, N_active_atoms, THRESHOLD=None, sanity_check=True):

    if PySCF_scf_obj.mo_coeff is None:
        raise ValueError('need to perform SCF calculation before localization')


    S_ovlp = PySCF_scf_obj.get_ovlp()
    AO_slice_matrix = PySCF_scf_obj.mol.aoslice_by_atom()

    # run localization scheme
    if localization_method.lower() == 'spade':

        # Take occupied orbitals of global system calc
        occupied_orbs = PySCF_scf_obj.mo_coeff[:,PySCF_scf_obj.mo_occ>0]

        # Get active AO indices
        N_active_AO = AO_slice_matrix[N_active_atoms-1][3]  # find max AO index for active atoms (neg 1 as python indexs from 0)

        
        S_half = sp.linalg.fractional_matrix_power(S_ovlp , 0.5)
        orthogonal_orbitals = (S_half@occupied_orbs)[:N_active_AO, :] # Get rows (the active AO) of orthogonal orbs 

        # Comute singular vals
        u, singular_values, rotation_matrix = np.linalg.svd(orthogonal_orbitals, full_matrices=True)

        # find where largest step change 
        delta_s = singular_values[:-1] - singular_values[1:] # σ_i - σ_(i+1)
        print('delta_singular_vals:')
        print(delta_s, '\n')

        n_act_mos = np.argmax(delta_s)+1 # add one due to python indexing
        n_env_mos = len(singular_values) - n_act_mos

        # define active and environment orbitals from localization
        act_orbitals = occupied_orbs @ rotation_matrix.T[:, :n_act_mos]
        env_orbitals = occupied_orbs @ rotation_matrix.T[:, n_act_mos:]

        C_matrix_all_localized_orbitals = occupied_orbs @ rotation_matrix.T

        active_MO_inds  = np.arange(n_act_mos)
        enviro_MO_inds = np.arange(n_act_mos, n_act_mos+n_env_mos)

    else:
        if not isinstance(THRESHOLD, float):
            raise ValueError ('if localization method is not SPADE then a threshold parameter is requried to choose active system')

        # Take C (FULL) matrix from SCF calc 
        opt_C = PySCF_scf_obj.mo_coeff

        # run localization scheme
        if localization_method.lower() == 'pipekmezey':
            ### PipekMezey
            PM = lo.PipekMezey(PySCF_scf_obj.mol, opt_C)
            PM.pop_method = 'mulliken' # 'meta-lowdin', 'iao', 'becke'
            C_loc = PM.kernel() # includes virtual orbs too!
            C_loc_occ = C_loc[:,PySCF_scf_obj.mo_occ>0]

        elif localization_method.lower() == 'boys':
            ### Boys
            boys_SCF = lo.boys.Boys(PySCF_scf_obj.mol, opt_C)
            C_loc  = boys_SCF.kernel()
            C_loc_occ = C_loc[:,PySCF_scf_obj.mo_occ>0]

        elif localization_method.lower() == 'ibo':
            ### intrinsic bonding orbs
            #
            # mo_occ = PySCF_scf_obj.mo_coeff[:,PySCF_scf_obj.mo_occ>0]
            # iaos = lo.iao.iao(PySCF_scf_obj.mol, mo_occ)
            # # Orthogonalize IAO
            # iaos = lo.vec_lowdin(iaos, S_ovlp)
            # C_loc_occ = lo.ibo.ibo(PySCF_scf_obj.mol, mo_occ, locmethod='IBO', iaos=iaos)#.kernel()


            iaos = lo.iao.iao(PySCF_scf_obj.mol, PySCF_scf_obj.mo_coeff)
            # Orthogonalize IAO
            iaos = lo.vec_lowdin(iaos, S_ovlp)
            C_loc = lo.ibo.ibo(PySCF_scf_obj.mol, PySCF_scf_obj.mo_coeff, locmethod='IBO', iaos=iaos)#.kernel()
            C_loc_occ = C_loc[:,PySCF_scf_obj.mo_occ>0]
        else:
            raise ValueError(f'unknown localization method {localization_method}')
        
                        
        # find indices of AO of active atoms
        ao_active_inds = np.arange(AO_slice_matrix[0,2], AO_slice_matrix[N_active_atoms-1,3])




        #### my method using S_matrix ####
        MO_AO_overlap = S_ovlp@C_loc_occ  #  < ϕ_AO_i | ψ_MO_j >
        MO_active_AO_overlap = np.einsum('ij->j', MO_AO_overlap[ao_active_inds]) # sum over rows of active AOs of MOs!

        print('\noverlap:', MO_active_AO_overlap)
        print(f'threshold for active part: {THRESHOLD} \n')

        active_MO_inds = np.where(MO_active_AO_overlap>THRESHOLD)[0]
        enviro_MO_inds = np.array([i for i in range(C_loc_occ.shape[1]) if i not in active_MO_inds]) # get all non active MOs

        # #### mulliken charge method ####
        # # Use threshold of 1
        # dm_loc = 2 * C_loc_occ @ C_loc_occ.conj().T
        # PS_mu = np.diag(dm_loc@S_ovlp)[ao_active_inds]
        # print('\nactive occupancies:', PS_mu)
        # print(f'threshold for active part: {THRESHOLD} \n')
        # inds_above_thres = np.where(PS_mu>THRESHOLD)[0]
        # active_MO_inds = ao_active_inds[inds_above_thres]
        # enviro_MO_inds = np.array([i for i in range(C_loc_occ.shape[1]) if i not in active_MO_inds]) # get all non active MOs





        # define active MO orbs and environment
        act_orbitals = C_loc[:, active_MO_inds] # take MO (columns of C_matrix) that have high dependence from active AOs
        env_orbitals = C_loc[:, enviro_MO_inds]
        
        C_matrix_all_localized_orbitals = C_loc_occ

        n_act_mos = len(active_MO_inds)
        n_env_mos = len(enviro_MO_inds)


    print(f'number of active MOs: {n_act_mos}')
    print(f'number of enviro MOs: {n_env_mos} \n')

    return act_orbitals, env_orbitals, C_matrix_all_localized_orbitals, active_MO_inds, enviro_MO_inds # C_active, C_enviro, C_all_localized, active_MO_inds, enviro_MO_inds

def Draw_cube_orbital(PySCF_scf_obj, xyz_string, C_matrix, index_list, width=400, height=400, jupyter_notebook=False):
    """

    """
    if jupyter_notebook is True:
        import rdkit
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import IPythonConsole
        rdkit.Chem.Draw.IPythonConsole.ipython_3d = True  # enable py3Dmol inline visualization

    if not set(index_list).issubset(set(range(C_matrix.shape[1]))):
        raise ValueError('list of MO indices to plot is outside of C_matrix columns')

    plotted_orbitals = []
    for index in index_list:

        File_name = f'temp_MO_orbital_index{index}.cube'
        cubegen.orbital(PySCF_scf_obj.mol, File_name, C_matrix[:, index])
        
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

def Get_active_and_envrio_dm(PySCF_scf_obj, C_active, C_envrio, C_all_localized, sanity_check=True):

    # get density matrices
    dm_active =  2 * C_active @ C_active.T
    dm_enviro =  2 * C_envrio @ C_envrio.T

    if sanity_check:

        S_ovlp = PySCF_scf_obj.get_ovlp()

        ## check number of electrons is still the same after orbitals have been localized (change of basis)
        N_active_electrons = np.trace(dm_active@S_ovlp)
        N_enviro_electrons = np.trace(dm_enviro@S_ovlp)
        N_all_electrons = PySCF_scf_obj.mol.nelectron

        bool_flag_electron_number = np.isclose(( N_active_electrons + N_enviro_electrons), N_all_electrons)
        if not bool_flag_electron_number:
            raise ValueError('number of electrons in localized orbitals is incorrect')
        print(f'N_active_elec + N_environment_elec = N_total_elec is: {bool_flag_electron_number}')

        # checking denisty matrix parition makes sense:
        dm_localised_full_system = 2* C_all_localized@ C_all_localized.conj().T
        bool_density_flag = np.allclose(dm_localised_full_system, dm_active + dm_enviro)
        if not bool_density_flag:
            raise ValueError('gamma_full != gamma_active + gamma_enviro')
        print(f'y_active + y_enviro = y_total is: {bool_density_flag}')

    return dm_active, dm_enviro

def check_Hcore_is_standard_Hcore(PySCF_scf_obj):
    H_core_standard = scf.hf.get_hcore(PySCF_scf_obj.mol) # calculate Hcore using PySCF inbuilt.
    H_core_in_SCF_calculation = PySCF_scf_obj.get_hcore() #

    H_core_is_standard = np.allclose(H_core_standard, H_core_in_SCF_calculation)

    if H_core_is_standard:
        print('H core is standard H_core')
    else:
        print('H core is NOT standard H_core')

    return H_core_is_standard

def Get_cross_terms(PySCF_scf_obj, dm_active, dm_enviro, J_env, J_act, e_xc_act, e_xc_env):
    """
    Get Energy from denisty matrix

    Note this uses the standard hcore (NO embedding potential here!)
    """

    two_e_term_total =  PySCF_scf_obj.get_veff(dm=dm_active+dm_enviro)
    e_xc_total = two_e_term_total.exc

    j_cross = 0.5 * ( np.einsum('ij,ij', dm_active, J_env) + np.einsum('ij,ij', dm_enviro, J_act) )
    k_cross = 0.0

    xc_cross = e_xc_total - e_xc_act - e_xc_env
    two_e_cross = j_cross + k_cross + xc_cross

    return two_e_cross

def Get_embedded_potential_operator(projector_method, PySCF_scf_obj, dm_active, dm_enviro, check_Hcore_is_correct=True, mu_shift_val=1e6, check_Vemb=True):
    """
    
    Args:
        projector_method = huzinaga, mu_shfit
    """

    if check_Hcore_is_correct:
        if not check_Hcore_is_standard_Hcore(PySCF_scf_obj):
            raise ValueError('Hcore is not standard Hcore')

    S_mat = PySCF_scf_obj.get_ovlp()

    if projector_method == 'huzinaga':
        # Fock = PySCF_scf_obj.get_hcore() + PySCF_scf_obj.get_veff(dm=dm_active + dm_enviro)
        Fock = PySCF_scf_obj.get_fock(h1e=PySCF_scf_obj.get_hcore(), dm=dm_active + dm_enviro)
        F_gammaB_S = Fock @ dm_enviro @ S_mat
        # S_gammaB_F = S_mat @ dm_enviro @ Fock
        # print('CHECK:', np.allclose(F_gammaB_S, S_gammaB_F.T))
        projector = -0.5 * (F_gammaB_S + F_gammaB_S.T)
    elif projector_method == 'mu_shfit':
        projector = mu_shift_val * (S_mat @ dm_enviro  @ S_mat)
    else:
        raise ValueError(f'Unknown projection method {projector_method}')


    # define the embedded term
    g_A_and_B = PySCF_scf_obj.get_veff(dm=dm_active+dm_enviro)

    g_A = PySCF_scf_obj.get_veff(dm=dm_active)

    v_emb = g_A_and_B - g_A + projector

    if check_Vemb:
        # PsiEmbed definition
        J_env = PySCF_scf_obj.get_j(dm = dm_enviro)

        J_total = PySCF_scf_obj.get_j(dm = dm_active+dm_enviro)
        two_e_term_total =  PySCF_scf_obj.get_veff(dm=dm_active+dm_enviro)
        v_xc_total = two_e_term_total - J_total

        J_act = PySCF_scf_obj.get_j(dm = dm_active)
        two_e_term_act =  PySCF_scf_obj.get_veff(dm=dm_active)
        v_xc_act = two_e_term_act - J_act

        v_emb2 = (J_env + v_xc_total - v_xc_act + projector)

        if not np.allclose(v_emb, v_emb2):
            raise ValueError('V_embed definition incorrect')

    return v_emb

def Get_energy_and_matrices_from_dm(PySCF_scf_obj, dm_matrix, check_E_with_pyscf=True):
    """
    Get Energy from denisty matrix

    Note this uses the standard hcore (NO embedding potential here!)
    """

    # It seems that PySCF lumps J and K in the J array 
    J_mat = PySCF_scf_obj.get_j(dm = dm_matrix)
    K_mat = np.zeros_like(J_mat)
    two_e_term =  PySCF_scf_obj.get_veff(dm=dm_matrix)
    e_xc = two_e_term.exc
    v_xc = two_e_term - J_mat 

    H_core_standard = scf.hf.get_hcore(PySCF_scf_obj.mol) # No embedding potential
    Energy_elec = np.einsum('ij, ij', dm_matrix, H_core_standard + J_mat/2) + e_xc
    
    if check_E_with_pyscf:
        Energy_elec_pyscf = PySCF_scf_obj.energy_elec(dm=dm_matrix)[0]
        if not np.isclose(Energy_elec_pyscf, Energy_elec):
            raise ValueError('Energy calculation incorrect')

    return Energy_elec, J_mat, K_mat, e_xc, v_xc

def Get_embedded_one_and_two_body_integrals_MO_basis(PySCF_scf_obj_EMBEDDED, N_enviroment_MOs, physists_notation=False):


    # check we have embedded Hcore
    if check_Hcore_is_standard_Hcore(PySCF_scf_obj_EMBEDDED) is not False:
        raise ValueError('Hcore is not standard Hcore')

    C_emb_matrix_occ_and_virt = PySCF_scf_obj_EMBEDDED.mo_coeff
    canonical_orbitals_EMBEDDED  = C_emb_matrix_occ_and_virt[:,:-N_enviroment_MOs] # projector means last orbs are environment!

    # one body terms
    Hcore_embedded = PySCF_scf_obj_EMBEDDED.get_hcore()
    one_body_integrals = canonical_orbitals_EMBEDDED.conj().T @ Hcore_embedded @ canonical_orbitals_EMBEDDED

    ## two body terms
    eri = ao2mo.kernel(PySCF_scf_obj_EMBEDDED.mol, canonical_orbitals_EMBEDDED)
    n_orbitals = canonical_orbitals_EMBEDDED.shape[1]
    eri = ao2mo.restore(1, # no permutation symmetry
                          eri, 
                        n_orbitals)


    if physists_notation is True:
        two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order='C')
    else:
        # chemists notation
        two_body_integrals = eri

    return one_body_integrals, two_body_integrals

def Get_SpinOrbs_from_Spatial(one_body_integrals, two_body_integrals, physists_notation=False, EQ_Tolerance=1e-8):
    n_qubits = 2*one_body_integrals.shape[0]

    one_body_terms = np.zeros((n_qubits, n_qubits))
    two_body_terms = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

    if physists_notation is False:
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
    else:
        for p in range(n_qubits // 2):
            for q in range(n_qubits // 2):

                # Populate 1-body coefficients
                # p and q must have same spin.
                one_body_terms[2 * p, 2 * q] = one_body_integrals[p, q]
                one_body_terms[2 * p + 1, 2 * q + 1] = one_body_integrals[p, q]

                # Populate 2-body coefficients.
                for r in range(n_qubits // 2):
                    for s in range(n_qubits // 2):

                        # Mixed spin
                        two_body_terms[2 * p, 2 * q + 1, 2 * r + 1, 2 *s] = (two_body_integrals[p, q, r, s]) # up down down up
                        two_body_terms[2 * p + 1, 2 * q, 2 * r, 2 * s +1] = (two_body_integrals[p, q, r, s]) # down up up down

                        # Same spin
                        two_body_terms[2 * p, 2 * q, 2 * r, 2 *s] = (two_body_integrals[p, q, r, s]) # up up up up
                        two_body_terms[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s +1] = (two_body_integrals[p, q, r, s]) # down down down down


    ### remove vanishing terms
    one_body_terms[np.abs(one_body_terms)<EQ_Tolerance]=0
    two_body_terms[np.abs(two_body_terms)<EQ_Tolerance]=0

    return one_body_terms, two_body_terms

def Get_fermionic_H(one_body_terms, two_body_terms, Nuclear_energy, core_constant=0,  physists_notation=False):
    H_fermionic = FermionOperator((),  Nuclear_energy + core_constant)

    # one body terms
    if physists_notation:
        for p in range(one_body_terms.shape[0]):
            for q in range(one_body_terms.shape[0]):

                H_fermionic += one_body_terms[p,q] * FermionOperator(((p, 1), (q, 0)))

                # two body terms
                for r in range(two_body_terms.shape[0]):
                    for s in range(two_body_terms.shape[0]):

                        ####### physist notation
                        # differs from chemist notation... where two_body_terms order changed to: two_body_terms transpose (0,2,3,1) 
                        H_fermionic += 0.5*two_body_terms[p,q,r,s] * FermionOperator(((p, 1), (q, 1), (r,0), (s, 0)))
    else:
        # one body terms
        for p in range(one_body_terms.shape[0]):
            for q in range(one_body_terms.shape[0]):

                H_fermionic += one_body_terms[p,q] * FermionOperator(((p, 1), (q, 0)))

                # two body terms
                for r in range(two_body_terms.shape[0]):
                    for s in range(two_body_terms.shape[0]):
                        ######## chemist notation
                        H_fermionic += 0.5*two_body_terms[p,q,r,s] * FermionOperator(((p, 1), (r, 1), (s,0), (q, 0)))
    return H_fermionic

class embeddeding_SCF_driver():

    def __init__(self, geometry,
                 N_active_atoms,
                 projector_method,
                 cheap_global_SCF_method='RKS', 
                 cheap_global_DFT_xc= 'lda, vwn',
                 expensive_global_DFT_xc = 'b3lyp',
                 cheap_WF_method = 'RHF',
                 expensive_WF_method = 'CCSD',
                 E_convergence_tol = 1e-6,
                 basis = 'STO-3G',
                 output_file_name='output.dat',
                 unit= 'angstrom',
                 pyscf_print_level=1,
                 memory=4000,
                 charge=0,
                 spin=0,
                 run_fci=False,
                 run_cisd=False,
                 mu_value=1e6,
                 physists_notation=False):
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
        self.cheap_global_SCF_method = cheap_global_SCF_method
        self.cheap_global_DFT_xc = cheap_global_DFT_xc

        self.expensive_global_DFT_xc = expensive_global_DFT_xc


        self.cheap_WF_method = cheap_WF_method
        self.expensive_WF_method = expensive_WF_method

        self.projector_method = projector_method
        self.mu_value = mu_value


        self.E_convergence_tol = E_convergence_tol

        # file_system and print
        self.output_file_name = output_file_name
        self.pyscf_print_level =  pyscf_print_level

        ## QC info
        self.physists_notation=physists_notation



    def initialize_PySCF_molecule_obj(self):

        full_system_mol = gto.Mole(atom= self.geometry,
                                        basis=self.basis,
                                        charge=self.charge,
                                        spin= self.spin,
                                        )

        full_system_mol.unit = self.unit
        full_system_mol.build()

        return full_system_mol

    def Draw_molecule(self, PySCF_mol_obj, width=400, height=400, jupyter_notebook=True):
        xyz_string = Get_xyz_string(PySCF_mol_obj)
        return Draw_molecule(xyz_string, width=width, height=400, jupyter_notebook=jupyter_notebook)

    def _run_cheap_global_calc(self, PySCF_mol_obj, cheap_global_SCF_method, cheap_global_DFT_xc=None):

        if cheap_global_SCF_method == 'RKS':
            full_system_scf = scf.RKS(PySCF_mol_obj)
            full_system_scf.verbose = self.pyscf_print_level
            full_system_scf.max_memory= self.memory
            full_system_scf.conv_tol = self.E_convergence_tol

            if cheap_global_DFT_xc is None:
                raise ValueError('no functional specified for global DFT calculation')
            full_system_scf.xc = self.cheap_global_DFT_xc
        
        else:
            # TODO add other PySCF methods here
            raise ValueError(f'Unknown SCF method: {cheap_global_SCF_method}')

        full_system_scf.kernel()
        return full_system_scf

    def _localize_orbitals(self, PySCF_scf_obj, localization_method, orbtial_loc_threshold=0.9, localized_orb_sanity_check=True):
         (C_active, 
         C_envrio, 
         C_all_localized, 
         active_MO_inds,
         enviro_MO_inds) = Localize_orbitals(localization_method, 
                                             PySCF_scf_obj, 
                                             self.N_active_atoms, 
                                             THRESHOLD=orbtial_loc_threshold, 
                                             sanity_check=localized_orb_sanity_check)

         return C_active, C_envrio, C_all_localized, active_MO_inds, enviro_MO_inds

    def _run_embedded_scf_calc(self, PySCF_mol_embedded_obj, V_embed):

        if self.cheap_global_SCF_method == 'RKS':
            EMBEDDED_full_system_scf = scf.RKS(PySCF_mol_embedded_obj)
            EMBEDDED_full_system_scf.verbose= self.pyscf_print_level
            EMBEDDED_full_system_scf.max_memory= self.memory
            EMBEDDED_full_system_scf.conv_tol = self.E_convergence_tol
            EMBEDDED_full_system_scf.xc = self.cheap_global_DFT_xc        
        else:
            # TODO add other PySCF methods here
            raise ValueError(f'Unknown SCF method: {self.cheap_global_SCF_method}')


        h_core_standard = EMBEDDED_full_system_scf.get_hcore()
        ## VERY IMPORTANT STEP - overwriting core H with embedding potential!
        EMBEDDED_full_system_scf.get_hcore = lambda *args: V_embed + h_core_standard

        # run SCF embedded calculation (to give optimized embedded e- density)
        E_emb = EMBEDDED_full_system_scf.kernel()

        if EMBEDDED_full_system_scf.conv_check is not True:
            raise ValueError('embedded SCF calculation not converged')


        return EMBEDDED_full_system_scf

    def _run_expensive_DFT_calc_with_embedded_dm(self, PySCF_mol_embedded_obj, dm_embedded_matrix):

        if self.cheap_global_SCF_method == 'RKS':
            full_system_scf_HIGH_LEVEL = scf.RKS(PySCF_mol_embedded_obj)
            full_system_scf_HIGH_LEVEL.verbose= self.pyscf_print_level
            full_system_scf_HIGH_LEVEL.max_memory= self.memory
            full_system_scf_HIGH_LEVEL.conv_tol = self.E_convergence_tol
            full_system_scf_HIGH_LEVEL.xc = self.expensive_global_DFT_xc # <--- better functional!        
        else:
            # TODO add other PySCF methods here
            raise ValueError(f'Unknown SCF method: {self.cheap_global_SCF_method}')


        e_act_emb_HIGH_LVL = full_system_scf_HIGH_LEVEL.energy_elec(dm=dm_embedded_matrix)
        return e_act_emb_HIGH_LVL[0]

    def _get_HF_scf_with_embedded_dm(self, PySCF_mol_embedded_obj, PySCF_scf_DFT_embedded_obj,
                                      V_embed, dm_embedded_matrix):
        
        # PySCF_scf_DFT_embedded_obj.mo_coeff == C_active
        if self.cheap_WF_method == 'RHF':
            EMBEDDED_full_system_scf_HF = scf.RHF(PySCF_mol_embedded_obj)
            EMBEDDED_full_system_scf_HF.verbose= self.pyscf_print_level
            EMBEDDED_full_system_scf_HF.max_memory= self.memory
            EMBEDDED_full_system_scf_HF.conv_tol = self.E_convergence_tol
        else:
            # TODO add other PySCF methods here
            raise ValueError(f'Unknown SCF method: {self.cheap_WF_method}')

        # overwrite h_core to include embedding term!!!!
        h_core_standard = EMBEDDED_full_system_scf_HF.get_hcore()
        EMBEDDED_full_system_scf_HF.get_hcore = lambda *args: V_embed + h_core_standard
        # EMBEDDED_full_system_scf_HF.kernel() # <------ do NOT RUN!

        # instead overwrite with C_active found from embedded DFT calculation!
        EMBEDDED_full_system_scf_HF.mo_coeff = PySCF_scf_DFT_embedded_obj.mo_coeff 
        EMBEDDED_full_system_scf_HF.mo_occ = PySCF_scf_DFT_embedded_obj.mo_occ 
        EMBEDDED_full_system_scf_HF.mo_energy = PySCF_scf_DFT_embedded_obj.mo_energy

        return EMBEDDED_full_system_scf_HF


        e_act_emb_HIGH_LVL = full_system_scf_HIGH_LEVEL.energy_elec(dm=dm_embedded_matrix)
        return e_act_emb_HIGH_LVL[0]


    def _run_expensive_WF_calculation(self, PySCF_scf_HF_embedded_obj, list_frozen_orbitals=[]):

        # PySCF_scf_DFT_embedded_obj.mo_coeff == C_active
        # PySCF_scf_HF_embedded_obj contains EMBEDDED HCore


        if self.expensive_WF_method == 'CCSD':
            embedded_WF_SCF_obj = cc.CCSD(PySCF_scf_HF_embedded_obj)
            if list_frozen_orbitals:
                embedded_WF_SCF_obj.frozen = list_frozen_orbitals
            # run SCF calculation
            e_correlation, t1, t2 = embedded_WF_SCF_obj.kernel()
            return e_correlation, t1, t2, embedded_WF_SCF_obj
        else:
            # TODO add other PySCF methods here
            raise ValueError(f'Unknown SCF method: {self.expensive_WF_method}')


    def run_experiment(self, localization_method,
                            orbtial_loc_threshold=0.9, localized_orb_sanity_check=True, dm_localized_sanity_check=True,
                            check_Hcore_is_correct_Vembed=False, check_Vembed=True, check_HFock_embedded=True, check_expensive_WF_HF_calc=True):

        ###### 1. Define full system mol obj
        full_system_mol = self.initialize_PySCF_molecule_obj()

        ###### 2. Run CHEAP full system calculation #######
        full_system_scf_cheap = self._run_cheap_global_calc(full_system_mol,
                                                            self.cheap_global_SCF_method,
                                                            cheap_global_DFT_xc= self.cheap_global_DFT_xc)

        Nuclear_energy =  full_system_scf_cheap.energy_nuc()
        ###### 3. Localize orbitals #######
        (C_active, 
         C_envrio, 
         C_all_localized, 
         active_MO_inds,
         enviro_MO_inds)= self._localize_orbitals(full_system_scf_cheap,
                                                 localization_method, 
                                                 orbtial_loc_threshold=orbtial_loc_threshold, 
                                                 localized_orb_sanity_check=localized_orb_sanity_check)

        ####### 4. Get active and enviro  density matrices #######
        dm_active, dm_enviro = Get_active_and_envrio_dm(
                                                full_system_scf_cheap,
                                                C_active, 
                                                C_envrio, 
                                                C_all_localized, 
                                                sanity_check=dm_localized_sanity_check)

        ####### 5. Get active, enviro  and cross terms #######
        E_act, J_act, K_act, e_xc_act, v_xc_act = Get_energy_and_matrices_from_dm( full_system_scf_cheap, 
                                                                             dm_active, # <- ACTIVE
                                                                             check_E_with_pyscf=True)

        # use enviro density
        E_env, J_env, K_env, e_xc_env, v_xc_env = Get_energy_and_matrices_from_dm(
                                                                            full_system_scf_cheap,
                                                                             dm_enviro, # <- ENVIRO
                                                                             check_E_with_pyscf=True)

        # cross terms!
        two_e_cross = Get_cross_terms(full_system_scf_cheap, 
                                                 dm_active, 
                                                 dm_enviro, 
                                                 J_env, 
                                                 J_act, 
                                                 e_xc_act,
                                                 e_xc_env)

        ####### 6. Get V_embed #######
        V_embed  = Get_embedded_potential_operator(self.projector_method, 
                                        full_system_scf_cheap, 
                                        dm_active, 
                                        dm_enviro, 
                                        check_Hcore_is_correct=check_Hcore_is_correct_Vembed, 
                                        mu_shift_val=self.mu_value,
                                        check_Vemb=check_Vembed)


        ####### 7. Define embedded molecular system _localize_orbitals
        full_system_mol_EMBEDDED = self.initialize_PySCF_molecule_obj()
        # OVERWRITE number of electrons! (VERY IMPORTANT STEP)
        full_system_mol_EMBEDDED.nelectron = 2*len(active_MO_inds) 


        ####### 8. Run SCF with V_embed to get optimized embedded e- density #######
        embedded_DFT_scf = self._run_embedded_scf_calc(full_system_mol_EMBEDDED, V_embed)


        ####### 9. Get embedded density matrix
        EMBEDDED_occ_orbs = embedded_DFT_scf.mo_coeff[:, embedded_DFT_scf.mo_occ>0]

        # optimized embedded denisty matrix
        density_emb = 2 * EMBEDDED_occ_orbs @ EMBEDDED_occ_orbs.conj().T

        ## check number of electrons makes sense:
        electron_check = np.isclose(np.trace(density_emb@full_system_scf_cheap.get_ovlp()), 2*len(active_MO_inds))
        print(f'\nnumber of e- in gamma_embedded is correct: {electron_check}\n')


        ####### 10. Calculate electronic energy with new density matrix and NORMAL global molecule
        e_act_emb = full_system_scf_cheap.energy_elec(dm=density_emb,
                                                vhf= full_system_scf_cheap.get_veff(dm=density_emb),
                                               h1e = full_system_scf_cheap.get_hcore())[0]
        
        ####### 11. Get dm embedding correction
        dm_correction = np.einsum('ij, ij', V_embed, density_emb-dm_active)
        print(f'\nRKS correction: {dm_correction}\n')

        ####### 12. Putting everything together
        e_mf_emb = e_act_emb + E_env + two_e_cross + Nuclear_energy + dm_correction # <-- energy from embedded DFT calc
        # expected as same functional used!
        print(f'\nglobal DFT calculation == seperated calculation: {np.isclose(e_mf_emb, full_system_scf_cheap.e_tot)}')
        print(f'Cheap DFT energy Calculation: {e_mf_emb}\n')


        ####### 13. Prepare higher level DFT Calculation
        expensive_DFT_mol = self.initialize_PySCF_molecule_obj()
        E_DFT_expensive = self._run_expensive_DFT_calc_with_embedded_dm(expensive_DFT_mol, density_emb)
        # calculate embedding correction term
        E_DFT_emb_high_lvl = E_DFT_expensive + E_env + two_e_cross + Nuclear_energy + dm_correction # <-- energy from embedded DFT calc
        print(f'\nExpensive DFT energy Calculation: {E_DFT_emb_high_lvl}\n')



        ####### 14. Prepare WF calculation - Hartree Fock first
        HFock_mol_embedded = self.initialize_PySCF_molecule_obj()
        # OVERWRITE number of electrons! (VERY IMPORTANT STEP)
        HFock_mol_embedded.nelectron =  2*len(active_MO_inds) 

        HFock_scf_embedded = self._get_HF_scf_with_embedded_dm( HFock_mol_embedded, 
                                                                embedded_DFT_scf,
                                                                V_embed,
                                                                density_emb
                                                                )
        E_elec_HF_embedded = HFock_scf_embedded.energy_tot(dm=density_emb)

        if check_HFock_embedded is True:
            E_elec_PySCF = HFock_scf_embedded.energy_tot()
            if not np.isclose(E_elec_PySCF, E_elec_HF_embedded):
                raise ValueError('Embedded HF calculation is going wrong somewhere... electronic energies not matching')


        ####### 15. Run CCSD calcualtion
        frozen_orbitals = [i for i in range(HFock_scf_embedded.mol.nao - len(enviro_MO_inds),
                                           HFock_scf_embedded.mol.nao)] # note NOT spin orbs here!

        e_correlation_WF, t1, t2, embedded_WF_SCF_obj = self._run_expensive_WF_calculation(HFock_scf_embedded, list_frozen_orbitals=frozen_orbitals)

        if check_expensive_WF_HF_calc is True:
            CC_flag_check = np.isclose(E_elec_HF_embedded, embedded_WF_SCF_obj.e_hf)
            print(f'\nWF hartree fock energy matches HF embedded calc: {CC_flag_check}!')
            if not CC_flag_check:
                raise ValueError('WF HF calc does NOT match base HF calculation')

        

        ####### 16. Get WF correction
        WF_correction = np.einsum('ij, ij', V_embed, dm_active)
        print(f'\n\nWF correction: {WF_correction}')

        ### 17. Putting everything together
        E_WF = E_elec_HF_embedded +e_correlation_WF  + E_env + two_e_cross - WF_correction
        print(f'Expensive WF (classic) energy Calculation: {E_WF}\n')
        
        ##################### QC part ########################

        N_enviroment_MOs = len(enviro_MO_inds) 
        one_body_integrals, two_body_integrals = Get_embedded_one_and_two_body_integrals_MO_basis(HFock_scf_embedded,
                                                                                            N_enviroment_MOs,
                                                                                            physists_notation=self.physists_notation)

        one_body_terms, two_body_terms = Get_SpinOrbs_from_Spatial(one_body_integrals,
                                                                   two_body_integrals,
                                                                   physists_notation=self.physists_notation,
                                                                   EQ_Tolerance=1e-8)


        H_fermionic = Get_fermionic_H(one_body_terms, 
                                             two_body_terms, 
                                             Nuclear_energy,
                                             core_constant=0, 
                                             physists_notation=self.physists_notation)

        H_sparse = get_sparse_operator(H_fermionic)

        if H_sparse.shape[0]>(2**9):
            raise ValueError('diagonlizing LARGE matrix')

        eigvals_EMBED, eigvecs_EMBED = sp.sparse.linalg.eigsh(H_sparse, which='SA', k=1)

        E_VQE = eigvals_EMBED[0]  + E_env + two_e_cross - WF_correction
        
        N_electrons_expected = 2*len(active_MO_inds)
        N_electrons_Q_state = np.binary_repr(np.where(np.abs(eigvecs_EMBED)>1e-2)[0][0]).count('1')  

        print(f'expect {N_electrons_expected} electrons')
        print(f'quantum state has {N_electrons_Q_state} electrons \n')
        print(f'number of electrons correct: {N_electrons_expected == N_electrons_Q_state}')

        where=np.where(np.around(eigvecs_EMBED, 4)>0)[0]
        print('superposition states:', where)

        return e_act_emb, E_DFT_emb_high_lvl, E_WF, E_VQE

class embeddeding_SCF_experiment():

    def __init__(self, geometry,
                 N_active_atoms, 
                 low_level_scf_method='RKS', 
                 low_level_xc_functional= 'lda, vwn',
                 high_level_xc_functional = 'b3lyp',
                 E_convergence_tol = 1e-6,
                 basis = 'STO-3G',
                 output_file_name='output.dat',
                 unit= 'angstrom',
                 pyscf_print_level=1,
                 memory=4000,
                 charge=0,
                 spin=0,
                 run_fci=False,
                 run_cisd=False):
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


        self._init_pyscf_system()

        self.E_FCI = self._run_fci() if run_fci else None
        self.E_CISD = self._run_cisd() if run_cisd else None

    def TODO():
        pass











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

        # self.my_fci = fci.FCI(HF_scf).run()
        # # print('E(UHF-FCI) = %.12f' % self.my_fci.e_tot)
        # self.E_FCI = self.my_fci.e_tot

        self.my_fci = ci.CISD(HF_scf).run()
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
                delta_s = singular_values[:-1] - singular_values[1:] # σ_i - σ_(i+1)
                print(delta_s)
                self.n_act_mos = np.argmax(delta_s)+1 # add one due to python indexing

                # delta_s = [(i, (singular_values[i] - singular_values[i+1] )) for i in range(len(singular_values) - 1)] # contains (index, delta_s)
                # print(delta_s)

                # delta_s_TEST = [-(singular_values[i+1] - singular_values[i]) for i in range(len(singular_values) - 1)]
                # n_act_mos_TEST = np.argpartition(delta_s_TEST, -1)[-1] + 1
                # print(n_act_mos_TEST)
                # print(max(delta_s, key=lambda x: x[0])[0] + 1)

                # self.n_act_mos = max(delta_s, key=lambda x: x[0])[0] + 1 # finds index where largest step change is! (adds 1 due to python indexing going from 0)
                self.n_env_mos = len(singular_values) - self.n_act_mos

                # define active and environment orbitals from localization
                self.act_orbitals = occupied_orbs @ rotation_matrix.T[:, :self.n_act_mos]
                self.env_orbitals = occupied_orbs @ rotation_matrix.T[:, self.n_act_mos:]

                self.C_matrix_all_localized_orbitals = occupied_orbs @ rotation_matrix.T

                self.active_MO_inds  = np.arange(self.n_act_mos)
                self.enviro_MO_inds = np.arange(self.n_act_mos, self.n_act_mos+self.n_env_mos)

            else:
                THRESHOLD = 0.9#0.25

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
                    #
                    mo_occ = self.full_system_scf.mo_coeff[:,self.full_system_scf.mo_occ>0]
                    iaos = lo.iao.iao(self.full_system_scf.mol, mo_occ)
                    C_loc_occ = lo.ibo.ibo(self.full_system_scf.mol, mo_occ, locmethod='IBO', iaos=iaos)#.kernel()

                    # C_loc = lo.ibo.ibo(self.full_system_mol, sum(self.full_system_scf.mo_occ)//2, locmethod='IBO', verbose=1)
                    # C_loc_occ = C_loc[:,self.full_system_scf.mo_occ>0]
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

                # active_MO_ind_list=[]
                # for mo_ind in range(C_loc_occ.shape[1]):
                #     MO_orb = C_loc_occ[:, mo_ind] # MO coefficients (c_j)
                #     MO_active_AO_overlap=0
                #     for active_AO_index in ao_active_inds:
                #         bra_aoI_ket_All_AOs = self.S_ovlp[active_AO_index, :] # [ < ϕ_AO_SELECTED | ϕ_AO_0> , < ϕ_AO_SELECTED | ϕ_AO_1>, ..., < ϕ_AO_SELECTED | ϕ_AO_M> ]
                #         AO_i_overlap_MO = np.dot(bra_aoI_ket_All_AOs.T, MO_orb) # < ϕ_AO_i |ψ_MO> = Σ_j  (c_j < ϕ_AO_i | ϕ_AO_j >) # aka summation of ci and overlap (MO_orb are the c_j terms!)
                #         MO_active_AO_overlap+=AO_i_overlap_MO
                #     print(MO_active_AO_overlap)
                #     if MO_active_AO_overlap>THRESHOLD:
                #         active_MO_ind_list.append(mo_ind)
                # self.active_MO_inds = np.array(active_MO_ind_list)

                MO_AO_overlap = self.S_ovlp@C_loc_occ  #  < ϕ_AO_i | ψ_MO_j >
                MO_active_AO_overlap = np.einsum('ij->j', MO_AO_overlap[ao_active_inds]) # sum over rows of active AOs of MOs!
                self.active_MO_inds = np.where(MO_active_AO_overlap>THRESHOLD)[0]


                # active_MO_ind_list=[]
                # for mo_ind in range(C_loc_occ.shape[1]):
                #     MO_orb = C_loc_occ[:, mo_ind] # MO coefficients (c_j)
                #     MO_orb = MO_orb/np.linalg.norm(MO_orb) # <--- NORMALIZE WAVEFUNCTION!
                #     MO_active_AO_overlap=0
                #     for active_AO_index in ao_active_inds:
                #         bra_aoI_ket_All_AOs = self.S_ovlp[active_AO_index, :] # [ < ϕ_AO_SELECTED | ϕ_AO_0> , < ϕ_AO_SELECTED | ϕ_AO_1>, ..., < ϕ_AO_SELECTED | ϕ_AO_M> ]
                #         AO_i_overlap_MO = np.dot(bra_aoI_ket_All_AOs.T, MO_orb) # < ϕ_AO_i |ψ_MO> = Σ_j  (c_j < ϕ_AO_i | ϕ_AO_j >) # aka summation of ci and overlap (MO_orb are the c_j terms!)
                #         # MO_active_AO_overlap+=abs(AO_i_overlap_MO)
                #         MO_active_AO_overlap+=AO_i_overlap_MO # <--- NOT absolute val !
                #     print(MO_active_AO_overlap)
                #     if MO_active_AO_overlap>THRESHOLD:
                #         active_MO_ind_list.append(mo_ind)

                # self.active_MO_inds = np.array(active_MO_ind_list)

                # ## based on occupation of AO orbital
                # dm_loc = 2 * C_loc_occ @ C_loc_occ.conj().T
                # PS = dm_loc @ self.S_ovlp
                # active_diag = np.diag(PS)[ao_active_inds]
                # self.active_MO_inds = np.where(active_diag>THRESHOLD)[0]
                

                # active_MO_ind_list=[]
                # S_half = sp.linalg.fractional_matrix_power(self.S_ovlp, 0.5)
                # C_loc_occ_ORTHO = S_half@C_loc_occ # Get orbs in an orthogonal basis!
                # # diagonal_elements = np.diag(C_loc_occ_ORTHO)[ao_active_inds]
                # # print(diagonal_elements)
                # # self.active_MO_inds = np.where(diagonal_elements>THRESHOLD)[0]
                # for mo_ind in range(C_loc_occ_ORTHO.shape[1]):
                #     MO_orb = C_loc_occ_ORTHO[:, mo_ind] # MO coefficients (c_j) in orthogonal basis!
                #     MO_active_AO_overlap=0
                #     for active_AO_index in ao_active_inds:
                #         bra_aoI_ket_All_AOs = self.S_ovlp[active_AO_index, :] # [ < ϕ_AO_SELECTED | ϕ_AO_0> , < ϕ_AO_SELECTED | ϕ_AO_1>, ..., < ϕ_AO_SELECTED | ϕ_AO_M> ]
                #         AO_i_overlap_MO = np.dot(bra_aoI_ket_All_AOs.T, MO_orb) # < ϕ_AO_i |ψ_MO> = Σ_j  (c_j < ϕ_AO_i | ϕ_AO_j >) # aka summation of ci and overlap
                #         MO_active_AO_overlap+=AO_i_overlap_MO
                #     print(MO_active_AO_overlap)
                #     if MO_active_AO_overlap>THRESHOLD:
                #         active_MO_ind_list.append(mo_ind)
                # self.active_MO_inds = np.array(active_MO_ind_list)


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
                # SavNonee localized orbitals as molden file
                with open('LOCALIZED_orbs.molden', 'w') as outfile:
                    tools.molden.header(self.full_system_mol,
                                        outfile)
                    tools.molden.orbital_coeff(
                                        self.full_system_mol,
                                         outfile, 
                                         self.act_orbitals, # <- active orbitals!
                                         ene=self.full_system_scf.mo_energy[self.active_MO_inds],
                                         occ=self.full_system_scf.mo_occ[self.active_MO_inds])

            return 

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





class SCF_base():


    def __init__(self, PySCF_scf_obj):
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

        self.PySCF_scf_obj = PySCF_scf_obj
        self.N_active_atoms = N_active_atoms

        self._init_pyscf_system()

    def _init_pyscf_system(self):

        print(self.PySCF_scf_obj.geometry)
        print(self.PySCF_scf_obj.basis)
        print(self.PySCF_scf_obj.unit)
        print(self.PySCF_scf_obj.memory)
        print(self.PySCF_scf_obj.charge.spin)

        print(self.PySCF_scf_obj.xc)

    def check_hcore_is_embedded(self):

        H_core_standard = scf.hf.get_hcore(self.PySCF_scf_obj.mol) # calculate Hcore using PySCF inbuilt.
        H_core_in_SCF_calculation = self.PySCF_scf_obj.get_hcore() #

        if np.allclose(H_core_standard, H_core_in_SCF_calculation):
            print('H core is standard H_core')
        else:
            print('H core is NOT standard H_core')

    def run_PySCF_kernel(self):
        self.S_ovlp = self.PySCF_scf_obj.get_ovlp()
        self.full_system_scf.kernel()

        two_e_term_total = self.PySCF_scf_obj.get_veff()
        self.e_xc_total = two_e_term_total.exc
        self.v_xc_total = two_e_term_total - self.PySCF_scf_obj.get_j() 


    def _get_xyz_format(self):

        if isinstance(self.geometry, str):
            # if already xzy format
            return self.geometry
        
        n_atoms = len(self.geometry)
        xyz_str=f'{n_atoms}'
        xyz_str+='\n \n'
        for atom, xyz in self.geometry:
            xyz_str+= f'{atom}\t{xyz[0]}\t{xyz[1]}\t{xyz[2]}\n'
        
        self.geometry = xyz_str

        return None

    def draw_molecule_3D(self, width=400, height=400, jupyter_notebook=False):

        if jupyter_notebook is True:
            import rdkit
            from rdkit.Chem import Draw
            from rdkit.Chem.Draw import IPythonConsole
            rdkit.Chem.Draw.IPythonConsole.ipython_3d = True  # enable py3Dmol inline visualization

        xyz_geom = self.geometry
        view = py3Dmol.view(width=width, height=height)
        view.addModel(xyz_geom, "xyz")
        view.setStyle({'stick':{}})
        view.zoomTo()
        return(view.show())


    def Run_fci(self, pyscf_print_level=0):

        HF_scf = scf.RHF(self.PySCF_scf_obj.mol)
        HF_scf.verbose= pyscf_print_level
        HF_scf.max_memory= self.PySCF_scf_obj.memory
        HF_scf.conv_tol = self.PySCF_scf_obj.conv_tol
        HF_scf.kernel()

        self.my_fci = fci.FCI(HF_scf).run()
        # print('E(UHF-FCI) = %.12f' % self.my_fci.e_tot)
        self.E_FCI = self.my_fci.e_tot

        # self.my_fci = ci.CISD(HF_scf).run()
        # self.E_FCI = self.my_fci.e_tot

        # myci = ci.CISD(HF_scf).run() # this is UCISD
        # print('UCISD total energy = ', myci.e_tot)
        return self.my_fci.e_tot

    def Run_CISD(self, pyscf_print_level=0):

        HF_scf = scf.RHF(self.PySCF_scf_obj.mol)
        HF_scf.verbose= pyscf_print_level
        HF_scf.max_memory= self.PySCF_scf_obj.memory
        HF_scf.conv_tol = self.PySCF_scf_obj.conv_tol
        HF_scf.kernel()

        self.my_CISD = ci.CISD(HF_scf).run()
        self.E_CISD = self.my_CISD.e_tot

        # myci = ci.CISD(HF_scf).run() # this is UCISD
        print('CISD total energy = ', self.E_CIS)
        return None

    def Draw_cube_orbital(self, cube_file, width=400, height=400):

        view = py3Dmol.view(width=width, height=height)
        view.addModel(self.geometry, "xyz")
        view.setStyle({'stick':{}})
        
        with open(cube_file, 'r') as f:
            view.addVolumetricData(f.read(), "cube", {'isoval': -0.02, 'color': "red", 'opacity': 0.75})
        with open(cube_file, 'r') as f2:
            view.addVolumetricData(f2.read(), "cube", {'isoval': 0.02, 'color': "blue", 'opacity': 0.75})
        view.zoomTo()
        return view.show()

    def plot_orbital(self, C_matrix, index, width=400, height=400):
        

        xyz_geom = self.geometry
        if not isinstance(index, int):
            raise ValueError(f'index: {index} required for slice is not an integar')
        if  C_matrix.shape[1]<=index:
            raise ValueError('index is outside of C_matrix shape')


        File_name = f'temp_MO_orbital_index{index}.cube'
        cubegen.orbital(self.full_system_mol, File_name, C_matrix[:, index])
        self.Draw_cube_orbital(File_name, width=width, height=height)
        os.remove(File_name) # delete file once orbital is drawn

        return None

    def plot_all_orbitals(self, C_matrix, width=400, height=400):
        
        for MO_ind in range(C_matrix.shape[1]):
            self.plot_orbital(C_matrix, MO_ind, width=width, height=height)
        
        return None

class global_mol(SCF_base):

    def __init__(self, PySCF_scf_obj,
                        localization_method,
                        projection_method,
                        N_active_atoms,
                         mu_level_shift=1e6):
        """
        Initialize the full molecular system.

            Args:
                PySCF_scf_obj (pyscf.SCF obj): PySCF SCF object. Note active atoms must be defined in geometry first 

        """
        super().__init__(PySCF_scf_obj)

        self.N_active_atoms = N_active_atoms
        self.localization_method = localization_method #'spade' 'pipekmezey' 'boys 'ibo'
        self.projection_method = projection_method #'spade' 'pipekmezey' 'boys 'ibo'
        self.mu_level_shift = mu_level_shift

        ### 
        self.v_emb = None
        self.dm_active = None
        self.density_emb_active = None


        def Localize_orbitals(self, localization_method, PySCF_scf_obj, sanity_check=False, THRESHOLD=None):

            if self.PySCF_scf_obj.mo_coeff is None:
                # run SCF calculation
                self.run_PySCF_kernel()

            AO_slice_matrix = self.PySCF_scf_obj.mol.aoslice_by_atom()

            # run localization scheme
            if localization_method.lower() == 'spade':

                # Take occupied orbitals of global system calc
                occupied_orbs = self.PySCF_scf_obj.mo_coeff[:,self.PySCF_scf_obj.mo_occ>0]

                # Get active AO indices
                N_active_AO = AO_slice_matrix[self.N_active_atoms-1][3]  # find max AO index for active atoms (neg 1 as python indexs from 0)

                
                S_half = sp.linalg.fractional_matrix_power(self.S_ovlp, 0.5)
                orthogonal_orbitals = (S_half@occupied_orbs)[:N_active_AO, :] # Get rows (the active AO) of orthogonal orbs 

                # Comute singular vals
                u, singular_values, rotation_matrix = np.linalg.svd(orthogonal_orbitals, full_matrices=True)

                # find where largest step change 
                delta_s = singular_values[:-1] - singular_values[1:] # σ_i - σ_(i+1)
                # print(delta_s)
                self.n_act_mos = np.argmax(delta_s)+1 # add one due to python indexing
                self.n_env_mos = len(singular_values) - self.n_act_mos

                # define active and environment orbitals from localization
                self.act_orbitals = occupied_orbs @ rotation_matrix.T[:, :self.n_act_mos]
                self.env_orbitals = occupied_orbs @ rotation_matrix.T[:, self.n_act_mos:]

                self.C_matrix_all_localized_orbitals = occupied_orbs @ rotation_matrix.T

                self.active_MO_inds  = np.arange(self.n_act_mos)
                self.enviro_MO_inds = np.arange(self.n_act_mos, self.n_act_mos+self.n_env_mos)

            else:
                if THRESHOLD is None:
                    raise ValueError('THRESHOLD for active MO needs to be set')

                # Take C matrix from SCF calc
                opt_C = self.PySCF_scf_obj.mo_coeff

                # run localization scheme
                if localization_method.lower() == 'pipekmezey':
                    ### PipekMezey
                    PM = lo.PipekMezey(self.PySCF_scf_obj.mol, opt_C)
                    PM.pop_method = 'mulliken' # 'meta-lowdin', 'iao', 'becke'
                    C_loc = PM.kernel() # includes virtual orbs too!
                    C_loc_occ = C_loc[:,self.PySCF_scf_obj.mo_occ>0]

                elif localization_method.lower() == 'boys':
                    ### Boys
                    boys_SCF = lo.boys.Boys(self.PySCF_scf_obj.mol, opt_C)
                    C_loc  = boys_SCF.kernel()
                    C_loc_occ = C_loc[:,self.PySCF_scf_obj.mo_occ>0]

                elif localization_method.lower() == 'ibo':
                    ### intrinsic bonding orbs
                    #
                    mo_occ = self.PySCF_scf_obj.mo_coeff[:,self.PySCF_scf_obj.mo_occ>0]
                    iaos = lo.iao.iao(self.PySCF_scf_obj.mol, mo_occ)
                    C_loc_occ = lo.ibo.ibo(self.PySCF_scf_obj.mol, mo_occ, locmethod='IBO', iaos=iaos)#.kernel()
                else:
                    raise ValueError(f'unknown localization method {localization_method}')
                
                                
                # find indices of AO of active atoms
                ao_active_inds = np.arange(AO_slice_matrix[0,2], AO_slice_matrix[self.N_active_atoms-1,3])

                MO_AO_overlap = self.S_ovlp@C_loc_occ  #  < ϕ_AO_i | ψ_MO_j >
                MO_active_AO_overlap = np.einsum('ij->j', MO_AO_overlap[ao_active_inds]) # sum over rows of active AOs of MOs!
                
                self.active_MO_inds = np.where(MO_active_AO_overlap>THRESHOLD)[0]
                self.enviro_MO_inds = np.array([i for i in range(C_loc_occ.shape[1]) if i not in self.active_MO_inds]) # get all non active MOs

                # define active MO orbs and environment
                self.act_orbitals = C_loc_occ[:, self.active_MO_inds] # take MO (columns of C_matrix) that have high dependence from active AOs
                self.env_orbitals = C_loc_occ[:, self.enviro_MO_inds]
                
                self.C_matrix_all_localized_orbitals = C_loc_occ

                self.n_act_mos = len(self.active_MO_inds)
                self.n_env_mos = len(self.enviro_MO_inds)



            self.dm_active =  2 * self.act_orbitals @ self.act_orbitals.T
            self.dm_enviro =  2 * self.env_orbitals @ self.env_orbitals.T


            if sanity_check is True:

                print(f'number of active MOs: {n_act_mos}')
                print(f'number of enviro MOs: {n_env_mos} \n')


                bool_flag_electron_number = np.isclose(np.trace(dm_active@S_mat) + np.trace(dm_enviro@S_mat), full_system_mol.nelectron)
                print(f'N_active_elec + N_environment_elec = N_total is: {bool_flag_electron_number}')
                ## check number of electrons is still the same after orbitals have been localized (change of basis)
                if not np.isclose((np.trace(self.dm_active@self.S_ovlp) + np.trace(self.dm_enviro@self.S_ovlp)), self.PySCF_scf_obj.mol.nelectron):
                    raise ValueError('number of electrons in localized orbitals is incorrect')
                

                # checking denisty matrix parition makes sense:

                # gamma_localized_full_system = gamma_act + gamma_env
                dm_localised_full_system = 2* self.C_matrix_all_localized_orbitals@ self.C_matrix_all_localized_orbitals.conj().T
                if not np.allclose(dm_localised_full_system, self.dm_active + self.dm_enviro):
                    raise ValueError('gamma_full != gamma_active + gamma_enviro')
            

            return self.C_matrix_all_localized_orbital, self.active_MO_inds, self.enviro_MO_inds
