from copy import copy
from typing import Dict
from .spade import fill_defaults
import numpy as np
from scipy import linalg
from pyscf import gto, scf, cc, ao2mo, fci



def get_exact_energy(mol: gto.Mole, keywords: Dict):
    hf = mol.RHF().run()

    ref_fci = fci.FCI(hf)
    ref_fci.conv_tol = keywords["e_convergence"]
    fci_result = ref_fci.kernel()

    return fci_result[0]

def spade_localisation(scf, keywords):
    n_occupied_orbitals = np.count_nonzero(scf.mo_occ == 2)
    occupied_orbitals = scf.mo_coeff[:, :n_occupied_orbitals]

    n_act_aos = scf.mol.aoslice_by_atom()[keywords['n_active_atoms']-1][-1]
    ao_overlap = scf.get_ovlp()

    # Orbital rotation and partition into subsystems A and B
    #rotation_matrix, sigma = embed.orbital_rotation(occupied_orbitals,
    #    n_act_aos, ao_overlap)

    rotated_orbitals = linalg.fractional_matrix_power(ao_overlap, 0.5) @ occupied_orbitals
    _, sigma, right_vectors = linalg.svd(rotated_orbitals[:n_act_aos, :])

    #n_act_mos, n_env_mos = embed.orbital_partition(sigma)
    value_diffs = sigma[1:]-sigma[:-1]
    n_act_mos = np.argmin(value_diffs) + 1
    n_env_mos = len(sigma) - n_act_mos

    # Defining active and environment orbitals and density
    act_orbitals = occupied_orbitals @ right_vectors.T[:, :n_act_mos]
    env_orbitals = occupied_orbitals @ right_vectors.T[:, n_act_mos:]
    act_density = 2.0 * act_orbitals @ act_orbitals.T
    env_density = 2.0 * env_orbitals @ env_orbitals.T
    return n_act_mos, n_env_mos, act_density, env_density

def closed_shell_subsystem(scf, density):
    #It seems that PySCF lumps J and K in the J array 
    j = scf.get_j(dm = density)
    k = np.zeros(np.shape(j))
    two_e_term =  scf.get_veff(scf.mol, density)
    e_xc = two_e_term.exc
    v_xc = two_e_term - j

    # Energy
    e = np.einsum("ij,ij", density, scf.get_hcore() + j/2) + e_xc
    return e, e_xc, j, k, v_xc

def run_sim(mol: gto.Mole, keywords: Dict):

    ks = scf.RKS(mol)
    ks.conv_tol = keywords["e_convergence"]
    ks.xc = keywords["low_level"]
    e_initial = ks.kernel()

    # Store the initial value of h core as this is needed later,
    # but is overwritten

    initial_h_core = ks.get_hcore()

    mol_copy = copy(mol)

    expected_energy = get_exact_energy(mol_copy)

    n_act_mos, n_env_mos, act_density, env_density = spade_localisation(ks)

    # Get cross terms from the initial density
    e_act, e_xc_act, j_act, k_act, v_xc_act = (
    closed_shell_subsystem(scf, act_density))
    e_env, e_xc_env, j_env, k_env, v_xc_env = (
    closed_shell_subsystem(scf, env_density))

    # Computing cross subsystem terms
    # Note that the matrix dot product is equivalent to the trace.
    j_cross = 0.5 * (np.einsum("ij,ij",act_density, j_env)
            + np.einsum("ij,ij", env_density, j_act))

    k_cross = 0.0

    xc_cross = scf.get_veff().exc - e_xc_act - e_xc_env
    two_e_cross = j_cross + k_cross + xc_cross

    # Define the mu-projector
    print(f"{keywords['level_shift']=}")
    projector = keywords['level_shift'] * (ks.get_ovlp() @ env_density
        @ ks.get_ovlp())

    v_xc_total = ks.get_veff() - ks.get_j()

    # Defining the embedded core Hamiltonian
    v_emb = (j_env + v_xc_total - v_xc_act + projector)

    # Run RHF with Vemb to do embedding 
    embedded_scf = scf.RHF(mol)
    embedded_scf.conv_tol = keywords["e_convergence"]
    embedded_scf.mol.nelectron = 2*n_act_mos

    h_core = embedded_scf.get_hcore()

    embedded_scf.get_hcore = lambda *args, **kwargs: h_core + v_emb

    embedded_scf.kernel()

    embedded_occ_orbs = embedded_scf.mo_coeff[:, embedded_scf.mo_occ>0]
    embedded_density = 2*embedded_occ_orbs @ embedded_occ_orbs.T

    # Calculate energy correction
    # - There are two versions used for different embeddings
    wf_correction = np.einsum("ij,ij", v_emb, embedded_density - act_density)
    dft_correction = np.einsum("ij,ij", act_density, v_emb)

    print(f"{wf_correction=}, {dft_correction=}")

    # Calculate the energy of embedded A
    embedded_scf.get_hcore = lambda *args, **kwargs: h_core

    # Can use either of these methods 
    # This needs to change if we're not using PySCFEmbed
    # The j and k matrices are defined differently in PySCF and Psi4
    e_act_emb_explicit = np.einsum("ij,ij", embedded_density,  initial_h_core + 0.5 * embedded_scf.get_j() - 0.25 * embedded_scf.get_k())
    e_act_emb = embedded_scf.energy_elec(dm=embedded_density, vhf=embedded_scf.get_veff())[0]
    print(f"E_HF = {e_act_emb}")
    print(f"Difference between HF methods: {e_act_emb - e_act_emb_explicit}")

    # Run CCSD as WF method
    ccsd = cc.CCSD(embedded_scf)
    ccsd.conv_tol = keywords["e_convergence"]

    # Set which orbitals are to be frozen
    shift = mol.nao - n_env_mos
    fos = [i for i in range(shift, mol.nao)]
    ccsd.frozen = fos

    ccsd.run()
    correlation = ccsd.e_corr
    e_act_emb += correlation


    # Add all the parts up
    e_nuc = mol.energy_nuc()

    e_mf_emb = e_act_emb + e_env + two_e_cross + wf_correction + e_nuc
    print("Component contributions")
    print(f"{e_act_emb=}, {e_env=}, {two_e_cross=}, {wf_correction=}, {e_nuc=}\n")

    # Print out the final value.
    print(f"Full system low-level Energy:\t{e_initial}\n")
    print(f"FCI Energy:\t\t\t{expected_energy}")
    print(f"Embedding Energy:\t\t{e_mf_emb}")
    print(f"Error:\t\t\t\t{(expected_energy-e_mf_emb)*100/expected_energy:.2f}%")

    return e_initial, expected_energy, e_mf_emb

def main():
    "Run for some molecule and a range of values"

    water_geometry = """
    O          0.00000        0.00000        0.1653507
    H          0.00000        0.7493682     -0.4424329
    H          0.00000       -0.7493682     -0.4424329
        """

    options = {}
    options['basis'] = 'STO-6G' # basis set 
    options['low_level'] = 'b3lyp' # level of theory of the environment 
    options['high_level'] = 'mp2' # level of theory of the embedded system
    options['n_active_atoms'] = 1 # number of active atoms (first n atoms in the geometry string)
    options['low_level_reference'] = 'rhf'
    options['high_level_reference'] = 'rhf'
    options['package'] = 'pyscf'

    keywords = fill_defaults(options)

    # Assuming we want to look at water later.
    options['geometry'] = water_geometry
        
    distances = np.linspace(0.1, 1.5, 20)
    values = {"FCI":[], "Embedding":[], "DFT":[]}
    for distance in distances:
        options['geometry'] = f"""
        Li 0.0 0.0 0.0
        H  0.0 0.0 {distance}
        """

        mol = gto.Mole(atom=keywords['geometry'], basis=keywords['basis'], charge=0).build()

        e_initial, expected_energy, e_mf_emb = run_sim(mol, keywords)

        values["DFT"].append(e_initial)
        values["FCI"].append(expected_energy)
        values["Embedding"].append(e_mf_emb)
    
    # Plot the results
    plt.plot(distances, values["DFT"], label="DFT")
    plt.plot(distances, values["FCI"], label="FCI")
    plt.plot(distances, values["Embedding"], label="Embedding")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()