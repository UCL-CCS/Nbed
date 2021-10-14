"""Main embedding functionality."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import jordan_wigner
from pyscf import ao2mo, cc, gto, scf
from pyscf.lib import StreamObject

from vqe_in_dft.localisation import boys, ibo, mullikan, spade
from vqe_in_dft.utils import parse, setup_logs

import pyscf

from pyscf.dft.rks import get_veff as RKS_get_veff    
from pyscf import lib
from pyscf.dft import numint

logger = logging.getLogger(__name__)
setup_logs()


def Get_new_Hcore(H_core: np.array, Unitary_rot: np.array)-> np.array:
    """
    Function to get H_core in new basis

    Args:
        H_core (np.array): standard core Hamiltonian
        Unitary_rot (np.array): Operator to change basis  (in this code base this should be: cannonical basis to localized basis)

    """
    H_core_rot = Unitary_rot.conj().T @ H_core @Unitary_rot 
    return H_core_rot

def Get_new_RKS_Veff(pyscf_RKS_obj: pyscf.dft.RKS, Unitary_rot: np.array, dm=None, check_result:bool=False) -> lib.tag_array:
    """
    Function to get V_eff in new basis. 

    Note in RKS calculation Veff = J + Vxc
    Whereas for RHF calc it is Veff = J - 0.5k

    Args:
        pyscf_RKS_obj (pyscf.dft.RKS): PySCF RKS obj
        Unitary_rot (np.array): Operator to change basis  (in this code base this should be: cannonical basis to localized basis)
        dm (np.array): Optional input density matrix. If not defined, finds whatever is available from pyscf_RKS_obj
        check_result (bool): Flag to check result against PySCF functions

    Returns:
        output (lib.tag_array): Tagged array containing J, K, E_coloumb, E_xcorr, Vxc
    """
    if dm is None:
        if pyscf_RKS_obj.mo_coeff is not None:
            density_mat = pyscf_RKS_obj.make_rdm1(pyscf_RKS_obj.mo_coeff, pyscf_RKS_obj.mo_occ)
        else:
            density_mat = pyscf_RKS_obj.init_guess_by_1e()
    else:
        density_mat = dm
    
    
    # Evaluate RKS/UKS XC functional and potential matrix on given meshgrids
    # for a set of density matrices.
    nelec, exc, vxc = numint.nr_vxc(pyscf_RKS_obj.mol,
                                            pyscf_RKS_obj.grids,
                                            pyscf_RKS_obj.xc,
                                            density_mat)

    # definition in new basis
    vxc =  Unitary_rot.conj().T @ vxc @ Unitary_rot
    
    
    Veff = RKS_get_veff(pyscf_RKS_obj, dm=density_mat)
    if Veff.vk is not None:
        K = Unitary_rot.conj().T @ Veff.vk @ Unitary_rot
        J = Unitary_rot.conj().T @ Veff.vj @ Unitary_rot
        vxc += J - K * .5
    else:
        J = Unitary_rot.conj().T @ Veff.vj @ Unitary_rot
        K = None
        vxc += J 
    
    if check_result is True:
        M1 = Unitary_rot.conj().T @ Veff.__array__() @ Unitary_rot
        if not np.allclose(vxc, M1):
            raise ValueError('Veff in new basis NOT correct')
    
    ecoul = np.einsum('ij,ji', density_mat, J).real * .5 # note J matrix is in new basis!
    ## this ecoul term changes if the full density matrix is NOT 
    # (aka for dm_active and dm_enviroment we get different V_eff under different bases!)
    
    output = lib.tag_array(vxc, ecoul=ecoul, exc=Veff.exc, vj=J, vk=K)
    return output

def Get_energy_and_matrices_from_dm_DFT(PySCF_RKS_obj: pyscf.dft.RKS, 
            dm_matrix: np.array, check_E_with_pyscf:bool=True) -> Tuple[float, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Calculate the components of subsystem energy from a DFT calculation
    
    Args:
        PySCF_RKS_obj (pyscf.dft.RKS): PySCF RKS object
        dm_matrix (np.array): density matrix (to calculate all matrices from)
        check_E_with_pyscf (bool): optional flag to check manual energy calc against PySCF calc     
    Returns:
        Energy_elec (float): DFT energy defubed by input density matrix 
        J_mat (np.array): J_matrix defined by input density matrix
        K_mat (np.array): K_matrix defined by input density matrix
        e_xc (float): exchange correlation energy defined by input density matrix 
        v_xc (np.array): V_exchangeCorrelation matrix defined by input density matrix (note Coloumbic contribution (J_mat) has been subtracted to give this term)
    """

    # It seems that PySCF lumps J and K in the J array 
    two_e_term =  PySCF_RKS_obj.get_veff(dm=dm_matrix)
    J_mat = two_e_term.vj
    K_mat = np.zeros_like(J_mat)
    
    e_xc = two_e_term.exc
    v_xc = two_e_term - J_mat 

    Energy_elec = (np.einsum('ij,ji->', PySCF_RKS_obj.get_hcore(), dm_matrix) + 
                   two_e_term.ecoul + two_e_term.exc)
    
    if check_E_with_pyscf:
        Energy_elec_pyscf = PySCF_RKS_obj.energy_elec(dm=dm_matrix)[0]
        if not np.isclose(Energy_elec_pyscf, Energy_elec):
            raise ValueError('Energy calculation incorrect')

    return Energy_elec, J_mat, K_mat, e_xc, v_xc


def Get_new_RHF_Veff(pyscf_RHF_obj: pyscf.hf.RHF, Unitary: np.array, dm=None, hermi:int=1) -> np.array:
    """
    Function to get V_eff in new basis. 

    Note in RKS calculation Veff = J + Vxc
    Whereas for RHF calc it is Veff = J - 0.5k

    Args:
        pyscf_RHF_obj (pyscf.hf.RHF): PySCF RHF obj
        Unitary_rot (np.array): Operator to change basis  (in this code base this should be: cannonical basis to localized basis)
        dm (np.array): Optional input density matrix. If not defined, finds whatever is available from pyscf_RKS_obj
        hermi (int): TODO
    """
    if dm is None:
        if pyscf_obj.mo_coeff is not None:
            density_mat = pyscf_obj.make_rdm1(pyscf_obj.mo_coeff, pyscf_obj.mo_occ)
        else:
            density_mat = pyscf_obj.init_guess_by_1e()
    else:
        density_mat = dm
    
    vj, vk = pyscf_obj.get_jk(dm=density_mat, hermi=hermi)
    Veff = vj - vk * .5
    
    # Veff = pyscf_obj.get_veff(dm=density_mat)
    Veff_new = Unitary.conj().T @ Veff @ Unitary

    return Veff_new

def Get_cross_terms_DFT(PySCF_RKS_obj: pyscf.dft.RKS, dm_active: np.array, dm_enviro: np.array, 
                    J_env: np.array, J_act: np.array, e_xc_act: float, e_xc_env: float) -> float:
    """
    Get two electron cross term energy. As Veff = J + Vxc, need Colombic cross term energy (J_cross) 
    and XC cross term energy

    Args:
        PySCF_RKS_obj (pyscf.dft.RKS): PySCF RKS object
        dm_active (np.array): density matrix of active subsystem
        dm_enviro (np.array): density matrix of enironment subsystem
        J_env (np.array): J_matrix defined by enviornemnt density
        J_act (np.array): J_matrix defined by active density 
        e_xc_act (float): exchange correlation energy defined by input active density matrix 
        e_xc_env (float): exchange correlation energy defined by input enviornemnt density matrix 

    Returns:
        two_e_cross (float): two electron energy from cross terms (includes exchange correlation and Coloumb contribution)
    """
    two_e_term_total =  PySCF_RKS_obj.get_veff(dm=dm_active+dm_enviro)
    e_xc_total = two_e_term_total.exc

    j_cross = 0.5 * ( np.einsum('ij,ij', dm_active, J_env) + np.einsum('ij,ij', dm_enviro, J_act) )
    k_cross = 0.0

    xc_cross = e_xc_total - e_xc_act - e_xc_env

    # overall two_electron cross energy
    two_e_cross = j_cross + k_cross + xc_cross
    
    return two_e_cross

def Enivornment_projector(C_loc_occ_and_virt, S_mat, active_MO_inds, enviro_MO_inds):
    """
    Get Projector onto environement MOs

    P_env = Σ_{i ∈ env} |MO_i> <MO_i| 
    
    Args:
        C_loc_occ_and_virt (np.array): C_matrix of localized MO (virtual and occupied)
        S_mat (np.array): AO overlap matrix
        active_MO_inds (np.array): 1D array of active MO indices
        enviro_MO_inds (np.array): 1D array of enviornemnt MO indices

    Returns:
        projector (np.array): Operator that projects environement MOs onto themselves and ative MOs onto zero vector
    """

    ## 1. convert to orthogonal C_matrix
    S_half = sp.linalg.fractional_matrix_power(S_mat, 0.5)
    S_neg_half = sp.linalg.fractional_matrix_power(S_mat, -0.5)

    Loc_Ortho = S_half@ C_loc_occ_and_virt # orthogonal C matrix (localized)

    ## 2. Define projector that projects MO orbs of subsystem B onto themselves and system A onto zero state!
    ##### (do this in orthongoal basis!)
    ### not we only take MO environment indices!
    PROJ_ortho = np.einsum('ik,jk->ij', Loc_Ortho[:, enviro_MO_inds], Loc_Ortho[:, enviro_MO_inds])
    # PROJ_ortho = np.zeros_like(S_mat)
    # for MO_ind in range(C_all_localized_and_virt.shape[1]):
    #     if MO_ind in enviro_MO_inds:
    #         outer = np.outer(Loc_Ortho[:, MO_ind], Loc_Ortho[:, MO_ind])
    #         PROJ_ortho+=outer
    #     else:
    #         continue


    print(f'''Are subsystem B (env) projected onto themselves in ORTHO basis: {
            np.allclose(PROJ_ortho@Loc_Ortho[:, enviro_MO_inds], 
            Loc_Ortho[:, enviro_MO_inds])}''') # projected onto itself

    print(f'''Is subsystem A traced out  in ORTHO basis?: {
            np.allclose(PROJ_ortho@Loc_Ortho[:, active_MO_inds], 
            np.zeros_like(Loc_Ortho[:, active_MO_inds]))}''') # # projected onto zeros!



    ##### 3. Define projector in standard (non-orthogonal basis)
    projector = S_half @ PROJ_ortho  @ S_half

    print(f'''Are subsystem B (env) projected onto themselves in ORTHO basis: {
            np.allclose(projector@C_loc_occ_and_virt[:, enviro_MO_inds], 
            C_loc_occ_and_virt[:, enviro_MO_inds])}''') # projected onto itself

    print(f'''Is subsystem A traced out  in ORTHO basis?: {
            np.allclose(PROJ_ortho@C_loc_occ_and_virt[:, active_MO_inds], 
            np.zeros_like(C_loc_occ_and_virt[:, active_MO_inds]))}''') # # projected onto zeros!

    return projector



def get_active_indices(
    scf_method: StreamObject,
    n_act_mos: int,
    n_env_mos: int,
    qubits: Optional[int] = None,
) -> np.ndarray:
    """Return an array of active indices for QHam construction.

    Args:
        scf_method (StreamObject): A pyscf self consisten method.
        n_act_mos (int): Number of active-space moleclar orbitals.
        n_env_mos (int): Number of environment moleclar orbitals.
        qubits (int): Number of qubits to be used in final calclation.

    Returns:
        np.ndarray: A 1D array of integer indices.
    """
    # Find the active indices
    active_indices = [i for i in range(len(scf_method.mo_occ) - n_env_mos)]

    # This is not the best way to simplify.
    # TODO some more sophisticated thing with frozen core
    # rather than just cutting high level MOs
    if qubits:
        # Check that the reduction is sensible
        # Needs 1 qubit per spin state
        if qubits < 2 * n_act_mos:
            raise Exception(f"Not enouch qubits for active MOs, minimum {2*n_act_mos}.")

        logger.info("Restricting to low level MOs for %s qubits.", qubits)
        active_indices = active_indices[: qubits // 2]

    return np.array(active_indices)


def get_qubit_hamiltonian(
    scf_method: StreamObject, active_indices: List[int]
) -> object:
    """Return the qubit hamiltonian.

    Args:
        scf_method (StreamObject): A pyscf self-consistent method.
        active_indices (list[int]): A list of integer indices of active moleclar orbitals.

    Returns:
        object: A qubit hamiltonian.
    """
    n_orbs = len(active_indices)

    mo_coeff = scf_method.mo_coeff[:, active_indices]

    one_body_integrals = mo_coeff.T @ scf_method.get_hcore() @ mo_coeff

    # temp_scf.get_hcore = lambda *args, **kwargs : initial_h_core
    scf_method.mol.incore_anyway is True

    # Get two electron integrals in compressed format.
    two_body_compressed = ao2mo.kernel(scf_method.mol, mo_coeff)

    two_body_integrals = ao2mo.restore(
        1, two_body_compressed, n_orbs  # no permutation symmetry
    )

    # Openfermion uses pysicist notation whereas pyscf uses chemists
    two_body_integrals = np.asarray(two_body_integrals.transpose(0, 2, 3, 1), order="C")

    one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
        one_body_integrals, two_body_integrals
    )

    molecular_hamiltonian = InteractionOperator(
        0, one_body_coefficients, 0.5 * two_body_coefficients
    )

    Qubit_Hamiltonian = jordan_wigner(molecular_hamiltonian)

    return Qubit_Hamiltonian


def nbed(
    geometry: Path,
    active_atoms: int,
    basis: str,
    xc_functional: str,
    output: str,
    convergence: float = 1e-6,
    localisation: str = "spade",
    level_shift: float = 1e6,
    run_ccsd: bool = False,
    qubits: int = None,
) -> Tuple[object, float]:
    """Function to return the embedding Qubit Hamiltonian.

    Args:
        geometry (Path): A path to an .xyz file describing moleclar geometry.
        active_atoms (int): The number of atoms to include in the active region.
        basis (str): The name of an atomic orbital basis set to use for chemistry calculations.
        xc_functonal (str): The name of an Exchange-Correlation functional to be used for DFT.
        output (str): one of "Openfermion" (TODO other options)
        convergence (float): The convergence tolerance for energy calculations.
        localisation (str): Orbital Localisation method to use. One of 'spade', 'mullikan', 'boys' or 'ibo'.
        level_shift (float): Level shift parameter to use for mu-projector.
        run_ccsd (bool): Whether or not to find the CCSD energy of the system for reference.
        qubits (int): The number of qubits available for the output hamiltonian.

    Returns:
        object: A Qubit Hamiltonian of some kind
        float: The classical contribution to the total energy.

    """
    logger.debug("Construcing molecule.")
    mol: gto.Mole = gto.Mole(atom=geometry, basis=basis, charge=0).build()

    e_nuc = mol.energy_nuc()
    logger.debug(f"Nuclear energy: {e_nuc}.")

    ks = scf.RKS(mol)
    ks.conv_tol = convergence
    ks.xc = xc_functional
    ks.run()

    # Function names must be the same as the imput choices.
    logger.debug(f"Using {localisation} localisation method.")
    loc_method = globals()[localisation]
    n_act_mos, n_env_mos, act_density, env_density = loc_method(ks, active_atoms)

    # Get cross terms from the initial density
    logger.debug("Calculating cross subsystem terms.")
    e_act, e_xc_act, j_act, k_act, v_xc_act = closed_shell_subsystem(ks, act_density)
    e_env, e_xc_env, j_env, k_env, v_xc_env = closed_shell_subsystem(ks, env_density)

    active_indices = get_active_indices(ks, n_act_mos, n_env_mos, qubits)

    # Computing cross subsystem terms
    # Note that the matrix dot product is equivalent to the trace.
    j_cross = 0.5 * (
        np.einsum("ij,ij", act_density, j_env) + np.einsum("ij,ij", env_density, j_act)
    )

    k_cross = 0.0

    xc_cross = ks.get_veff().exc - e_xc_act - e_xc_env
    two_e_cross = j_cross + k_cross + xc_cross

    # Define the mu-projector
    projector = level_shift * (ks.get_ovlp() @ env_density @ ks.get_ovlp())

    v_xc_total = ks.get_veff() - ks.get_j()

    # Defining the embedded core Hamiltonian
    v_emb = j_env + v_xc_total - v_xc_act + projector

    # Run RHF with Vemb to do embedding
    embedded_scf = scf.RHF(mol)
    embedded_scf.conv_tol = convergence
    embedded_scf.mol.nelectron = 2 * n_act_mos

    h_core = embedded_scf.get_hcore()

    embedded_scf.get_hcore = lambda *args, **kwargs: h_core + v_emb

    embedded_scf.kernel()

    embedded_occ_orbs = embedded_scf.mo_coeff[:, embedded_scf.mo_occ > 0]
    embedded_density = 2 * embedded_occ_orbs @ embedded_occ_orbs.T

    # if "complex" in embedded_occ_orbs.dtype.name:
    #     act_density = act_density.real
    #     env_density = env_density.real
    #     embedded_density = embedded_density.real
    #     embedded_scf.mo_coeff = embedded_scf.mo_coeff.real
    #     print("WARNING - IMAGINARY PARTS TO DENSITY")

    embedded_scf.get_hcore = lambda *args, **kwargs: h_core + v_emb

    # Calculate energy correction
    # - There are two versions used for different embeddings
    dm_correction = np.einsum("ij,ij", v_emb, embedded_density - act_density)
    wf_correction = np.einsum("ij,ij", act_density, v_emb)

    e_wf_act = embedded_scf.energy_elec(
        dm=embedded_density, vhf=embedded_scf.get_veff()
    )[0]

    if run_ccsd:
        # Run CCSD as WF method
        ccsd = cc.CCSD(embedded_scf)
        ccsd.conv_tol = convergence

        # Set which orbitals are to be frozen
        shift = mol.nao - n_env_mos
        fos = [i for i in range(shift, mol.nao)]
        ccsd.frozen = fos

        try:
            ccsd.run()
            correlation = ccsd.e_corr
            e_wf_act += correlation
        except np.linalg.LinAlgError as e:
            print("\n====CCSD ERROR====\n")
            print(e)

        # Add up the parts again
        e_wf_emb = e_wf_act + e_env + two_e_cross + e_nuc - wf_correction

        print("CCSD Energy:\n\t%s", e_wf_emb)

    # WF Method
    # Calculate the energy of embedded A
    # embedded_scf.get_hcore = lambda *args, **kwargs: h_core

    # Quantum Method
    q_ham = get_qubit_hamiltonian(embedded_scf, active_indices)

    # TODO Change the output type here
    if output.lower() != "openfermion":
        raise NotImplementedError(
            "No output format other than 'OpenFermion' is implemented."
        )

    classical_energy = e_env + two_e_cross + e_nuc - wf_correction

    return q_ham, classical_energy


def cli() -> None:
    """CLI Interface."""
    setup_logs()
    args = parse()
    qham, e_classical = nbed(
        geometry=args["geometry"],
        active_atoms=args["active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        output=args["output"],
        localisation=args["localisation"],
        convergence=args["convergence"],
        run_ccsd=args["ccsd"],
        qubits=args["qubits"],
    )
    print(f"Qubit Hamiltonian:")
    print(qham)
    print(f"Classical Energy (Ha): {e_classical}")


if __name__ == "__main__":
    cli()
