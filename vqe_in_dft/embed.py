"""Main embedding functionality."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import scipy as sp
import numpy as np
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import jordan_wigner
from pyscf import ao2mo, cc, gto, scf
from pyscf.lib import StreamObject

from vqe_in_dft.utils import parse, setup_logs

import pyscf

from pyscf.dft.rks import get_veff as rks_get_veff
from pyscf import lib
from pyscf.dft import numint

logger = logging.getLogger(__name__)
setup_logs()


def get_Hcore_new_basis(h_core: np.array, unitary_rot: np.ndarray) -> np.array:
    """
    Function to get H_core in new basis

    Args:
        h_core (np.ndarray): standard core Hamiltonian
        unitary_rot (np.ndarray): Operator to change basis  (in this code base this should be: cannonical basis to
        localized basis)
    Returns:
        H_core_rot (np.ndarray): core Hamiltonian in new basis
    """
    H_core_rot = unitary_rot.conj().T @ h_core @ unitary_rot
    return H_core_rot


def get_new_RKS_Veff(pyscf_RKS: StreamObject, unitary_rot: np.ndarray, dm=None,
                     check_result: bool = False) -> lib.tag_array:
    """
    Function to get V_eff in new basis.  Note this function is based on: pyscf.dft.rks.get_veff

    Note in RKS calculation Veff = J + Vxc
    Whereas for RHF calc it is Veff = J - 0.5k

    Args:
        pyscf_RKS (StreamObject): PySCF RKS obj
        unitary_rot (np.ndarray): Operator to change basis  (in this code base this should be: cannonical basis
                                to localized basis)
        dm (np.ndarray): Optional input density matrix. If not defined, finds whatever is available from pyscf_RKS_obj
        check_result (bool): Flag to check result against PySCF functions

    Returns:
        output (lib.tag_array): Tagged array containing J, K, E_coloumb, E_xcorr, Vxc
    """
    if dm is None:
        if pyscf_RKS.mo_coeff is not None:
            dm = pyscf_RKS.make_rdm1(pyscf_RKS.mo_coeff, pyscf_RKS.mo_occ)
        else:
            dm = pyscf_RKS.init_guess_by_1e()

    # Evaluate RKS/UKS XC functional and potential matrix on given meshgrids
    # for a set of density matrices.
    nelec, exc, vxc = numint.nr_vxc(pyscf_RKS.mol,
                                    pyscf_RKS.grids,
                                    pyscf_RKS.xc,
                                    dm)

    # definition in new basis
    vxc = unitary_rot.conj().T @ vxc @ unitary_rot
    
    v_eff = rks_get_veff(pyscf_RKS, dm=dm)
    if v_eff.vk is not None:
        k_mat = unitary_rot.conj().T @ v_eff.vk @ unitary_rot
        j_mat = unitary_rot.conj().T @ v_eff.vj @ unitary_rot
        vxc += j_mat - k_mat * .5
    else:
        j_mat = unitary_rot.conj().T @ v_eff.vj @ unitary_rot
        k_mat = None
        vxc += j_mat
    
    if check_result is True:
        veff_check = unitary_rot.conj().T @ v_eff.__array__() @ unitary_rot
        if not np.allclose(vxc, veff_check):
            raise ValueError('Veff in new basis does not match rotated PySCF value.')

    # note J matrix is in new basis!
    ecoul = np.einsum('ij,ji', dm, j_mat).real * .5
    # this ecoul term changes if the full density matrix is NOT
    #    (aka for dm_active and dm_enviroment we get different V_eff under different bases!)
    
    output = lib.tag_array(vxc, ecoul=ecoul, exc=v_eff.exc, vj=j_mat, vk=k_mat)
    return output


def calc_RKS_components_from_dm(pyscf_RKS: StreamObject,
              dm_matrix: np.ndarray, check_E_with_pyscf: bool = True) \
              -> Tuple[float, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Calculate the components of subsystem energy from a RKS DFT calculation.

    For a given density matrix this function returns the electronic energy, exchange correlation energy and
    J,K, V_xc matrices.
    
    Args:
        pyscf_RKS (StreamObject): PySCF RKS object
        dm_matrix (np.ndarray): density matrix (to calculate all matrices from)
        check_E_with_pyscf (bool): optional flag to check manual energy calc against PySCF calc     
    Returns:
        Energy_elec (float): DFT energy defubed by input density matrix 
        J_mat (np.ndarray): J_matrix defined by input density matrix
        K_mat (np.ndarray): K_matrix defined by input density matrix
        e_xc (float): exchange correlation energy defined by input density matrix 
        v_xc (np.ndarray): V_exchangeCorrelation matrix defined by input density matrix (note Coloumbic
                         contribution (J_mat) has been subtracted to give this term)
    """

    # It seems that PySCF lumps J and K in the J array 
    two_e_term = pyscf_RKS.get_veff(dm=dm_matrix)
    j_mat = two_e_term.vj
    k_mat = np.zeros_like(j_mat)
    
    e_xc = two_e_term.exc
    v_xc = two_e_term - j_mat

    energy_elec = (np.einsum('ij,ji->', pyscf_RKS.get_hcore(), dm_matrix) +
                   two_e_term.ecoul + two_e_term.exc)
    
    if check_E_with_pyscf:
        energy_elec_pyscf = pyscf_RKS.energy_elec(dm=dm_matrix)[0]
        if not np.isclose(energy_elec_pyscf, energy_elec):
            raise ValueError('Energy calculation incorrect')

    return energy_elec, j_mat, k_mat, e_xc, v_xc


def get_new_RHF_Veff(pyscf_RHF: StreamObject, unitary_rot: np.ndarray, dm=None, hermi: int = 1) -> np.ndarray:
    """
    Function to get V_eff in new basis. 

    Note in RKS calculation Veff = J + Vxc
    Whereas for RHF calc it is Veff = J - 0.5k

    Args:
        pyscf_RHF (StreamObject): PySCF RHF obj
        unitary_rot (np.ndarray): Operator to change basis  (in this code base this should be: cannonical basis
                                to localized basis)
        dm (np.ndarray): Optional input density matrix. If not defined, finds whatever is available from pyscf_RKS_obj
        hermi (int): TODO
    """
    if dm is None:
        if pyscf_RHF.mo_coeff is not None:
            dm = pyscf_RHF.make_rdm1(pyscf_RHF.mo_coeff, pyscf_RHF.mo_occ)
        else:
            dm = pyscf_RHF.init_guess_by_1e()

    vj, vk = pyscf_RHF.get_jk(dm=dm, hermi=hermi)
    v_eff = vj - vk * .5
    
    # v_eff = pyscf_obj.get_veff(dm=dm)
    v_eff_new = unitary_rot.conj().T @ v_eff @ unitary_rot

    return v_eff_new


def get_cross_terms_DFT(pyscf_RKS: StreamObject, dm_active: np.ndarray, dm_enviro: np.ndarray,
                        j_env: np.ndarray, j_act: np.ndarray, e_xc_act: float, e_xc_env: float) -> float:
    """
    Get two electron cross term energy. As Veff = J + Vxc, need Colombic cross term energy (J_cross) 
    and XC cross term energy

    Args:
        pyscf_RKS (StreamObject): PySCF RKS object
        dm_active (np.ndarray): density matrix of active subsystem
        dm_enviro (np.ndarray): density matrix of enironment subsystem
        j_env (np.ndarray): J_matrix defined by enviornemnt density
        j_act (np.ndarray): J_matrix defined by active density
        e_xc_act (float): exchange correlation energy defined by input active density matrix 
        e_xc_env (float): exchange correlation energy defined by input enviornemnt density matrix 

    Returns:
        two_e_cross (float): two electron energy from cross terms (includes exchange correlation
                             and Coloumb contribution)
    """
    two_e_term_total = pyscf_RKS.get_veff(dm=dm_active+dm_enviro)
    e_xc_total = two_e_term_total.exc

    j_cross = 0.5 * (np.einsum('ij,ij', dm_active, j_env) + np.einsum('ij,ij', dm_enviro, j_act))
    k_cross = 0.0

    xc_cross = e_xc_total - e_xc_act - e_xc_env

    # overall two_electron cross energy
    two_e_cross = j_cross + k_cross + xc_cross
    
    return two_e_cross


def get_enivornment_projector(c_loc_occ_and_virt: np.ndarray, s_mat: np.ndarray,
                              active_MO_inds: np.ndarray, enviro_MO_inds: np.ndarray) -> np.ndarray:
    """
    Get projector onto environement MOs

    P_env = Σ_{i ∈ env} |MO_i> <MO_i| 
    
    Args:
        c_loc_occ_and_virt (np.ndarray): C_matrix of localized MO (virtual and occupied)
        s_mat (np.ndarray): AO overlap matrix
        active_MO_inds (np.ndarray): 1D array of active MO indices
        enviro_MO_inds (np.ndarray): 1D array of enviornemnt MO indices

    Returns:
        projector (np.ndarray): Operator that projects environement MOs onto themselves and ative MOs onto zero vector
    """

    # 1. convert to orthogonal C_matrix
    s_half = sp.linalg.fractional_matrix_power(s_mat, 0.5)

    # orthogonal C matrix (localized)
    c_loc_ortho = s_half @ c_loc_occ_and_virt

    # 2. Define projector that projects MO orbs of subsystem B onto themselves and system A onto zero state!
    #    (do this in orthongoal basis!)
    #    note we only take MO environment indices!
    ortho_proj = np.einsum('ik,jk->ij', c_loc_ortho[:, enviro_MO_inds], c_loc_ortho[:, enviro_MO_inds])

    # env projected onto itself
    logger.info(f'''Are subsystem B (env) projected onto themselves in ORTHO basis: {
            np.allclose(ortho_proj @ c_loc_ortho[:, enviro_MO_inds], 
            c_loc_ortho[:, enviro_MO_inds])}''')

    # act projected onto zero vec
    logger.info(f'''Is subsystem A traced out  in ORTHO basis?: {
            np.allclose(ortho_proj @ c_loc_ortho[:, active_MO_inds], 
            np.zeros_like(c_loc_ortho[:, active_MO_inds]))}''')

    # 3. Define projector in standard (non-orthogonal basis)
    projector = s_half @ ortho_proj  @ s_half

    logger.info(f'''Are subsystem B (env) projected onto themselves in ORTHO basis: {
            np.allclose(projector @ c_loc_occ_and_virt[:, enviro_MO_inds], 
            c_loc_occ_and_virt[:, enviro_MO_inds])}''')

    logger.info(f'''Is subsystem A traced out  in ORTHO basis?: {
            np.allclose(projector@c_loc_occ_and_virt[:, active_MO_inds], 
            np.zeros_like(c_loc_occ_and_virt[:, active_MO_inds]))}''')

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
    # dm_correction = np.einsum("ij,ij", v_emb, embedded_density - act_density)
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
    print("Qubit Hamiltonian:")
    print(qham)
    print(f"Classical Energy (Ha): {e_classical}")


if __name__ == "__main__":
    cli()
