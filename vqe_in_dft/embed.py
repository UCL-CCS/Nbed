import argparse
from copy import copy
from ctypes import wstring_at
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
from pyscf.gto import basis
from pyscf.lib.misc import alias
import yaml

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.linalg import eigenspectrum, expectation
from openfermion.ops.representations import (
    InteractionOperator,
    get_active_space_integrals,
)
from openfermion.transforms import jordan_wigner
from pyscf import ao2mo, cc, fci, gto, scf
from scipy import linalg


def parse():
    parser = argparse.ArgumentParser(description="Output embedded Qubit Hamiltonian.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a config file. Overwrites other arguments."
    )
    parser.add_argument(
        "--geometry",
        type=str,
        help="Path to an XYZ file.",
    )
    parser.add_argument(
        "--active_atoms",
        "--active",
        type=int,
        help="Number of atoms to include in active region.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        help="Basis set to use.",
    )
    parser.add_argument(
        "--xc_functional",
        "--xc",
        "--functional",
        type=str,
        help="Exchange correlation functional to use in DFT calculations.",
    )
    parser.add_argument(
        "--convergence",
        "--conv",
        type=float,
        help="Convergence tolerance for calculations.",
    )
    parser.add_argument(
        "--output",
        type=str,
        choice=["openfermion"],
        help="Quantum computing backend to output the qubit hamiltonian for.",
    )
    args = parser.parse_args()

    if args.config:
        args = yaml.safe_load(args.config)
    return args

def get_exact_energy(mol: gto.Mole, keywords: Dict):
    hf = mol.RHF().run()

    ref_fci = fci.FCI(hf)
    ref_fci.conv_tol = keywords["e_convergence"]
    fci_result = ref_fci.kernel()

    return fci_result[0]


def spade_localisation(scf_method: Callable, active_atoms: int):
    n_occupied_orbitals = np.count_nonzero(scf_method.mo_occ == 2)
    occupied_orbitals = scf_method.mo_coeff[:, :n_occupied_orbitals]

    n_act_aos = scf_method.mol.aoslice_by_atom()[active_atoms - 1][-1]
    ao_overlap = scf_method.get_ovlp()

    # Orbital rotation and partition into subsystems A and B
    # rotation_matrix, sigma = embed.orbital_rotation(occupied_orbitals,
    #    n_act_aos, ao_overlap)

    rotated_orbitals = (
        linalg.fractional_matrix_power(ao_overlap, 0.5) @ occupied_orbitals
    )
    _, sigma, right_vectors = linalg.svd(rotated_orbitals[:n_act_aos, :])

    print(f"Singular Values: {sigma}")

    # n_act_mos, n_env_mos = embed.orbital_partition(sigma)
    value_diffs = sigma[:-1] - sigma[1:]
    n_act_mos = np.argmax(value_diffs) + 1
    n_env_mos = n_occupied_orbitals - n_act_mos

    # Defining active and environment orbitals and density
    act_orbitals = occupied_orbitals @ right_vectors.T[:, :n_act_mos]
    env_orbitals = occupied_orbitals @ right_vectors.T[:, n_act_mos:]
    act_density = 2.0 * act_orbitals @ act_orbitals.T
    env_density = 2.0 * env_orbitals @ env_orbitals.T
    return n_act_mos, n_env_mos, act_density, env_density


def closed_shell_subsystem(scf_method: Callable, density: np.ndarray):
    # It seems that PySCF lumps J and K in the J array
    j = scf_method.get_j(dm=density)
    k = np.zeros(np.shape(j))
    two_e_term = scf_method.get_veff(scf_method.mol, density)
    e_xc = two_e_term.exc
    v_xc = two_e_term - j

    # Energy
    e = np.einsum("ij,ij", density, scf_method.get_hcore() + j / 2) + e_xc
    return e, e_xc, j, k, v_xc


def get_active_indices(
    scf_method: Callable, n_act_mos: int, n_env_mos: int, reduction: Optional[int] = None
) -> np.ndarray:
    nao = scf_method.mol.nao
    max_reduction = nao - n_act_mos - n_env_mos

    # Check that the reduction is sensible
    # Needs to set to none for slice
    if reduction > max_reduction:
        raise Exception(f"Active space reduction too large. Max size {max_reduction}")

    # Find the active indices
    active_indices = [i for i in range(len(scf_method.mo_occ) - n_env_mos)]

    if reduction:
        active_indices = active_indices[:-reduction]

    return np.array(active_indices)


def get_qubit_hamiltonian(scf_method: Callable, active_indices: List[int]):

    n_orbs = len(active_indices)

    mo_coeff = scf_method.mo_coeff[:, active_indices]

    one_body_integrals = mo_coeff.T @ scf_method.get_hcore() @ mo_coeff

    # temp_scf.get_hcore = lambda *args, **kwargs : initial_h_core
    scf_method.mol.incore_anyway == True

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


def embedding_hamiltonian(
    geometry: Path,
    active_atoms: int,
    basis: str,
    xc_functional: str,
    convergence: float,
    output: str,
    level_shift: float = 1e6,
    run_ccsd: bool = False,
    ) -> Tuple[Object, float]:
    """
    Function to return the embedding Qubit Hamiltonian.
    """

    mol: gto.Mole = gto.Mole(atom=geometry, basis=basis, charge=0).build()

    e_nuc = mol.energy_nuc()

    ks = scf.RKS(mol)
    ks.conv_tol = convergence
    ks.xc = xc_functional

    n_act_mos, n_env_mos, act_density, env_density = spade_localisation(ks, active_atoms)

    # Get cross terms from the initial density
    e_act, e_xc_act, j_act, k_act, v_xc_act = closed_shell_subsystem(ks, act_density)
    e_env, e_xc_env, j_env, k_env, v_xc_env = closed_shell_subsystem(ks, env_density)

    active_indices = get_active_indices(ks, n_act_mos, n_env_mos, 0)

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
    else:
        e_wf_emb = 0

    # WF Method
    # Calculate the energy of embedded A
    # embedded_scf.get_hcore = lambda *args, **kwargs: h_core

    # Quantum Method
    q_ham = get_qubit_hamiltonian(embedded_scf, active_indices)

    classical_energy = e_env + two_e_cross + e_nuc - wf_correction

    return q_ham, classical_energy

if __name__ == "__main__":
    args = parse()
    embedding_hamiltonian(args**)

