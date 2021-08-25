from copy import copy
from ctypes import wstring_at
from typing import Dict, Optional
from spade import fill_defaults
import numpy as np
from scipy import linalg
from pyscf import gto, scf, cc, ao2mo, fci
import pandas as pd
from pathlib import Path
from pyscf import ao2mo
from openfermion.ops.representations import (
    InteractionOperator,
    get_active_space_integrals,
)
from openfermion.linalg import eigenspectrum, expectation
from openfermion.transforms import jordan_wigner
from typing import List, Dict, Any

# Xserver is not set up on my WSL so turn of display
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

import argparse


def parse():
    parser = argparse.ArgumentParser(description="Get results for MRes")
    parser.add_argument(
        "--comparison",
        type=str,
        choices=["method", "space"],
        default=2,
        help="Number of active atoms",
    )
    parser.add_argument(
        "--as_reduction",
        type=int,
        default=0,
        help="Number of orbitals to remove from active space.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save the results to a csv file.",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="Plot the results."
    )
    args = parser.parse_args()

    return args


def get_exact_energy(mol: gto.Mole, keywords: Dict):
    hf = mol.RHF().run()

    ref_fci = fci.FCI(hf)
    ref_fci.conv_tol = keywords["e_convergence"]
    fci_result = ref_fci.kernel()

    return fci_result[0]


def spade_localisation(scf_method, keywords):
    n_occupied_orbitals = np.count_nonzero(scf_method.mo_occ == 2)
    occupied_orbitals = scf_method.mo_coeff[:, :n_occupied_orbitals]

    n_act_aos = scf_method.mol.aoslice_by_atom()[keywords["n_active_atoms"] - 1][-1]
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


def closed_shell_subsystem(scf_method, density):
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
    scf_method, n_act_mos: int, n_env_mos: int, reduction: Optional[int] = None
) -> np.ndarray:
    nao = scf_method.mol.nao
    max_reduction = nao - n_act_mos - n_env_mos

    # Check that the reduction is sensible
    # Needs to set to none for slice
    if reduction > max_reduction:
        raise Exception(f"Active space reduction too large. Max size {max_reduction}")

    # Find the active indices
    env_indices = [i + n_act_mos for i in range(n_env_mos)]
    active_indices = [i for i in range(nao) if i not in env_indices]

    if reduction:
        active_indices = active_indices[:-reduction]

    return np.array(active_indices)


def get_qubit_hamiltonian(scf_method, active_indices: List[int]):
    one_body_ints = scf_method.mo_coeff.T @ scf_method.get_hcore() @ scf_method.mo_coeff

    # Get the 1e and 2e integrals for whole system
    eri = scf_method.mol.intor("int2e", aosym=1)
    mo_eri = ao2mo.incore.full(eri, scf_method.mo_coeff, compact=False)
    nao = scf_method.mol.nao
    two_body_ints = mo_eri.reshape(nao, nao, nao, nao)
    _, act_one_body, act_two_body = get_active_space_integrals(
        one_body_ints, two_body_ints, active_indices=active_indices
    )

    molecular_hamiltonian = InteractionOperator(0, act_one_body, 0.5 * act_two_body)

    Qubit_Hamiltonian = jordan_wigner(molecular_hamiltonian)

    return Qubit_Hamiltonian


def run_sim(
    mol: gto.Mole,
    keywords: Dict[str, Any],
    orbital_reduction: int = None,
):

    if orbital_reduction is None:
        orbital_reduction = 0

    e_nuc = mol.energy_nuc()

    ks = scf.RKS(mol)
    ks.conv_tol = keywords["e_convergence"]
    ks.xc = keywords["low_level"]
    e_initial = ks.kernel()

    n_act_mos, n_env_mos, act_density, env_density = spade_localisation(ks, keywords)

    # Get cross terms from the initial density
    e_act, e_xc_act, j_act, k_act, v_xc_act = closed_shell_subsystem(ks, act_density)
    e_env, e_xc_env, j_env, k_env, v_xc_env = closed_shell_subsystem(ks, env_density)

    active_indices = get_active_indices(ks, n_act_mos, n_env_mos, orbital_reduction)

    # Computing cross subsystem terms
    # Note that the matrix dot product is equivalent to the trace.
    j_cross = 0.5 * (
        np.einsum("ij,ij", act_density, j_env) + np.einsum("ij,ij", env_density, j_act)
    )

    k_cross = 0.0

    xc_cross = ks.get_veff().exc - e_xc_act - e_xc_env
    two_e_cross = j_cross + k_cross + xc_cross

    # Define the mu-projector
    projector = keywords["level_shift"] * (ks.get_ovlp() @ env_density @ ks.get_ovlp())

    v_xc_total = ks.get_veff() - ks.get_j()

    # Defining the embedded core Hamiltonian
    v_emb = j_env + v_xc_total - v_xc_act + projector

    # Run RHF with Vemb to do embedding
    embedded_scf = scf.RHF(mol)
    embedded_scf.conv_tol = keywords["e_convergence"]
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

    print(f"{wf_correction=}, {dm_correction=}")

    # Can use either of these methods
    # This needs to change if we're not using PySCFEmbed
    # The j and k matrices are defined differently in PySCF and Psi4
    e_wf_act = embedded_scf.energy_elec(
        dm=embedded_density, vhf=embedded_scf.get_veff()
    )[0]

    # Run CCSD as WF method
    ccsd = cc.CCSD(embedded_scf)
    ccsd.conv_tol = keywords["e_convergence"]

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
    
    # WF Method
    # Calculate the energy of embedded A
    embedded_scf.get_hcore = lambda *args, **kwargs: h_core

    # Quantum Method
    q_ham = get_qubit_hamiltonian(embedded_scf, active_indices)
    e_vqe_act = eigenspectrum(q_ham)[0]

    # Add all the parts up
    e_vqe_emb = e_vqe_act + e_env + two_e_cross - wf_correction + e_nuc


    # Add up the parts again
    e_wf_emb = e_wf_act + e_env + two_e_cross + e_nuc - wf_correction

    print("Component contributions")
    print(
        f"{e_vqe_act=}, {e_wf_act=}, {e_env=}, {two_e_cross=}, {wf_correction=}, {e_nuc=}\n"
    )

    return e_initial, e_vqe_emb, e_wf_emb


def plot_methods(distances, values, mol_name):
    # Plot the results with no active space reduction
    mins = {}
    lengths = {}
    fig = plt.figure()
    for key, data in values[0].items():
        mins[key] = min([i for i in data if i is not None])
        lengths[key] = distances[np.argmin(data)]
        plt.plot(distances, data, label=f"{key} (Length={lengths[key]:.3f} Emin={mins[key]:.3f})")


    plt.legend()
    plt.xlabel(r"Distance ($a_0$)")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{mol_name} Embedding Methods")
    figpath = Path(__file__).parent / f"data/{mol_name}-{0}-{pd.Timestamp.now()}.png"
    plt.savefig(figpath)


def plot_vqe_reductions(distances, values, mol_name, mo_reduction):
    # Plot vqe with reduced active space
    mins = {}
    fig = plt.figure()
    vqe_data = {i: values[i]["VQE"] for i in mo_reduction}
    for key, data in vqe_data.items():
        mins[key] = min([i for i in data if i is not None])
        plt.plot(distances, data, label=f"Space Reduction={key} (min={mins[key]:.3f})")

    plt.legend()
    plt.xlabel(r"Distance ($a_0$)")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{mol_name} VQE Active Space Reduction")
    figpath = Path(__file__).parent / f"data/{mol_name}-VQE-{pd.Timestamp.now()}.png"
    plt.savefig(figpath)


def main(
    dmin: int,
    dmax: int,
    mo_reduction: List[int] = None, 
    save: bool = False, 
    plot: bool = False,
    test: bool = False,
    max_runs: int = 100,
    ):
    "Run for some molecule and a range of values"

    water_geometry = """
    O          0.00000        0.00000        0.1653507
    H          0.00000        0.7493682     -0.4424329
    H          0.00000       -0.7493682     -0.4424329
        """

    options = {}
    options["basis"] = "STO-6G"  # basis set
    options["low_level"] = "b3lyp"  # level of theory of the environment
    options["high_level"] = "mp2"  # level of theory of the embedded system
    options[
        "n_active_atoms"
    ] = 1  # number of active atoms (first n atoms in the geometry string)
    options["low_level_reference"] = "rhf"
    options["high_level_reference"] = "rhf"
    options["package"] = "pyscf"

    keywords = fill_defaults(options)

    # Assuming we want to look at water later.
    options["geometry"] = water_geometry

    if not test:
        # max runs is presumptively 100
        #num_runs = int(10 * (dmax - dmin)) if dmax-dmin < max_runs / 10 else max_runs 
        distances = np.concatenate((np.linspace(0.5, 1.49, 9), np.linspace(1.50, 1.6, 40), np.linspace(1.7, 3, 13)))
        #distances = np.linspace(dmin, dmax, num_runs)
    else:
        distances = np.linspace(dmin, dmax, 3)
        mo_reduction = mo_reduction[:1]
        save = False
        plot = True

    values = {
        i: {
            "FCI": [],
            "DFT": [],
            "WF-in-DFT": [],
            #"VQE-in-DFT": [],
        }
        for i in mo_reduction
    }
    mol_name = "LiH"
    for distance in distances:
        options[
            "geometry"
        ] = f"""
        Li 0.0 0.0 0.0
        H 0.0 0.0 {distance}
        """
        # options["geometry"] = water_geometry

        mol = gto.Mole(
            atom=keywords["geometry"], basis=keywords["basis"], charge=0
        ).build()
        expected_energy = get_exact_energy(mol, keywords)

        for reduction in mo_reduction:
            mol = gto.Mole(
                atom=keywords["geometry"], basis=keywords["basis"], charge=0
            ).build()

            e_initial, e_vqe_emb, e_wf_emb = run_sim(mol, keywords, reduction)

            # Print out the final value.
            print(f"Calculating with {distance=}")
            print(f"Active space {reduction=}")
            print(f"FCI Energy:\t\t\t{expected_energy}\n")
            print(f"Full system low-level Energy:\t{e_initial}")
            print(
                f"Error:\t\t\t\t{(expected_energy-e_initial)*100/expected_energy:.2f}%\n"
            )
            print(f"VQE-in-DFT Energy:\t\t{e_vqe_emb}")
            print(
                f"Error:\t\t\t\t{(expected_energy-e_vqe_emb)*100/expected_energy:.2f}%\n"
            )
            print(f"WF-in-DFT Energy:\t\t{e_wf_emb}")
            print(
                f"Error:\t\t\t\t{(expected_energy-e_wf_emb)*100/expected_energy:.2f}%\n\n"
            )

            values[reduction]["FCI"].append(expected_energy)
            values[reduction]["DFT"].append(e_initial)
            values[reduction]["WF-in-DFT"].append(e_wf_emb)
            # TODO fix VQE bit
            #values[reduction]["VQE-in-DFT"].append(e_vqe_emb)

    if save:
        for reduction in mo_reduction:
            # Make a dataframe and write to file
            df = pd.DataFrame(data=values[reduction], index=distances)
            datapath = (
                Path(__file__).parent
                / f"data/{mol_name}-{reduction=}-{pd.Timestamp.now()}.csv"
            )
            df.to_csv(datapath, mode="w")


    if plot:
        plot_methods(distances, values, mol_name)

        # can use this when vqe works...
        #plot_vqe_reductions(distances, values, mol_name, mo_reduction)


if __name__ == "__main__":
    args = parse()
    main(mo_reduction=[0], plot=True, save=False, dmin=1, dmax=5, max_runs=200)
