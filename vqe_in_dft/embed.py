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

from vqe_in_dft.ham_converter import HamiltonianConverter
from vqe_in_dft.localisation import boys, ibo, mullikan, spade
from vqe_in_dft.utils import parse, setup_logs

logger = logging.getLogger(__name__)
setup_logs()


def closed_shell_subsystem(
    scf_method: StreamObject, density: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Calculate the components of subsystem energy.

    Args:
        scf_method (StreamObject): A self consistent method from pyscf.
        density (np.ndarray): Density matrix for the subsystem.

    Returns:
        Tuple(float, float, np.ndarray, np.ndarray, np.ndarray)

    """
    # It seems that PySCF lumps J and K in the J array
    j: np.ndarray = scf_method.get_j(dm=density)
    k: np.ndarray = np.zeros(np.shape(j))
    two_e_term: np.ndarray = scf_method.get_veff(scf_method.mol, density)
    e_xc: float = two_e_term.exc
    v_xc: np.ndarray = two_e_term - j

    # Energy
    e: float = np.einsum("ij,ij", density, scf_method.get_hcore() + j / 2) + e_xc
    return e, e_xc, j, k, v_xc


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
    qubits: Optional[int] = None,
    savefile: Optional[Path] = None,
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

    q_ham = get_qubit_hamiltonian(embedded_scf, active_indices)

    converter = HamiltonianConverter(q_ham)
    q_ham = converter.convert(output)

    if savefile:
        converter.save(savefile)

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
        savefile=args["savefile"],
    )
    print("Qubit Hamiltonian:")
    print(qham)
    print(f"Classical Energy (Ha): {e_classical}")


if __name__ == "__main__":
    cli()
