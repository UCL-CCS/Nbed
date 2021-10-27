"""Main embedding functionality."""

import logging
from typing import Path, Optional

import numpy as np
from openfermion import QubitOperator
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import bravyi_kitaev, bravyi_kitaev_tree, jordan_wigner
from pyscf.dft import numint
from pyscf.dft.rks import get_veff as rks_get_veff
from pyscf.lib import StreamObject, tag_array

from .driver import NbedDriver
from .exceptions import NbedConfigError
from .ham_converter import HamiltonianConverter
from .utils import parse, setup_logs

logger = logging.getLogger(__name__)

def nbed(
    geometry: Path,
    n_active_atoms: int,
    basis: str,
    xc_functional: str,
    projector: str,
    output: str,
    transform: str,
    localisation: Optional[str] = "spade",
    convergence: Optional[float] = 1e-6,
    qubits: Optional[int] = None,
    charge: Optional[int] = 0,
    mu_level_shift: Optional[float] = 1e6,
    run_ccsd_emb: Optional[bool] = False,
    run_fci_emb: Optional[bool] = False,
    max_ram_memory: Optional[int] = 4000,
    pyscf_print_level: int = 1,
    savefile: Optional[Path] = None,
    ):
    """Import interface for the nbed package.
    
    This functin first the NbedDriver class to create a second quantized hamiltonian 
    using configuration provided. Then it calls the HamiltonianConverter class to
    apply a transformation to a qubit hamiltonian and output the desired backend object.

    Args:
        geometry (Path): A path to an .xyz file describing moleclar geometry.
        n_active_atoms (int): The number of atoms to include in the active region.
        basis (str): The name of an atomic orbital basis set to use for chemistry calculations.
        xc_functonal (str): The name of an Exchange-Correlation functional to be used for DFT.
        projector (str): Type of projector to use in embedding. One of "mu" or "huzinaga".
        output (str): The name of the quantum backend to output a qubit hamiltonian object for.
        localisation (str): Orbital Localisation method to use. One of 'spade', 'mullikan', 'boys' or 'ibo'.
        convergence (float): The convergence tolerance for energy calculations.
        charge (int): Charge of molecular species
        mu_level_shift (float): Level shift parameter to use for mu-projector.
        run_ccsd_emb (bool): Whether or not to find the CCSD energy of embbeded system for reference.
        run_fci_emb (bool): Whether or not to find the FCI energy of embbeded system for reference.
        max_ram_memory (int): Amount of RAM memery in MB available for PySCF calculation
        pyscf_print_level (int): Amount of information PySCF prints
        qubits (int): The number of qubits available for the output hamiltonian.

    Returns:
        object: A qubit hamiltonian object which can be used in the quantum backend specified by 'output'.
    """
    driver = NbedDriver(
        geometry=geometry,
        n_active_atoms=n_active_atoms,
        basis=basis,
        xc_functional=xc_functional,
        projector=projector,
        localisation=localisation,
        convergence=convergence,
        qubits=qubits,
        savefile=savefile,
        charge=charge,
        mu_level_shift=mu_level_shift,
        run_ccsd_emb=run_ccsd_emb,
        run_fci_emb=run_fci_emb,
        max_ram_memory=max_ram_memory,
        pyscf_print_level=pyscf_print_level,    
        )
    converter = HamiltonianConverter(driver.molecular_ham, transform=transform)
    qham = getattr(converter, output)

    print("Qubit Hamiltonian:")
    print(qham)
    print(f"Classical Energy (Ha): {driver.classical_energy}")

    return qham


def cli() -> None:
    """CLI Interface."""
    setup_logs()
    args = parse()
    qham = nbed(
        geometry=args["geometry"],
        n_active_atoms=args["active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        projector=args["projector"],
        localisation=args["localisation"],
        convergence=args["convergence"],
        qubits=args["qubits"],
        savefile=args["savefile"],
    )


if __name__ == "__main__":
    cli()
