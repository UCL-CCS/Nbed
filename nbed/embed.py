"""Main embedding functionality."""

import logging
from pathlib import Path
from typing import Optional

from .driver import NbedDriver
from .ham_converter import HamiltonianConverter
from .utils import parse, print_summary, setup_logs

logger = logging.getLogger(__name__)


def nbed(
    geometry: str,
    n_active_atoms: int,
    basis: str,
    xc_functional: str,
    projector: str,
    output: str,
    transform: str,
    localization: Optional[str] = "spade",
    convergence: Optional[float] = 1e-6,
    charge: Optional[int] = 0,
    mu_level_shift: Optional[float] = 1e6,
    run_ccsd_emb: Optional[bool] = False,
    run_fci_emb: Optional[bool] = False,
    max_ram_memory: Optional[int] = 4000,
    pyscf_print_level: int = 1,
    savefile: Optional[Path] = None,
    unit: Optional[str] = "angstrom",
    occupied_threshold: Optional[float] = 0.95,
    virtual_threshold: Optional[float] = 0.95,
):
    """Import interface for the nbed package.

    This functin first the NbedDriver class to create a second quantized hamiltonian
    using configuration provided. Then it calls the HamiltonianConverter class to
    apply a transformation to a qubit hamiltonian and output the desired backend object.

    Args:
        geometry (str): Path to .xyz file containing molecular geometry or raw xyz string.
        n_active_atoms (int): The number of atoms to include in the active region.
        basis (str): The name of an atomic orbital basis set to use for chemistry calculations.
        xc_functonal (str): The name of an Exchange-Correlation functional to be used for DFT.
        projector (str): Type of projector to use in embedding. One of "mu" or "huzinaga".
        output (str): The name of the quantum backend to output a qubit hamiltonian object for.
        localization (str): Orbital localization method to use. One of 'spade', 'pipek-mezey', 'boys' or 'ibo'.
        convergence (float): The convergence tolerance for energy calculations.
        charge (int): Charge of molecular species
        mu_level_shift (float): Level shift parameter to use for mu-projector.
        run_ccsd_emb (bool): Whether or not to find the CCSD energy of embbeded system for reference.
        run_fci_emb (bool): Whether or not to find the FCI energy of embbeded system for reference.
        max_ram_memory (int): Amount of RAM memery in MB available for PySCF calculation
        pyscf_print_level (int): Amount of information PySCF prints
        qubits (int): The number of qubits available for the output hamiltonian.
        unit (str): molecular geometry unit 'angstrom' or 'bohr'

    Returns:
        object: A qubit hamiltonian object which can be used in the quantum backend specified by 'output'.
    """
    driver = NbedDriver(
        geometry=geometry,
        n_active_atoms=n_active_atoms,
        basis=basis,
        xc_functional=xc_functional,
        projector=projector,
        localization=localization,
        convergence=convergence,
        savefile=savefile,
        charge=charge,
        mu_level_shift=mu_level_shift,
        run_ccsd_emb=run_ccsd_emb,
        run_fci_emb=run_fci_emb,
        max_ram_memory=max_ram_memory,
        pyscf_print_level=pyscf_print_level,
        unit=unit,
        occupied_threshold=occupied_threshold,
        virtual_threshold=virtual_threshold,
    )
    converter = HamiltonianConverter(driver.molecular_ham, transform=transform)
    qham = getattr(converter, output)
    print_summary(driver, fci=True)

    from openfermion import eigenspectrum

    logger.info(eigenspectrum(driver.molecular_ham)[0])

    return qham


def cli() -> None:
    """CLI Interface."""
    setup_logs()
    args = parse()
    qham = nbed(
        geometry=args["geometry"],
        n_active_atoms=args["n_active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        projector=args["projector"],
        localization=args["localization"],
        transform=args["transform"],
        output=args["output"],
        convergence=args["convergence"],
        savefile=args["savefile"],
        run_ccsd_emb=args["run_ccsd_emb"],
        run_fci_emb=args["run_ccsd_emb"],
        unit=args["unit"],
        occupied_threshold=args["occupied_threshold"],
        virtual_threshold=args["virtual_threshold"],
    )


if __name__ == "__main__":
    cli()
