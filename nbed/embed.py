"""Main embedding functionality."""

import logging
from datetime import datetime
from os import mkdir
from pathlib import Path
from typing import Optional

from numpy import save
from openfermion.utils import save_operator

from nbed.exceptions import NbedConfigError
from nbed.ham_builder import HamiltonianBuilder

from .driver import NbedDriver
from .ham_converter import HamiltonianConverter
from .utils import parse, print_summary

logger = logging.getLogger(__name__)


def nbed(
    geometry: str,
    n_active_atoms: int,
    basis: str,
    xc_functional: str,
    projector: str,
    output: str,
    transform: str,
    qubits: Optional[int] = None,
    localization: Optional[str] = "spade",
    convergence: Optional[float] = 1e-6,
    charge: Optional[int] = 0,
    spin: Optional[int] = 0,
    mu_level_shift: Optional[float] = 1e6,
    run_ccsd_emb: Optional[bool] = False,
    run_fci_emb: Optional[bool] = False,
    max_ram_memory: Optional[int] = 4000,
    pyscf_print_level: int = 1,
    savefile: Optional[Path] = None,
    unit: Optional[str] = "angstrom",
    occupied_threshold: Optional[float] = 0.95,
    virtual_threshold: Optional[float] = 0.95,
    max_hf_cycles: int = 50,
    max_dft_cycles: int = 50,
    unrestricted: Optional[bool] = False,
):
    """Import interface for the nbed package.

    This functin first the NbedDriver class to create a second quantized hamiltonian
    using configuration provided. Then it calls the HamiltonianConverter class to
    apply a transformation to a qubit hamiltonian and output the desired backend object.

    Args:
        geometry (str): Path to .xyz file containing molecular geometry or raw xyz string.
        n_active_atoms (int): The number of atoms to include in the active region.
        basis (str): The name of an atomic orbital basis set to use for chemistry calculations.
        xc_functional (str): The name of an Exchange-Correlation functional to be used for DFT.
        projector (str): Type of projector to use in embedding. One of "mu" or "huzinaga".
        output (str): The name of the quantum backend to output a qubit hamiltonian object for.
        transform (str): Qubit transform to be applied to the Hamiltonian.
        localization (str): Orbital localization method to use. One of 'spade', 'pipek-mezey', 'boys' or 'ibo'.
        convergence (float): The convergence tolerance for energy calculations.
        charge (int): Charge of molecule
        spin (int): Spin of the molecule
        mu_level_shift (float): Level shift parameter to use for mu-projector.
        run_ccsd_emb (bool): Whether or not to find the CCSD energy of embbeded system for reference.
        run_fci_emb (bool): Whether or not to find the FCI energy of embbeded system for reference.
        max_ram_memory (int): Amount of RAM memery in MB available for PySCF calculation
        pyscf_print_level (int): Amount of information PySCF prints
        savefile (str): Path to file to save output Hamiltonain to.
        qubits (int): The number of qubits available for the output hamiltonian.
        unit (str): molecular geometry unit 'angstrom' or 'bohr'
        occupied_threshold (float): The occupancy threshold for localizing occupied orbitals.
        virtual_threshold (float): The occupancy threshold for localizing virtual orbitals.
        max_hf_cycles (int): max number of Hartree-Fock iterations allowed (for global and local HFock)
        max_dft_cycles (int): max number of DFT iterations allowed in scf calc
        unrestricted (bool): Whether to force unrestricted calculation.

    Returns:
        object: A qubit hamiltonian object which can be used in the quantum backend specified by 'output'.
    """
    if projector == "both":
        raise NbedConfigError("Cannot use 'both' as value of projector.")

    driver = NbedDriver(
        geometry=geometry,
        n_active_atoms=n_active_atoms,
        basis=basis,
        xc_functional=xc_functional,
        projector=projector,
        localization=localization,
        convergence=convergence,
        charge=charge,
        spin=spin,
        mu_level_shift=mu_level_shift,
        run_ccsd_emb=run_ccsd_emb,
        run_fci_emb=run_fci_emb,
        max_ram_memory=max_ram_memory,
        pyscf_print_level=pyscf_print_level,
        unit=unit,
        occupied_threshold=occupied_threshold,
        virtual_threshold=virtual_threshold,
        max_hf_cycles=max_hf_cycles,
        max_dft_cycles=max_dft_cycles,
        force_unrestricted=unrestricted,
    )
    if savefile is not None:
        data_directory = Path(savefile).absolute()
        data_directory.mkdir(parents=True, exist_ok=True)
        data_directory = str(data_directory)

    # Needed for 'both' projector
    if isinstance(driver.embedded_scf, tuple):
        hamiltonians = ()
        for scf, e_classical in zip(driver.embedded_scf, driver.e_classical):
            qham = HamiltonianBuilder(
                scf_method=scf,
                constant_e_shift=driver.classical_energy,
                transform=transform,
            ).build(n_qubits=qubits)

            converter = HamiltonianConverter(qham)
            qham = getattr(converter, output.lower(), qham)

            if savefile is not None:
                # because we'll have two in quick succession
                file_name = f"Nbed_{datetime.now()}"
                save_operator(
                    qham,
                    file_name,
                    data_directory,
                )

            hamiltonians += (qham,)
    else:
        qham = HamiltonianBuilder(
            scf_method=driver.embedded_scf,
            constant_e_shift=driver.classical_energy,
            transform=transform,
        ).build(n_qubits=qubits)

        converter = HamiltonianConverter(qham)
        qham = getattr(converter, output.lower(), qham)
        hamiltonians = qham

    if savefile is not None:
        file_name = f"Nbed_{datetime.now()}"
        save_operator(
            qham,
            file_name,
            data_directory,
        )

    print_summary(driver, transform, fci=False)
    return hamiltonians


def cli() -> None:
    """CLI Interface."""
    args = parse()
    nbed(
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
        max_hf_cycles=args["max_hf_cycles"],
        max_dft_cycles=args["max_dft_cycles"],
    )


if __name__ == "__main__":
    cli()
