"""Helper functions for the package."""

import argparse
import logging
from logging.config import dictConfig
from pathlib import Path

import yaml
from openfermion import count_qubits

from .driver import NbedDriver
from .ham_converter import HamiltonianConverter

logger = logging.getLogger(__name__)


def setup_logs() -> None:
    """Initialise logging."""
    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "%(asctime)s: %(name)s: %(levelname)s: %(message)s"},
        },
        "handlers": {
            "file_handler": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filename": Path("../../nbed.log"),
                "encoding": "utf-8",
            },
            "stream_handler": {
                "class": "logging.StreamHandler",
                "level": "WARNING",
                "formatter": "standard",
            },
        },
    }

    dictConfig(config_dict)


def parse():
    """Parse arguments from command line interface."""
    parser = argparse.ArgumentParser(description="Output embedded Qubit Hamiltonian.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a config file. Overwrites other arguments.",
    )
    parser.add_argument(
        "--geometry",
        "-g",
        type=str,
        help="Path to an XYZ file.",
    )
    parser.add_argument(
        "--active_atoms",
        "-a",
        type=int,
        help="Number of atoms to include in active region.",
    )
    parser.add_argument(
        "--basis",
        "-b",
        type=str,
        help="Basis set to use.",
    )
    parser.add_argument(
        "--xc_functional",
        "--xc",
        "-x",
        type=str,
        help="Exchange correlation functional to use in DFT calculations.",
    )
    parser.add_argument(
        "--localization",
        "--loc",
        "-l",
        type=str.lower,
        choices=[
            "spade",
            "pipek-mezey",
            "ibo",
            "boys",
        ],
        help="Method of localization to use.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str.lower,
        choices=["openfermion", "qiskit", "pennylane"],
        help="Quantum computing backend to output the qubit hamiltonian for.",
    )
    parser.add_argument(
        "--convergence",
        "--conv",
        "-c",
        type=float,
        help="Convergence tolerance for calculations.",
    )
    parser.add_argument(
        "--savefile",
        "-s",
        type=str,
        help="Path to save file.",
    )
    parser.add_argument(
        "--ccsd",
        action="store_true",
        help="Include if you want to run a ccsd calculation of the whole system.",
    )
    args = parser.parse_args()

    if args.config:
        logger.debug("Reading config file.")
        filepath = Path(args.config).absolute()
        stream = open(filepath, "r")
        args = yaml.safe_load(stream)["nbed"]

        # Optional argument defaults
        args["ccsd"] = args.get("ccsd", False)
    else:
        # Transform the namespace object to a dict.
        args = vars(args)

    if any([values is None for values in args.values()]):
        logger.info(
            f"Missing values for argument {[key for key, value in args.items() if value is None]}"
        )
        print("\nMissing values for arguments: ".upper())
        print(f"{[key for key, value in args.items() if value is None]}\n")
        raise Exception("Missing argument values.")

    args["savefile"] = str(Path(args["savefile"]).absolute())
    args["geometry"] = str(Path(args["geometry"]).absolute())
    args["convergence"] = float(args["convergence"])

    logger.debug(f"Arguments: {args}")
    return args


def load_hamiltonian(filepath: Path, output: str) -> object:
    """Create a Hamiltonian from a file.

    Reads the input file and converts to the desired output format.
    """
    return HamiltonianConverter(filepath).convert(output)


def print_summary(driver: NbedDriver, fci: bool = False):
    """Print a summary of the package results.

    Args:
        driver (NbedDriver): An NbedDriver to summarise.
        fci (bool): Whether to run full system fci.
    """
    if driver.molecular_ham is None:
        logger.error(
            "Driver does not have molecular hamiltonian. Cannot print summary."
        )
        print("Driver does not have molecular hamiltonian. Cannot print summary.")
        return

    print("".center(80, "*"))
    print("  Summary of Embedded Calculation".center(80))
    print("".center(80, "*"))

    print(f"global (cheap) DFT calculation {driver._global_rks}")

    if driver.projector in ["huzinaga", "both"]:
        print("".center(80, "*"))
        print("  Huzinaga calculation".center(20))
        print(f"Total energy - active system at RHF level: {driver.emb_rhf_etot_HUZ}")
        if driver.run_ccsd_emb is True:
            print(
                f"Total energy - active system at CCSD level: {driver.e_wf_ccsd_emb_HUZ}"
            )
        if driver.run_fci_emb is True:
            print(
                f"Total energy - active system at FCI level: {driver.e_wf_fci_emb_HUZ}"
            )

        print(
            f"length of huzinaga embedded fermionic Hamiltonian: {len(list(driver.molecular_ham_HUZ))}"
        )
        print(f"number of qubits required: {count_qubits(driver._huzinaga_ham)}")

    if driver.projector in ["mu", "both"]:
        print("".center(80, "*"))
        print("  Mu shift calculation".center(20))
        print(
            f"Total energy - active system at RHF level: {driver.emb_rhf_etot_mu_shift}"
        )
        if driver.run_ccsd_emb is True:
            print(
                f"Total energy - active system at CCSD level: {driver.e_wf_ccsd_emb_MU}"
            )
        if driver.run_fci_emb is True:
            print(
                f"Total energy - active system at FCI level: {driver.e_wf_fci_emb_MU}"
            )

        print(
            f"length of mu embedded fermionic Hamiltonian: {len(list(driver.molecular_ham_MU))}"
        )
        print(f"number of qubits required: {count_qubits(driver._mu_ham)}")

    print("".center(80, "*"))
    print("  Summary of reference Calculation".center(80))
    print("".center(80, "*"))

    if fci:
        print(f"global (expensive) full FCI calculation {driver._global_fci.e_tot}")
    print(
        f"length of full system fermionic Hamiltonian: {len(list(driver.full_system_hamiltonian))}"
    )
    print(f"number of qubits required: {driver.n_qubits_full_system}")
