"""Helper functions for the package."""

import argparse
import logging
from logging.config import dictConfig
from pathlib import Path

import yaml

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
                "filename": Path("../../vqe-in-dft.log"),
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
        "--config", type=str, help="Path to a config file. Overwrites other arguments."
    )
    parser.add_argument(
        "--savefile",
        "--save",
        type=str,
        help="Path to save file.",
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
        "--qubits",
        "-q",
        type=int,
        help="Maximum number of qubits to be used in Qubit Hamiltonian.",
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
        type=str.lower,
        choices=["openfermion", "qiskit", "pennylane"],
        help="Quantum computing backend to output the qubit hamiltonian for.",
    )
    parser.add_argument(
        "--localisation",
        "--loc",
        type=str.lower,
        choices=["spade"],  # TODO "mullikan", "ibo", "boys",],
        help="Method of localisation to use.",
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
