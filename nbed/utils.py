"""Helper functions for the package."""

import argparse
import logging
import os
from logging.config import dictConfig
from pathlib import Path
from typing import Optional

import yaml
from openfermion import count_qubits, transforms
from openfermion.chem.pubchem import geometry_from_pubchem

from nbed.ham_builder import HamiltonianBuilder

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


def restricted_float_percentage(x: float) -> float:
    """Checks input x is within 0-1 range (percentage) and is a float

    Args:
        x (float): input number between 0 and 1 (inclusive)

    Returns:
        x (float): input percentage
    """
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def parse():
    """Parse arguments from command line interface."""
    parser = argparse.ArgumentParser(description="Output embedded Qubit Hamiltonian.")
    parser.add_argument(
        "--config", type=str, help="Path to a config file. Overwrites other arguments.",
    )
    parser.add_argument(
        "--geometry",
        "-g",
        type=str,
        help="Path to an XYZ file or raw xyz string of molecular structure (note active atoms must appear first).",
    )
    parser.add_argument(
        "--active_atoms",
        "-a",
        type=int,
        help="Number of atoms to include in active region.",
    )
    parser.add_argument(
        "--basis", "-b", type=str, help="Basis set to use.",
    )
    parser.add_argument(
        "--xc_functional",
        "--xc",
        "-x",
        type=str,
        help="Exchange correlation functional to use in DFT calculations.",
    )
    parser.add_argument(
        "--projector",
        "-p",
        type=str,
        choices=["huzinaga", "mu",],
        help="Which projector method to use.",
    )
    parser.add_argument(
        "--unit",
        "-u",
        type=str,
        choices=["angstrom", "bohr"],
        help="Distance unit of molecular geometry",
    )
    parser.add_argument(
        "--localization",
        "--loc",
        "-l",
        type=str.lower,
        choices=["spade", "pipek-mezey", "ibo", "boys",],
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
        "--charge", type=int, help="Charge of molecular system.",
    )
    parser.add_argument(
        "--savefile", "-s", type=str, help="Path to save file.",
    )
    parser.add_argument(
        "--run_ccsd_emb",
        action="store_true",
        help="Include if you want to run a ccsd calculation of the active embedded system.",
    )
    parser.add_argument(
        "--run_fci_emb",
        action="store_true",
        help="Include if you want to run a fci calculation of the active embedded system.",
    )
    parser.add_argument(
        "--ram", type=str, help="amount of ram in MB that PySCF can use",
    )
    parser.add_argument(
        "--mu_shift", type=int, help="mu energy shift value",
    )
    parser.add_argument(
        "--virtual_localization",
        "--virt_loc",
        action="store_true",
        help="whether to run localization of virutal (unoccupied) orbitals",
    )
    parser.add_argument(
        "--occupied_threshold",
        "--ot",
        type=restricted_float_percentage,
        help="occupation threshold (float between 0 and 1 inclusive) used to localize occupied molecular orbs (unnecessary for spade approach)",
    )
    parser.add_argument(
        "--virtual_threshold",
        "--vt",
        type=restricted_float_percentage,
        help="threshold (float between 0 and 1 inclusive) used to localize unoccupied (virtual) molecular orbs (necessary for spade approach)",
    )
    parser.add_argument(
        "--max_hf_cycles",
        "--hf_c",
        type=int,
        help="max number of Hartree-Fock iterations allowed for global and local Hartree-Fock calcs",
    )
    args = parser.parse_args()

    if args.config:
        logger.debug("Reading config file.")
        filepath = Path(args.config).absolute()
        stream = open(filepath, "r")
        args = yaml.safe_load(stream)["nbed"]

        # Optional argument defaults
        args["unit"] = args.get("unit", "angstrom")
        args["charge"] = args.get("charge", 0)
        args["convergence"] = args.get("convergence", 1e-6)
        args["run_ccsd_emb"] = args.get("run_ccsd_emb", False)
        args["run_fci_emb"] = args.get("run_fci_emb", False)
        args["mu_shift"] = args.get("mu_shift", 1e6)
        args["ram"] = args.get("ram", 4_000)
        args["virtual_localization"] = args.get("virtual_localization", False)
        args["occupied_threshold"] = args.get("occupied_threshold", 0.95)
        args["virtual_threshold"] = args.get("virtual_threshold", 0.95)
        args["max_hf_cycles"] = args.get("max_hf_cycles", 50)
    else:
        # Transform the namespace object to a dict.
        args = vars(args)

    if any([values is None for values in args.values()]):
        logger.info(
            f"Missing values for argument {[key for key, value in args.items() if value is None]}"
        )
        logger.info("\nMissing values for arguments: ".upper())
        logger.info(f"{[key for key, value in args.items() if value is None]}\n")
        raise Exception("Missing argument values.")

    args["savefile"] = str(Path(args["savefile"]).absolute())
    args["convergence"] = float(args["convergence"])

    logger.debug(f"Arguments: {args}")
    return args


def load_hamiltonian(filepath: Path, output: str) -> object:
    """Create a Hamiltonian from a file.

    Reads the input file and converts to the desired output format.
    """
    return HamiltonianConverter(filepath).convert(output)


def print_summary(driver: NbedDriver, transform: str, fci: bool = False):
    """Print a summary of the package results.

    Args:
        driver (NbedDriver): An NbedDriver to summarise.
        fci (bool): Whether to run full system fci.
    """
    if driver.molecular_ham is None:
        logger.error(
            "Driver does not have molecular hamiltonian. Cannot print summary."
        )
        logger.info("Driver does not have molecular hamiltonian. Cannot print summary.")
        return

    logger.info("".center(80, "*"))
    logger.info("  Summary of Embedded Calculation".center(80))
    logger.info("".center(80, "*"))

    logger.info(f"global (cheap) DFT calculation {driver._global_rks.e_tot}")

    if driver.projector in ["huzinaga", "both"]:
        logger.info("".center(80, "*"))
        logger.info("  Huzinaga calculation".center(20))
        logger.info(
            f"Total energy - active system at RHF level: {driver._huzinaga['e_rhf']}"
        )
        if driver.run_ccsd_emb is True:
            logger.info(
                f"Total energy - active system at CCSD level: {driver._huzinaga['e_ccsd']}"
            )
        if driver.run_fci_emb is True:
            logger.info(
                f"Total energy - active system at FCI level: {driver._huzinaga['e_fci']}"
            )

        logger.info(
            f"length of huzinaga embedded fermionic Hamiltonian: {len(list(driver._huzinaga['hamiltonian']))}"
        )
        logger.info(
            f"number of qubits required: {count_qubits(driver._huzinaga['hamiltonian'])}"
        )

    if driver.projector in ["mu", "both"]:
        logger.info("".center(80, "*"))
        logger.info("  Mu shift calculation".center(20))
        logger.info(f"Total energy - active system at RHF level: {driver._mu['e_rhf']}")
        if driver.run_ccsd_emb is True:
            logger.info(
                f"Total energy - active system at CCSD level: {driver._mu['e_ccsd']}"
            )
        if driver.run_fci_emb is True:
            logger.info(
                f"Total energy - active system at FCI level: {driver._mu['e_fci']}"
            )

        logger.info(
            f"length of mu embedded fermionic Hamiltonian: {len(list(driver._mu['hamiltonian']))}"
        )
        logger.info(
            f"number of qubits required: {count_qubits(driver._mu['hamiltonian'])}"
        )

    logger.info("".center(80, "*"))
    logger.info("  Summary of reference Calculation".center(80))
    logger.info("".center(80, "*"))

    if fci:
        logger.info(
            f"global (expensive) full FCI calculation {driver._global_fci.e_tot}"
        )
    full_system_hamiltonian = HamiltonianBuilder(
        driver._global_hf, constant_e_shift=0, transform=transform
    )
    logger.info(
        f"length of full system fermionic Hamiltonian: {len(list(full_system_hamiltonian))}"
    )
    logger.info(f"number of qubits required: {count_qubits(full_system_hamiltonian)}")


def pubchem_mol_geometry(molecule_name) -> dict:
    """Wrapper of Openfermion function to extract geometry using the molecule's name from the PubChem.
    Returns a dictionary of atomic type and xyz location, each indexed by dictionary key

    Args:
        molecule_name (str): Name of molecule to search on pubchem
    Returns:
        struct_dict (dict): Keys index atoms and values contain Tuple of ('atom_id', (x_loc, y_loc, z_loc)

    Example

    output = pubchem_mol_geometry('H2O')
    print(output)

    >> { 0: ('O', (0, 0, 0)),
         1: ('H', (0.2774, 0.8929, 0.2544)),
         2: ('H', (0.6068, -0.2383, -0.7169))
         }

    """
    geometry_pubchem = geometry_from_pubchem(molecule_name, structure="3d")

    if geometry_pubchem is None:
        raise ValueError(
            f"""Could not find geometry of {molecule_name} on PubChem...
                                 make sure molecule input is a correct path to an xyz file or real molecule
                                """
        )

    struct_dict = {}
    for ind, atom_xyz in enumerate(geometry_pubchem):
        struct_dict[ind] = atom_xyz
    return struct_dict


def build_ordered_xyz_string(struct_dict: dict, active_atom_inds: list) -> str:
    """Get raw xyz string of molecular geometry.

    This function orders the atoms in struct_dict according to the ordering given in atom_ordering_by_inds list.

    Args:
        struct_dict (dict): Dictionary of indexed atoms and Cartesian coordinates (x,y,z)
        active_atom_inds (list): list of indices to be considered active. This will put these atoms to the top of the xyz file.
                                 Note indices are chosen from the struct_dict.
    Returns:
        xyz_string (str): raw xyz string of molecular geometry (atoms ordered by atom_ordering_by_inds list)

        Example

        input_struct_dict = { 0: ('O', (0, 0, 0)),
                              1: ('H', (0.2774, 0.8929, 0.2544)),
                              2: ('H', (0.6068, -0.2383, -0.7169))
                            }

        xyz_string = ordered_xyz_string('water', input_struct_dict, [1,0,2])
        print(xyz_string)

         >> 3

            H	0.2774	0.8929	0.2544
            O	0	0	0
            H	0.6068	-0.2383	-0.7169

    """
    if not set(active_atom_inds).issubset(set(list(struct_dict.keys()))):
        raise ValueError(
            f"active atom indices not subset of indices in structural dict "
        )

    ordering = (
        *active_atom_inds,
        *[ind for ind in struct_dict.keys() if ind not in active_atom_inds],
    )

    n_atoms = len(struct_dict)
    xyz_file = f"{n_atoms}"
    xyz_file += f"\n \n"
    for atom_ind in ordering:
        atom, xyz = struct_dict[atom_ind]
        xyz_file += f"{atom}\t{xyz[0]}\t{xyz[1]}\t{xyz[2]}\n"

    return xyz_file


def save_ordered_xyz_file(
    file_name: str,
    struct_dict: dict,
    active_atom_inds: list,
    save_location: Optional[Path] = None,
) -> Path:
    """Saves .xyz file in a molecular_structures directory. The location of this director is either at save_location,
    or if not defined then in current working dir. Function returns the path to xyz file.

    This function orders the atoms in struct_dict according to the ordering given in atom_ordering_by_inds list.

    Args:
        file_name (str): Name of xyz file
        struct_dict (dict): Dictionary of indexed atoms and Cartesian coordinates (x,y,z)
        struct_dict (dict): Dictionary of indexed atoms and Cartesian coordinates (x,y,z)
        active_atom_inds (list): list of indices to be considered active. This will put these atoms to the top of the xyz file.
                                 Note indices are chosen from the struct_dict.
        save_location (Path): Path of where to save xyz file. If not defined then current working dir used.
    Returns:
        xyz_file_path (Path): Path to xyz file

        Example

        input_struct_dict = { 0: ('O', (0, 0, 0)),
                              1: ('H', (0.2774, 0.8929, 0.2544)),
                              2: ('H', (0.6068, -0.2383, -0.7169))
                            }

        path = save_ordered_xyz_file('water', input_struct_dict, [1,0,2])
        print(path)
        >> ../molecular_structures/water.xyz

        with open(path,'r') as infile:
            xyz_string = infile.read()
        print(xyz_string)

         >> 3

            H	0.2774	0.8929	0.2544
            O	0	0	0
            H	0.6068	-0.2383	-0.7169

    """
    xyz_string = build_ordered_xyz_string(struct_dict, active_atom_inds)

    if save_location is None:
        save_location = os.getcwd()

    output_dir = os.path.join(save_location, "molecular_structures")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xyz_file_path = os.path.join(output_dir, f"{file_name}.xyz")

    with open(xyz_file_path, "w") as outfile:
        outfile.write(xyz_string)

    return xyz_file_path
