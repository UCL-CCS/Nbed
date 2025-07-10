"""Helper functions for the package."""

import argparse
import json
import logging
import logging.config
import os
from pathlib import Path
from typing import Optional

from openfermion import count_qubits
from openfermion.chem.pubchem import geometry_from_pubchem
from pydantic import ValidationError

from nbed.config import NbedConfig, ProjectorEnum
from nbed.ham_builder import HamiltonianBuilder

from .driver import NbedDriver

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
                "filename": ".nbed.log",
                "mode": "w",
                "encoding": "utf-8",
            },
            "stream_handler": {
                "class": "logging.StreamHandler",
                "level": "WARNING",
                "formatter": "standard",
            },
        },
        "loggers": {
            "": {"handlers": ["file_handler", "stream_handler"], "level": "DEBUG"}
        },
    }

    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(__name__)
    logger.debug("Logging initialised.")


def parse():
    """Parse arguments from command line interface."""
    logger.debug("Adding CLI arguments.")
    parser = argparse.ArgumentParser(description="Output embedded Qubit Hamiltonian.")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to a config file. Overwrites other arguments.",
    )
    parser.add_argument(
        "--transform",
        required=False,
        type=str,
        help="Fermion to Qubit encoding to use e.g.'Jordan-Wigner', 'Bravyi-Kitaev' "
    )
    parser.add_argument(
        "--savefile",
        "-s",
        type=str,
        help="Path to save file.",
    )
    logger.debug("Parsing CLI arguments.")
    args = parser.parse_args()

    logger.debug("Reading config file.")
    filepath = Path(args.config).absolute()
    with open(filepath) as f:
        config_data = json.load(f)
    logger.debug(f"Input data:\n{config_data=}")

    try:
        config = NbedConfig(config_data)
    except ValidationError as e:
        logger.error("Could not validate input data against NbedConfig model.")
        logger.error(e)

    return config, args["transform"], args["savefile"]


def print_summary(driver: NbedDriver, transform: str, fci: bool = False) -> None:
    """Print a summary of the package results.

    Args:
        driver (NbedDriver): An NbedDriver to summarise.
        transform (str): The transform used to generate a qubit Hamiltonian.
        fci (bool): Whether to run full system fci.
    """
    logger.debug("Printing summary of results.")
    # for get statements
    default = "Not calculated."

    qham = HamiltonianBuilder(
        driver.embedded_scf,
        constant_e_shift=driver.classical_energy,
        transform=transform,
    ).build()

    # Would be a great place for a switch statemet when
    # dependencies catch up with python 3.10
    match driver.config.projector:
        case ProjectorEnum.BOTH:
            mu_qham, huz_qham = qham
        case ProjectorEnum.HUZ:
            mu_qham, huz_qham = None, qham
        case ProjectorEnum.MU:
            mu_qham, huz_qham = qham, None

    print("".center(80, "*"))
    logger.info("".center(80, "*"))
    print("  Summary of Embedded Calculation".center(80))
    logger.info("  Summary of Embedded Calculation".center(80))
    print("".center(80, "*"))
    logger.info("".center(80, "*"))

    print(f"global (cheap) DFT calculation {driver._global_ks.e_tot}")
    logger.info(f"global (cheap) DFT calculation {driver._global_ks.e_tot}")

    if driver.config.projector in [ProjectorEnum.HUZ, ProjectorEnum.BOTH]:
        print("".center(80, "*"))
        logger.info("".center(80, "*"))
        print("  Huzinaga calculation".center(20))
        logger.info("  Huzinaga calculation".center(20))
        print(
            f"Total energy - active system at RHF level: {driver._huzinaga.get('e_rhf', default)}"
        )
        logger.info(
            f"Total energy - active system at RHF level: {driver._huzinaga.get('e_rhf', default)}"
        )
        if driver.config.run_ccsd_emb is True:
            print(
                f"Total energy - active system at CCSD level: {driver._huzinaga.get('e_ccsd', default)}"
            )
            logger.info(
                f"Total energy - active system at CCSD level: {driver._huzinaga.get('e_ccsd', default)}"
            )
        if driver.config.run_fci_emb is True:
            print(
                f"Total energy - active system at FCI level: {driver._huzinaga.get('e_fci', default)}"
            )
            logger.info(
                f"Total energy - active system at FCI level: {driver._huzinaga.get('e_fci', default)}"
            )

        print(
            f"length of huzinaga embedded fermionic Hamiltonian: {len(huz_qham.terms)}"
        )
        logger.info(
            f"length of huzinaga embedded fermionic Hamiltonian: {len(huz_qham.terms)}"
        )
        print(f"number of qubits required: {count_qubits(huz_qham)}")
        logger.info(f"number of qubits required: {count_qubits(huz_qham)}")

    if driver.config.projector in [ProjectorEnum.MU, ProjectorEnum.BOTH]:
        print("".center(80, "*"))
        logger.info("".center(80, "*"))
        print("  Mu shift calculation".center(20))
        logger.info("  Mu shift calculation".center(20))
        print(
            f"Total energy - active system at RHF level: {driver._mu.get('e_rhf', default)}"
        )
        logger.info(
            f"Total energy - active system at RHF level: {driver._mu.get('e_rhf', default)}"
        )
        if driver.config.run_ccsd_emb is True:
            print(
                f"Total energy - active system at CCSD level: {driver._mu.get('e_ccsd', default)}"
            )
            logger.info(
                f"Total energy - active system at CCSD level: {driver._mu.get('e_ccsd', default)}"
            )
        if driver.config.run_fci_emb is True:
            print(
                f"Total energy - active system at FCI level: {driver._mu.get('e_fci', default)}"
            )
            logger.info(
                f"Total energy - active system at FCI level: {driver._mu.get('e_fci', default)}"
            )

        print(f"length of mu embedded fermionic Hamiltonian: {len(mu_qham.terms)}")
        logger.info(
            f"length of mu embedded fermionic Hamiltonian: {len(mu_qham.terms)}"
        )
        print(f"number of qubits required: {count_qubits(mu_qham)}")
        logger.info(f"number of qubits required: {count_qubits(mu_qham)}")

    logger.debug("Building full system Hamiltonian for comparison.")
    full_system_hamiltonian = HamiltonianBuilder(
        driver._global_hf, constant_e_shift=0, transform=transform
    ).build()

    print("".center(80, "*"))
    logger.info("".center(80, "*"))
    print("  Summary of reference Calculation".center(80))
    logger.info("  Summary of reference Calculation".center(80))
    print("".center(80, "*"))
    logger.info("".center(80, "*"))

    if fci:
        print("Running Full system FCI and preparing Hamiltonian.")
        logger.info("Running Full system FCI and preparing Hamiltonian.")
        print(f"Global (expensive) full FCI calculation {driver._global_fci.e_tot}")
        logger.info(
            f"Global (expensive) full FCI calculation {driver._global_fci.e_tot}"
        )

    print(
        f"length of full system fermionic Hamiltonian: {len(full_system_hamiltonian.terms)}"
    )

    logger.info(
        f"length of full system fermionic Hamiltonian: {len(full_system_hamiltonian.terms)}"
    )
    print(f"number of qubits required: {count_qubits(full_system_hamiltonian)}")
    logger.info(f"number of qubits required: {count_qubits(full_system_hamiltonian)}")


def pubchem_mol_geometry(molecule_name) -> dict:
    """Wrapper of Openfermion function to extract geometry using the molecule's name from the PubChem.

    Returns a dictionary of atomic type and xyz location, each indexed by dictionary key.

    Args:
        molecule_name (str): Name of molecule to search on pubchem
    Returns:
        struct_dict (dict): Keys index atoms and values contain Tuple of ('atom_id', (x_loc, y_loc, z_loc)

    Example:
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

    Example:
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
            "active atom indices not subset of indices in structural dict "
        )

    ordering = (
        *active_atom_inds,
        *[ind for ind in struct_dict.keys() if ind not in active_atom_inds],
    )

    n_atoms = len(struct_dict)
    xyz_file = f"{n_atoms}"
    xyz_file += "\n \n"
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
    """Saves .xyz file in a molecular_structures directory.

    This function orders the atoms in struct_dict according to the ordering
    given in atom_ordering_by_inds list. The file is then saved.
    The location of this director is either at save_location, or if not defined then in current working dir.
    Function returns the path to xyz file.

    Args:
        file_name (str): Name of xyz file
        struct_dict (dict): Dictionary of indexed atoms and Cartesian coordinates (x,y,z)
        active_atom_inds (list): list of indices to be considered active. This will put these atoms to the top of the xyz file.
                                 Note indices are chosen from the struct_dict.
        save_location (Path): Path of where to save xyz file. If not defined then current working dir used.

    Returns:
        xyz_file_path (Path): Path to xyz file

    Example:
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
        save_location = Path(os.getcwd())

    output_dir = os.path.join(str(save_location), "molecular_structures")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xyz_file_path = os.path.join(output_dir, f"{file_name}.xyz")

    with open(xyz_file_path, "w") as outfile:
        outfile.write(xyz_string)

    return Path(xyz_file_path)
