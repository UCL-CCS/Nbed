"""Main embedding functionality."""

import logging

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


def cli() -> None:
    """CLI Interface."""
    setup_logs()
    args = parse()
    driver = NbedDriver(
        geometry=args["geometry"],
        n_active_atoms=args["active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        output=args["output"],
        projector=args["projector"],
        localisation=args["localisation"],
        convergence=args["convergence"],
        qubits=args["qubits"],
        savefile=args["savefile"],
    )
    converter = HamiltonianConverter(driver.molecular_ham, transform=args["transform"])
    qham = getattr(converter, args["output"])

    print("Qubit Hamiltonian:")
    print(qham)
    print(f"Classical Energy (Ha): {driver.classical_energy}")


if __name__ == "__main__":
    cli()
