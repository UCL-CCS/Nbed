"""
File to contain tests of the driver.py script.
"""
import logging
from pathlib import Path
from typing import Optional

import pytest
from numpy import isclose, number
from pyscf.lib.misc import StreamObject

from nbed.driver import NbedDriver
from nbed.exceptions import NbedConfigError

logger = logging.getLogger(__name__)


water_filepath = Path("tests/molecules/water.xyz").absolute()


class UnrestrictedDriver(NbedDriver):
    """Force use of unrestricted SCF for driver."""

    def __init__(
        self,
        geometry: str,
        n_active_atoms: int,
        basis: str,
        xc_functional: str,
        projector: str,
        localization: Optional[str] = "spade",
        convergence: Optional[float] = 1e-6,
        charge: Optional[int] = 0,
        spin: Optional[int] = 0,
        mu_level_shift: Optional[float] = 1e6,
        run_ccsd_emb: Optional[bool] = False,
        run_fci_emb: Optional[bool] = False,
        run_virtual_localization: Optional[bool] = False,
        run_dft_in_dft: Optional[bool] = False,
        max_ram_memory: Optional[int] = 4000,
        pyscf_print_level: int = 1,
        savefile: Optional[Path] = None,
        unit: Optional[str] = "angstrom",
        occupied_threshold: Optional[float] = 0.95,
        virtual_threshold: Optional[float] = 0.95,
        init_huzinaga_rhf_with_mu: bool = False,
        max_hf_cycles: int = 50,
        max_dft_cycles: int = 50,
    ):
        """Initialise class."""
        logger.debug("Initialising driver.")
        config_valid = True
        if projector not in ["mu", "huzinaga", "both"]:
            logger.error(
                "Invalid projector %s selected. Choose from 'mu' or 'huzinzaga'.",
                projector,
            )
            config_valid = False

        if localization not in ["spade", "ibo", "boys", "pipek-mezey"]:
            logger.error(
                "Invalid localization method %s. Choose from 'ibo','boys','pipek-mezey' or 'spade'.",
                localization,
            )
            config_valid = False

        if not config_valid:
            logger.error("Invalid config.")
            raise NbedConfigError("Invalid config.")

        self.geometry = geometry
        self.n_active_atoms = n_active_atoms
        self.basis = basis.lower()
        self.xc_functional = xc_functional.lower()
        self.projector = projector.lower()
        self.localization = localization.lower()
        self.convergence = convergence
        self.charge = charge
        self.spin = spin
        self.mu_level_shift = mu_level_shift
        self.run_ccsd_emb = run_ccsd_emb
        self.run_fci_emb = run_fci_emb
        self.run_virtual_localization = run_virtual_localization
        self.run_dft_in_dft = run_dft_in_dft
        self.max_ram_memory = max_ram_memory
        self.pyscf_print_level = pyscf_print_level
        self.savefile = savefile
        self.unit = unit
        self.occupied_threshold = occupied_threshold
        self.virtual_threshold = virtual_threshold
        self.max_hf_cycles = max_hf_cycles
        self.max_dft_cycles = max_dft_cycles

        self._check_active_atoms()
        self.localized_system = None
        self.two_e_cross = None
        self._dft_potential = None

        self._restricted_scf = False

        self.embed(
            init_huzinaga_rhf_with_mu=init_huzinaga_rhf_with_mu
        )  # TODO uncomment.
        logger.debug("Driver initialisation complete.")


def test_incorrect_geometry_path() -> None:
    """test to make sure that FileNotFoundError is thrown if invalid path to xyz geometry file is given"""

    molecule = "THIS/IS/NOT/AN/XYZ/FILE"

    args = {
        "geometry": molecule,
        "n_active_atoms": 1,
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "convergence": 1e-6,
        "savefile": None,
        "run_ccsd_emb": True,
        "run_fci_emb": True,
    }

    with pytest.raises(RuntimeError, match="Unsupported atom symbol .*"):
        # match will match with any printed error message
        NbedDriver(
            geometry=args["geometry"],
            n_active_atoms=args["n_active_atoms"],
            basis=args["basis"],
            xc_functional=args["xc_functional"],
            projector=args["projector"],
            localization=args["localization"],
            convergence=args["convergence"],
            savefile=args["savefile"],
            run_ccsd_emb=args["run_ccsd_emb"],
            run_fci_emb=args["run_fci_emb"],
        )


def test_driver_standard_xyz_file_input() -> None:
    """test to check driver works... path to xyz file given"""

    args = {
        "geometry": str(water_filepath),
        "n_active_atoms": 1,
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "convergence": 1e-6,
        "savefile": None,
        "run_ccsd_emb": True,
        "run_fci_emb": True,
    }

    driver = NbedDriver(
        geometry=args["geometry"],
        n_active_atoms=args["n_active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        projector=args["projector"],
        localization=args["localization"],
        convergence=args["convergence"],
        savefile=args["savefile"],
        run_ccsd_emb=args["run_ccsd_emb"],
        run_fci_emb=args["run_fci_emb"],
    )
    assert isinstance(driver.embedded_scf, StreamObject)
    assert isinstance(driver.classical_energy, number)


def test_driver_standard_xyz_string_input() -> None:
    """test to check driver works... raw xyz string given"""
    water_xyz_raw = (
        "3\n \nH\t0.2774\t0.8929\t0.2544\nO\t0\t0\t0\nH\t0.6068\t-0.2383\t-0.7169"
    )
    args = {
        "geometry": water_xyz_raw,
        "n_active_atoms": 1,
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "convergence": 1e-6,
        "savefile": None,
        "run_ccsd_emb": False,
        "run_fci_emb": False,
    }

    driver = NbedDriver(
        geometry=args["geometry"],
        n_active_atoms=args["n_active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        projector=args["projector"],
        localization=args["localization"],
        convergence=args["convergence"],
        savefile=args["savefile"],
        run_ccsd_emb=args["run_ccsd_emb"],
        run_fci_emb=args["run_fci_emb"],
    )
    assert isinstance(driver.embedded_scf, StreamObject)
    assert isinstance(driver.classical_energy, number)


def test_n_active_atoms_valid() -> None:
    """test to check driver works... path to xyz file given"""

    args = {
        "geometry": str(water_filepath),
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "convergence": 1e-6,
        "savefile": None,
        "run_ccsd_emb": True,
        "run_fci_emb": True,
    }
    error_msg = "Invalid number of active atoms. Choose a number from 1 to 2."

    with pytest.raises(NbedConfigError, match=error_msg):
        NbedDriver(
            geometry=args["geometry"],
            n_active_atoms=0,
            basis=args["basis"],
            xc_functional=args["xc_functional"],
            projector=args["projector"],
            localization=args["localization"],
            convergence=args["convergence"],
            savefile=args["savefile"],
            run_ccsd_emb=args["run_ccsd_emb"],
            run_fci_emb=args["run_fci_emb"],
        )

    with pytest.raises(NbedConfigError, match=error_msg):
        NbedDriver(
            geometry=args["geometry"],
            n_active_atoms=3,
            basis=args["basis"],
            xc_functional=args["xc_functional"],
            projector=args["projector"],
            localization=args["localization"],
            convergence=args["convergence"],
            savefile=args["savefile"],
            run_ccsd_emb=args["run_ccsd_emb"],
            run_fci_emb=args["run_fci_emb"],
        )


def test_subsystem_dft() -> None:
    """Check thatcmponenets match total dft energy."""
    args = {
        "geometry": str(water_filepath),
        "n_active_atoms": 2,
        "basis": "STO-3G",
        "xc_functional": "b3lyp",
        "projector": "mu",
        "localization": "spade",
        "convergence": 1e-6,
        "savefile": None,
        "run_ccsd_emb": False,
        "run_fci_emb": False,
    }

    driver = NbedDriver(
        geometry=args["geometry"],
        n_active_atoms=args["n_active_atoms"],
        basis=args["basis"],
        xc_functional=args["xc_functional"],
        projector=args["projector"],
        localization=args["localization"],
        convergence=args["convergence"],
        savefile=args["savefile"],
        run_ccsd_emb=args["run_ccsd_emb"],
        run_fci_emb=args["run_fci_emb"],
    )

    energy_DFT_components = (
        driver.e_act
        + driver.e_env
        + driver.two_e_cross
        + driver._global_ks.energy_nuc()
    )

    assert isclose(energy_DFT_components, driver._global_ks.e_tot)


# def test_subsystem_dft_spin_consistency() -> None:
#     """Check restricted & unrestricted components match."""
#     args = {
#         "geometry": str(water_filepath),
#         "n_active_atoms": 1,
#         "basis": "STO-3G",
#         "xc_functional": "b3lyp",
#         "projector": "mu",
#         "localization": "spade",
#         "convergence": 1e-6,
#         "savefile": None,
#         "run_ccsd_emb": True,
#         "run_fci_emb": True,
#     }

#     restricted_driver = NbedDriver(
#         geometry=args["geometry"],
#         n_active_atoms=args["n_active_atoms"],
#         basis=args["basis"],
#         xc_functional=args["xc_functional"],
#         projector=args["projector"],
#         localization=args["localization"],
#         convergence=args["convergence"],
#         savefile=args["savefile"],
#         run_ccsd_emb=args["run_ccsd_emb"],
#         run_fci_emb=args["run_fci_emb"],
#     )

#     unrestricted_driver = NbedDriver(
#         geometry=args["geometry"],
#         n_active_atoms=args["n_active_atoms"],
#         basis=args["basis"],
#         xc_functional=args["xc_functional"],
#         projector=args["projector"],
#         localization=args["localization"],
#         convergence=args["convergence"],
#         savefile=args["savefile"],
#         run_ccsd_emb=args["run_ccsd_emb"],
#         run_fci_emb=args["run_fci_emb"],
#     )
#     # Could be problems with caching here
#     unrestricted_driver._restricted_scf = False
#     unrestricted_driver.embed()

#     restricted_driver.e_act
#     restricted_driver.e_env
#     restricted_driver.two_e_cross
#     restricted_driver._global_ks.energy_nuc()


if __name__ == "__main__":
    pass
