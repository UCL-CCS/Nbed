"""Custom Types and Enums."""

import os
from enum import Enum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    FilePath,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    TypeAdapter,
)


class ProjectorEnum(Enum):
    """Implemented Projectors."""

    MU = "mu"
    HUZ = "huzinaga"
    BOTH = "both"


class LocalizerEnum(Enum):
    """Implemented Occupied Localizers."""

    SPADE = "spade"
    BOYS = "boys"
    IBO = "ibo"
    PM = "pm"


XYZGeometry = Annotated[
    str, Field(pattern="^\\d+\n\\s?\n(?:\\w(?:\\s+\\-?\\d\\.\\d+){3}\n?)*")
]


def validate_xyz_file(maybe_xyz: Any) -> str:
    """Validates the the filepath given leads to a valid XYZ formatted file.

    Args:
        maybe_xyz (Any): A path to an existing file.

    Returns:
        str: an XYZ geometry string.
    """
    if os.path.exists(maybe_xyz):
        with open(maybe_xyz) as file:
            content = file.read()
        TypeAdapter(XYZGeometry).validate_strings(content)
        return content
    else:
        return maybe_xyz


class NbedConfig(BaseModel):
    """Config for Nbed.

    Args:
        geometry (str): Path to .xyz file containing molecular geometry or raw xyz string.
        n_active_atoms (int): The number of atoms to include in the active region.
        basis (str): The name of an atomic orbital basis set to use for chemistry calculations.
        xc_functional (str): The name of an Exchange-Correlation functional to be used for DFT.
        projector (str): Projector to screen out environment orbitals, One of 'mu' or 'huzinaga'.
        localization (str): Orbital localization method to use. One of 'spade', 'pipek-mezey', 'boys' or 'ibo'.
        convergence (float): The convergence tolerance for energy calculations.
        charge (int): Charge of molecular species
        mu_level_shift (float): Level shift parameter to use for mu-projector.
        run_ccsd_emb (bool): Whether or not to find the CCSD energy of embbeded system for reference.
        run_fci_emb (bool): Whether or not to find the FCI energy of embbeded system for reference.
        run_virtual_localization (bool): Whether or not to localize virtual orbitals.
        n_mo_overwrite (tuple[None| int, None | int]): Optional overwrite values for occupied localizers.
        max_ram_memory (int): Amount of RAM memery in MB available for PySCF calculation
        pyscf_print_level (int): Amount of information PySCF prints
        unit (str): molecular geometry unit 'Angstrom' or 'Bohr'
        max_hf_cycles (int): max number of Hartree-Fock iterations allowed (for global and local HFock)
        max_dft_cycles (int): max number of DFT iterations allowed in scf calc
        init_huzinaga_rhf_with_mu (bool): Hidden flag to seed huzinaga RHF with mu shift result (for developers only)
        savefile (FilePath): Location of file to save output to.
    """

    geometry: Annotated[XYZGeometry, BeforeValidator(validate_xyz_file)]
    n_active_atoms: PositiveInt
    basis: str
    xc_functional: str
    projector: ProjectorEnum = Field(default=ProjectorEnum.MU)
    localization: LocalizerEnum = Field(default=LocalizerEnum.SPADE)
    convergence: PositiveFloat = 1e-6
    charge: NonNegativeInt = Field(default=0)
    spin: NonNegativeInt = Field(default=0)
    unit: str = "angstrom"
    symmetry: bool = False

    savefile: FilePath | None = None
    transform: str | None = None

    run_ccsd_emb: bool = False
    run_fci_emb: bool = False
    run_virtual_localization: bool = True
    run_dft_in_dft: bool = False

    mm_coords: list | None = None
    mm_charges: list | None = None
    mm_radii: list | None = None

    n_mo_overwrite: tuple[None | NonNegativeInt, None | NonNegativeInt] = (None, None)
    mu_level_shift: PositiveFloat = 1e6
    occupied_threshold: float = Field(default=0.95, gt=0, lt=1)
    virtual_threshold: float = Field(default=0.95, gt=0, lt=1)
    max_shells: PositiveInt = 4
    init_huzinaga_rhf_with_mu: bool = False
    force_unrestricted: bool = False

    max_ram_memory: PositiveInt = 4000
    max_hf_cycles: PositiveInt = Field(default=50)
    max_dft_cycles: PositiveInt = Field(default=50)
