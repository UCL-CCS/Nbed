"""Custom Types and Enums."""

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    FilePath,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    TypeAdapter,
)

logger = logging.getLogger(__name__)


class ProjectorTypes(Enum):
    """Implemented Projectors."""

    MU = "mu"
    HUZ = "huzinaga"
    BOTH = "both"


class OccupiedLocalizerTypes(Enum):
    """Implemented Occupied Localizers."""

    SPADE = "spade"
    BOYS = "boys"
    IBO = "ibo"
    PM = "pm"


class VirtualLocalizerTypes(Enum):
    """Implemented Virtual Localizers."""

    CONCENTRIC = "cl"
    PROJECTED_AO = "pao"
    NONE = None


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
    match maybe_xyz:
        case str() | Path():
            if os.path.exists(maybe_xyz):
                with open(maybe_xyz) as file:
                    content = file.read()
                logger.debug("File content %s", content)
                TypeAdapter(XYZGeometry).validate_strings(content)
                return content
            else:
                logger.debug("Input geometry does not match existing file")
                return str(maybe_xyz)
        case _:
            return maybe_xyz


class NbedConfig(BaseModel):
    """Config for Nbed.

    Args:
        geometry (XYZGeometry): Path to .xyz file containing molecular geometry or raw xyz string.
        n_active_atoms (PositiveInt): The number of atoms to include in the active region.
        basis (str): The name of an atomic orbital basis set to use for chemistry calculations.
        xc_functional (str): The name of an Exchange-Correlation functional to be used for DFT.
        projector (ProjectorTypes): Projector to screen out environment orbitals, One of 'mu' or 'huzinaga'.
        localization (OccupiedLocalizerTypes): Orbital localization method to use. One of 'spade', 'pipek-mezey', 'boys' or 'ibo'.
        convergence (Annotated[float, Gt(gt=0), Lt(lt=1)]): The convergence tolerance for energy calculations.
        charge (PositiveInt): Charge of molecular species
        mu_level_shift (PositiveFloat): Level shift parameter to use for mu-projector.
        run_ccsd_emb (bool): Whether or not to find the CCSD energy of embbeded system for reference.
        run_fci_emb (bool): Whether or not to find the FCI energy of embbeded system for reference.
        run_virtual_localization (bool): Whether or not to localize virtual orbitals.
        n_mo_overwrite (tuple[None| PositiveInt, None | PositiveInt]): Optional overwrite values for occupied localizers.
        max_ram_memory (PositiveInt): Amount of RAM memery in MB available for PySCF calculation
        unit (str): molecular geometry unit 'Angstrom' or 'Bohr'
        max_hf_cycles (PositiveInt): max number of Hartree-Fock iterations allowed (for global and local HFock)
        max_dft_cycles (PositiveInt): max number of DFT iterations allowed in scf calc
        init_huzinaga_rhf_with_mu (bool): Hidden flag to seed huzinaga RHF with mu shift result (for developers only)
        savefile (FilePath): Location of file to save output to.
    """

    model_config = ConfigDict(extra="forbid")

    geometry: Annotated[XYZGeometry, BeforeValidator(validate_xyz_file)]
    n_active_atoms: PositiveInt
    basis: str
    xc_functional: str
    projector: ProjectorTypes = Field(default=ProjectorTypes.MU)
    localization: OccupiedLocalizerTypes = Field(default=OccupiedLocalizerTypes.SPADE)
    convergence: PositiveFloat = 1e-6
    charge: NonNegativeInt = Field(default=0)
    spin: NonNegativeInt = Field(default=0)
    unit: str = "angstrom"
    symmetry: bool = False

    savefile: FilePath | None = None

    run_ccsd_emb: bool = False
    run_fci_emb: bool = False
    run_dft_in_dft: bool = False

    mm_coords: list | None = None
    mm_charges: list | None = None
    mm_radii: list | None = None

    mu_level_shift: PositiveFloat = 1e6
    init_huzinaga_rhf_with_mu: bool = False

    virtual_localization: VirtualLocalizerTypes = Field(
        default=VirtualLocalizerTypes.CONCENTRIC
    )
    n_mo_overwrite: tuple[None | NonNegativeInt, None | NonNegativeInt] = (None, None)
    occupied_threshold: float = Field(default=0.95, gt=0, lt=1)
    virtual_threshold: float = Field(default=0.95, gt=0, lt=1)
    max_shells: PositiveInt = 4
    norm_cutoff: PositiveFloat = 0.05
    overlap_cutoff: PositiveFloat = 1e-5

    force_unrestricted: bool = False

    max_ram_memory: PositiveInt = 4000
    max_hf_cycles: PositiveInt = Field(default=50)
    max_dft_cycles: PositiveInt = Field(default=50)


def overwrite_config_kwargs(config: NbedConfig, **config_kwargs) -> NbedConfig:
    """Overwrites config values with key-words and revalidates.

    Args:
        config (NbedConfig): A config model.
        config_kwargs (dict): Any possible key-word arguments.

    Returns:
        NbedConfig: A validated config model.

    Raises:
        ValidationError: If key-word arguments provided are not part of model.
    """
    if config_kwargs != {}:
        logger.info("Overwriting select field with additonal config.")
        config_dict = config.model_dump()
        for k, v in config_kwargs.items():
            config_dict[k] = v
        return NbedConfig(**config_dict)
    else:
        return config


def parse_config(
    config: NbedConfig | str | None = None,
    **config_kwargs,
):
    """Parse the various config options and return a valid model.

    Args:
        config (NbedConfig): A validated config model or path to a '.json' config file.
        **config_kwargs: Allows arbitrary keyword arguments for manual configuration.

    Returns:
        NbedConfig: A valid config model.

    """
    match config:
        case NbedConfig():
            logger.info("Using validated config.")
            config = overwrite_config_kwargs(config, **config_kwargs)

        case str() | Path():
            logger.info("Using config file %s", config)
            logger.info("Validating config from file.")
            with open(FilePath(config)) as f:
                data = json.load(f)
            config = NbedConfig(**data)
            config = overwrite_config_kwargs(config, **config_kwargs)
        case None:
            logger.info("Validating config from passed arguments.")
            logger.debug(f"{config_kwargs=}")
            config = NbedConfig(**config_kwargs)
        case _:
            logger.warning("Unknown input to config argument will be ignored.")
            logger.debug(f"{config=}")
            logger.debug(f"{config_kwargs=}")
            config = NbedConfig(**config_kwargs)

    return config
