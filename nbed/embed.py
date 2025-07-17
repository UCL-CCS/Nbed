"""Main embedding functionality."""

import json
import logging
from pathlib import Path

from pydantic import FilePath

from .config import NbedConfig
from .driver import NbedDriver
from .utils import parse

logger = logging.getLogger(__name__)


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


def nbed(
    config: NbedConfig | str | None = None,
    **config_kwargs,
):
    """Import interface for the nbed package.

    This function calls the NbedDriver class to create a second quantized hamiltonian
    using configuration provided.

    Args:
        config (NbedConfig): A validated config model or path to a '.json' config file.
        **config_kwargs: Allows arbitrary keyword arguments for manual configuration.

    Returns:
        NbedDriver: An embedded driver.
    """
    logger.info(f"Running Nbed with:\n\tconfig\t{config}\n\tkeywords\t{config_kwargs}")
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

    driver = NbedDriver(config)
    driver.embed()
    return driver


def cli() -> None:
    """CLI Interface."""
    config = parse()
    nbed(config)


if __name__ == "__main__":
    cli()
