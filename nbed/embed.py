"""Main embedding functionality."""

import json
import logging
from pathlib import Path

from pydantic import FilePath

from .config import NbedConfig
from .driver import NbedDriver
from .utils import parse

logger = logging.getLogger(__name__)


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
    match config:
        case NbedConfig():
            logger.info("Using validated config.")
        case str() | Path():
            logger.info("Using config file %s", config)
            with open(FilePath(config)) as f:
                logger.info("Validating config from file.")
                data = json.load(f)
                config = NbedConfig(**data)
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
