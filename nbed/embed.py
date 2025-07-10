"""Main embedding functionality."""

import json
import logging

from pydantic import FilePath

from .config import NbedConfig
from .driver import NbedDriver
from .utils import parse

logger = logging.getLogger(__name__)


def nbed(
    config: NbedConfig | FilePath | None = None,
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
        case FilePath():
            logger.info("Using config file %s", config)
            with open(config) as f:
                logger.info("Validating config from file.")
                config = NbedConfig(json.loads(f))
        case None:
            logger.info("Validating config from passed arguments.")
            config = NbedConfig(**config_kwargs)
        case _:
            logger.warning("Unknown input to config argument will be ignored.")
            logger.debug(f"{config=}")
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
