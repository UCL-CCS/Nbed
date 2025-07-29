"""Main embedding functionality."""

import logging

from .config import NbedConfig, parse_config
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
    logger.info(f"Running Nbed with:\n\tconfig\t{config}\n\tkeywords\t{config_kwargs}")

    parsed_config = parse_config(config, **config_kwargs)
    driver = NbedDriver(parsed_config)
    driver.embed()
    return driver


def cli() -> None:
    """CLI Interface."""
    config = parse()
    nbed(config)


if __name__ == "__main__":
    cli()
