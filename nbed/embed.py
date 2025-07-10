"""Main embedding functionality."""

import logging

from .config import NbedConfig
from .driver import NbedDriver
from .utils import parse

logger = logging.getLogger(__name__)


def nbed(
    config: NbedConfig,
):
    """Import interface for the nbed package.

    This function calls the NbedDriver class to create a second quantized hamiltonian
    using configuration provided.

    Args:
        config (NbedConfig): A validated config model, overwrites other input.

    Returns:
        object: A qubit hamiltonian object which can be used in the quantum backend specified by 'output'.
    """
    driver = NbedDriver(config)
    driver.embed()
    return driver


def cli() -> None:
    """CLI Interface."""
    config = parse()
    nbed(config)


if __name__ == "__main__":
    cli()
