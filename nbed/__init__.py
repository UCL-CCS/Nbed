"""Init for Nbed package."""

from .embed import nbed
from .utils import setup_logs

__all__ = ["nbed", "load_hamiltonian"]

setup_logs()
