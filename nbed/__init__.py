"""Init for Nbed package."""

from .config import NbedConfig
from .embed import nbed
from .utils import setup_logs

__all__ = ["nbed", "NbedConfig"]

setup_logs()
