"""Init for Nbed package."""

from .ace import ace_of_spade
from .config import NbedConfig
from .embed import nbed
from .utils import setup_logs

__all__ = ["nbed", "NbedConfig", "ace_of_spade"]

setup_logs()
