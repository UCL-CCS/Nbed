"""Init for Nbed package."""

from .config import NbedConfig
from .embed import nbed
from .localizers.ace import ACELocalizer
from .utils import setup_logs

__all__ = ["nbed", "NbedConfig", "ACELocalizer"]

setup_logs()
