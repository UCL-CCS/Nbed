"""
Initialise
"""

from .log import setup_logs
from .exceptions import FragmenterError

__all__ = [setup_logs, FragmenterError]
