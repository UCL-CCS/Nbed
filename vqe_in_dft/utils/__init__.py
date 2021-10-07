"""
Initialise
"""

from .exceptions import FragmenterError
from .log import setup_logs

__all__ = [setup_logs, FragmenterError]
