"""
Logging config for whole package.
"""

import logging
from pathlib import Path

def setup_logs() -> None:
    "Initialise logging"

    config_dict = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s: %(name)s: %(levelname)s: %(message)s'
            },
        },
        'handlers': {
            'file_handler': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'filename': Path("../../vqe-in-dft.log"),
                'encoding': 'utf-8'
            },
            'stream_handler': {
                'class': 'logging.StreamHandler',
                'level': 'WARNING',
                'formatter': 'standard',
            },
        },
    }

    logging.config.dictConfig(config_dict)
