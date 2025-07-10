"""File to contain tests of the embed.py script."""

from nbed.embed import nbed
from nbed.driver import NbedDriver
import json

def test_config_input(nbed_config) -> None:
    """Test nbed"""
    assert isinstance(nbed(nbed_config), NbedDriver)

def test_file_input(config_file) -> None:
    assert isinstance(nbed(config_file), NbedDriver)

def test_args_input(config_file) -> None:
    with open(config_file) as f:
        args = json.load(f)
    assert isinstance(nbed(**args), NbedDriver)

if __name__ == "__main__":
    pass
