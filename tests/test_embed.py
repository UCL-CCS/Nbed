"""File to contain tests of the embed.py script."""

from nbed.embed import nbed
from nbed.driver import NbedDriver

def test_nbed(nbed_config) -> None:
    """Test nbed"""
    assert isinstance(nbed(nbed_config), NbedDriver)

if __name__ == "__main__":
    pass
