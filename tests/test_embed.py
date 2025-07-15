"""File to contain tests of the embed.py script."""

from nbed.embed import nbed
from nbed.driver import NbedDriver
import json
import pytest
from pydantic import ValidationError

def test_config_input(nbed_config) -> None:
    """Test nbed"""
    assert isinstance(nbed(nbed_config), NbedDriver)

def test_config_overwrite(nbed_config) -> None:
    driver = nbed(nbed_config, n_active_atoms=1)
    assert driver.config.n_active_atoms == 1

def test_file_input(config_file) -> None:
    assert isinstance(nbed(config_file), NbedDriver)


def test_args_input(config_file) -> None:
    with open(config_file) as f:
        args = json.load(f)
    assert isinstance(nbed(**args), NbedDriver)

def test_none_config_input(nbed_config)-> None:
    args = nbed_config.model_dump()
    assert nbed(config=None, **args).config == nbed_config

    args.pop("geometry")
    with pytest.raises(ValidationError) as excinfo:
        nbed(config=None, **args)

def test_wrong_config_object(nbed_config) -> None:
    args = nbed_config.model_dump()
    assert (nbed(config=["a","list"], **args).config == nbed_config)

if __name__ == "__main__":
    pass
