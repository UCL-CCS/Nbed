"""Shared fixtures for tests."""
import pytest
from pathlib import Path

@pytest.fixture(scope="module")
def water_filepath() -> Path:
    return Path("tests/molecules/water.xyz").absolute()