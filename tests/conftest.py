"""Shared fixtures for tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def water_filepath() -> Path:
    return Path("tests/molecules/water.xyz").absolute()
