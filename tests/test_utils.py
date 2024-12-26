"""Tests for utility functions."""

import os

import pytest

from nbed.utils import (
    build_ordered_xyz_string,
    pubchem_mol_geometry,
    save_ordered_xyz_file,
)


@pytest.fixture
def water_struct_dict() -> dict[int, tuple[str, tuple[float, float, float]]]:
    return {
        0: ("O", (0, 0, 0)),
        1: ("H", (0.2774, 0.8929, 0.2544)),
        2: ("H", (0.6068, -0.2383, -0.7169)),
    }


@pytest.fixture
def water_xyz_o_active():
    return "3\n \nO\t0\t0\t0\nH\t0.2774\t0.8929\t0.2544\nH\t0.6068\t-0.2383\t-0.7169\n"


def test_pubchem_mol_geometry(water_struct_dict) -> None:

    assert pubchem_mol_geometry("water") == water_struct_dict


def test_build_ordered_xyz_string(water_struct_dict, water_xyz_o_active) -> None:

    assert build_ordered_xyz_string(water_struct_dict, [0]) == water_xyz_o_active

    water_xyz_h_active = (
        "3\n \nH\t0.2774\t0.8929\t0.2544\nH\t0.6068\t-0.2383\t-0.7169\nO\t0\t0\t0\n"
    )

    assert build_ordered_xyz_string(water_struct_dict, [1, 2]) == water_xyz_h_active


def test_save_xyz(water_struct_dict, water_xyz_o_active) -> None:
    file_path = save_ordered_xyz_file(
        file_name="water_test", struct_dict=water_struct_dict, active_atom_inds=[0]
    )
    f = open(file_path, "r")

    try:
        assert f.read() == water_xyz_o_active
    except AssertionError:
        raise AssertionError("File not saved correctly")
    finally:
        f.close()
        os.remove("tests/molecules/water.xyz")
