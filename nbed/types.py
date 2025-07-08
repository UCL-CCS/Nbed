"""Custom Types and Enums."""

from enum import Enum
from typing import Annotated

from pydantic import Field, FilePath, TypeAdapter


class Projector(Enum):
    """Implemented Projectors."""

    MU = "mu"
    HUZ = "huzinaga"


class Localizer(Enum):
    """Implemented Occupied Localizers."""

    SPADE = "spade"
    BOYS = "boys"
    IBO = "ibo"
    PM = "pm"


XYZGeometry = Annotated[
    str, Field(pattern="^\\d+\n\\s?\n(?:\\w(?:\\s+\\-?\\d\\.\\d+){3}\n?)*")
]


def validate_xyz_file(path: FilePath) -> str:
    """Validates the the filepath given leads to a valid XYZ formatted file.

    Args:
        path (FilePath): A path to an existing file.

    Returns:
        str: an XYZ geometry string.
    """
    with open(path) as file:
        content = file.read()
    TypeAdapter(XYZGeometry).validate_strings(content)
    return content
