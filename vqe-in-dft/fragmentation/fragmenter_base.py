""" Base class for fragmenter algorithms """

import logging
from dataclasses import dataclass
from pyscf import gto
from pathlib import Path
from abc import ABC, abstractmethod, abstractproperty

logger = logging.getLogger(__name__)


class FragmentPair(ABC):
    """
    Two fragments in a tuple.
    
    TODO - once we know what the fragment objects can / should look like, 
    this can be subclassed into appropriate types.
    """
    pass

class Fragmenter(ABC):
    """
    Base class

    Properties
    ----------
    molecule: gto.Mol
        A pyscf molecule object.

    Methods
    -------
    _fragment: FragmentPair
        Method to fragment an input molecule.
    """

    def __init__(self, molecule: gto.Mol) -> None:
        self._molecule = molecule
        self._fragments = (None, None)

    @property
    def molecule(self) -> gto.Mol:
        return self._molecule

    @molecule.getter
    def get_molecule(self) -> gto.Mol:
        return self._molecule

    @molecule.setter
    def set_molecule(self) -> None:
        logger.error(
            "Please create a new Fragmenter instace to work with another molecule."
        )

    @abstractmethod
    def _fragment(self) -> FragmentPair:
        "Base method to split molecule into a pair"
        raise NotImplementedError

    @property
    def fragments(self) -> FragmentPair:
        return self._fragments()

    @fragments.getter
    def get_fragments(self) -> FragmentPair:
        return self._fragments()
