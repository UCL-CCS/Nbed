""" Base class for fragmenter algorithms """

import logging
from pyscf import gto, dft
from .enums import FragMethods
from .fragment_pair import FragmentPair
from ..utils.exceptions import FragmenterError

logger = logging.getLogger(__name__)


class Fragmenter:
    """
    Fragmenter class

    Properties
    ----------
    molecule: pyscf.gto.Mol
        A pyscf molecule object.

    """

    def __init__(self, molecule: gto.Mol) -> None:
        self._molecule = molecule
        self._fragments = {}

    @property
    def mol(self) -> gto.Mol:
        return self._molecule

    @property
    def molecule(self) -> gto.Mol:
        return self._molecule

    @mol.getter
    @molecule.getter
    def get_molecule(self) -> gto.Mol:
        return self._molecule

    @mol.setter
    @molecule.setter
    def set_molecule(self) -> None:
        logger.error(
            "Please create a new Fragmenter instace to work with another molecule."
        )

    @property
    def fragments(self) -> FragmentPair:
        return self._fragments()

    @fragments.getter
    def get_fragments(self) -> FragmentPair:
        return self._fragments()

    def run(self, method_ident: str) -> FragmentPair:
        """
        Base method to split molecule into a pair.
        
        Parameters
        ----------

        method: str
            The name of a fragmentation method in this class.
        """
        logger.debug(f"Running fragmentation with method {method_ident}.")
        if not hasattr(FragMethods, method_ident):
            logger.error("Method name not found in FragMethods.")
            raise FragmenterError("The method %s you have selected is not implemented", method_ident)
        
        method_name = FragMethods[method_ident].value
        frag_method = getattr(self, method_name, None)

        if frag_method is None:
            logger.error("Class method not found for %s", method_name)
            raise FragmenterError("Method %s not implemented.", method_name)

        return frag_method()

    def _absolute_localisation(self) -> FragmentPair:
        """Fragmentation by absolute localisation.
        
        Steps to fragmentation.

        """
        logger.debug("Fragmenting with Absolute Localisation")
        raise NotImplementedError

    def _spade(self) -> FragmentPair:
        "Fragmentation by SPADE"
        raise NotImplementedError