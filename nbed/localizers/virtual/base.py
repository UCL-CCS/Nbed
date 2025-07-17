"""Base Virtual Localizer Class."""

from abc import ABC, abstractmethod

from pyscf import gto
from pyscf.lib import StreamObject


class VirtualLocalizer(ABC):
    """Base class for virtual localizers.

    Args:
        embedded_scf (StreamObject): SCF object with occupied orbitals localized.
        n_active_atoms (int): Number of active atoms in the system.

    Attributes:
        embedded_scf (StreamObject): SCF object with occupied orbitals localized.
        n_active_atoms (int): Number of active atoms in the system.
    """

    def __init__(self, embedded_scf: StreamObject):
        """Initialize VirtualLocalizer.

        Args:
            embedded_scf (StreamObject): A pyscf SCF object.
        """
        self.embedded_scf = embedded_scf

    @abstractmethod
    def localize_virtual(self) -> gto.Mole:
        """Localize virtual orbitals.

        Returns:
            gto.Mole: Localized SCF object.
        """
        pass
