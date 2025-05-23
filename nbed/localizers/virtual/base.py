"""Base Virtual Localizer Class."""

from abc import ABC, abstractmethod

from pyscf import gto
from pyscf.lib import StreamObject


class BaseVirtualLocalizer(ABC):
    """Base class for virtual localizers.

    Args:
        embedded_scf (StreamObject): SCF object with occupied orbitals localized.
        n_active_atoms (int): Number of active atoms in the system.

    Attributes:
        embedded_scf (StreamObject): SCF object with occupied orbitals localized.
        n_active_atoms (int): Number of active atoms in the system.
    """

    def __init__(self, embedded_scf: StreamObject):
        """Initialize BaseVirtualLocalizer.

        Args:
            embedded_scf (StreamObject): A pyscf SCF object.
        """
        self.embedded_scf = embedded_scf

    @abstractmethod
    def localize_virtual(self, restricted: bool) -> gto.Mole:
        """Localize virtual orbitals.

        Args:
            restricted (bool): Whether the SCF is restricted or unrestricted.

        Returns:
            gto.Mole: Localized SCF object.
        """
        pass
