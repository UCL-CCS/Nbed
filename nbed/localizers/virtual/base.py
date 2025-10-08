"""Base Virtual Localizer Class."""

from abc import ABC, abstractmethod

from pyscf import scf  # type:ignore


class VirtualLocalizer(ABC):
    """Base class for virtual localizers.

    Args:
        embedded_scf (scf.hf.SCF): SCF object with occupied orbitals localized.
        n_active_atoms (int): Number of active atoms in the system.

    Attributes:
        embedded_scf (scf.hf.SCF): SCF object with occupied orbitals localized.
        n_active_atoms (int): Number of active atoms in the system.
    """

    def __init__(self, n_active_atoms: int):
        """Initialize VirtualLocalizer.

        Args:
            embedded_scf (scf.hf.SCF): A pyscf SCF object.
            n_active_atoms (int): The number of atoms in the active region.
        """
        self._n_active_atoms = n_active_atoms

    @abstractmethod
    def localize_virtual(self) -> scf.hf.SCF:
        """Localize virtual orbitals.

        Returns:
            gto.Mole: Localized SCF object.
        """
        pass
