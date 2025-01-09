"""Localized SCF Methods."""

from .embedded_hcore_funcs import energy_elec
from .huzinaga_scf import huzinaga_scf

all = [huzinaga_scf, energy_elec]
