"""Module containg the NbedDriver Class."""

import logging
import os
from copy import copy
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import scipy as sp
from cached_property import cached_property
from pyscf import cc, dft, fci, gto, scf
from pyscf.lib import StreamObject

from nbed.exceptions import NbedConfigError
from nbed.localizers import (
    BOYSLocalizer,
    IBOLocalizer,
    Localizer,
    PMLocalizer,
    SPADELocalizer,
)

from .scf import _absorb_h1e, energy_elec, huzinaga_HF, huzinaga_KS

# from .log_conf import setup_logs

# logfile = Path(__file__).parent/Path(".nbed.log")

# Create the Logger
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# # Create the Handler for logging data to a file
# file_handler = logging.FileHandler(filename=logfile, mode="w")
# file_handler.setLevel(logging.DEBUG)

# # Create a Formatter for formatting the log messages
# file_formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
# stream_formatter = logging.Formatter("%(levelname)s %(message)s")

# # Create the Handler for logging data to console
# stream_handler = logging.StreamHandler()
# stream_handler.setLevel(logging.INFO)

# # Add the Formatter to the Handlers
# file_handler.setFormatter(file_formatter)
# stream_handler.setFormatter(stream_formatter)

# # Add the Handler to the Logger
# logger.addHandler(file_handler)
# logger.addHandler(stream_handler)
# logger.debug("Logging configured.")


class NbedDriver:
    """Function to return the embedding Qubit Hamiltonian.

    Args:
        geometry (str): Path to .xyz file containing molecular geometry or raw xyz string.
        n_active_atoms (int): The number of atoms to include in the active region.
        basis (str): The name of an atomic orbital basis set to use for chemistry calculations.
        xc_functonal (str): The name of an Exchange-Correlation functional to be used for DFT.
        projector (str): Projector to screen out environment orbitals, One of 'mu' or 'huzinaga'.
        localization (str): Orbital localization method to use. One of 'spade', 'pipek-mezey', 'boys' or 'ibo'.
        convergence (float): The convergence tolerance for energy calculations.
        charge (int): Charge of molecular species
        mu_level_shift (float): Level shift parameter to use for mu-projector.
        run_ccsd_emb (bool): Whether or not to find the CCSD energy of embbeded system for reference.
        run_fci_emb (bool): Whether or not to find the FCI energy of embbeded system for reference.
        run_virtual_localization (bool): Whether or not to localize virtual orbitals.
        max_ram_memory (int): Amount of RAM memery in MB available for PySCF calculation
        pyscf_print_level (int): Amount of information PySCF prints
        unit (str): molecular geometry unit 'Angstrom' or 'Bohr'
        max_hf_cycles (int): max number of Hartree-Fock iterations allowed (for global and local HFock)
        max_dft_cycles (int): max number of DFT iterations allowed in scf calc
        init_huzinaga_rhf_with_mu (bool): Hidden flag to seed huzinaga RHF with mu shift result (for developers only)
        return_dict (boolean): returns a dictionary containing geometry, hamiltonian and energies (FCI, CCSD, DFT)

    Attributes:
        _global_fci (StreamObject): A Qubit Hamiltonian of some kind
        e_act (float): Active energy from subsystem DFT calculation
        e_env (float): Environment energy from subsystem DFT calculation
        two_e_cross (float): two electron energy from cross terms (includes exchange correlation
                             and Coulomb contribution) of subsystem DFT calculation
        molecular_ham (InteractionOperator): molecular Hamiltonian for active subsystem (projection using mu shift operator)
        classical_energy (float): environment correction energy to obtain total energy (for mu shift method)
        molecular_ham (InteractionOperator): molecular Hamiltonian for active subsystem (projection using huzianga operator)
        classical_energy (float): environment correction energy to obtain total energy (for huzianga method)
    """

    def __init__(
        self,
        geometry: str,
        n_active_atoms: int,
        basis: str,
        xc_functional: str,
        projector: str,
        localization: Optional[str] = "spade",
        convergence: Optional[float] = 1e-6,
        charge: Optional[int] = 0,
        spin: Optional[int] = 0,
        mu_level_shift: Optional[float] = 1e6,
        run_ccsd_emb: Optional[bool] = False,
        run_fci_emb: Optional[bool] = False,
        run_virtual_localization: Optional[bool] = False,
        run_dft_in_dft: Optional[bool] = False,
        max_ram_memory: Optional[int] = 4000,
        pyscf_print_level: int = 1,
        unit: Optional[str] = "angstrom",
        occupied_threshold: Optional[float] = 0.95,
        virtual_threshold: Optional[float] = 0.95,
        max_shells: Optional[int] = 4,
        init_huzinaga_rhf_with_mu: bool = False,
        max_hf_cycles: int = 50,
        max_dft_cycles: int = 50,
        return_dict: Optional[bool] = False,
        force_unrestricted: Optional[bool] = False,
    ):
        """Initialise class."""
        logger.debug("Initialising driver.")
        config_valid = True
        if projector not in ["mu", "huzinaga", "both"]:
            logger.error(
                "Invalid projector %s selected. Choose from 'mu' or 'huzinzaga'.",
                projector,
            )
            config_valid = False

        if localization not in ["spade", "ibo", "boys", "pipek-mezey"]:
            logger.error(
                "Invalid localization method %s. Choose from 'ibo','boys','pipek-mezey' or 'spade'.",
                localization,
            )
            config_valid = False

        if not config_valid:
            logger.error("Invalid config.")
            raise NbedConfigError("Invalid config.")

        self.geometry = geometry
        self.n_active_atoms = n_active_atoms
        self.basis = basis.lower()
        self.xc_functional = xc_functional.lower()
        self.projector = projector.lower()
        self.localization = localization.lower()
        self.convergence = convergence
        self.charge = charge
        self.spin = spin
        self.mu_level_shift = mu_level_shift
        self.run_ccsd_emb = run_ccsd_emb
        self.run_fci_emb = run_fci_emb
        self.run_virtual_localization = run_virtual_localization
        self.run_dft_in_dft = run_dft_in_dft
        self.max_ram_memory = max_ram_memory
        self.pyscf_print_level = pyscf_print_level
        self.unit = unit
        self.occupied_threshold = occupied_threshold
        self.virtual_threshold = virtual_threshold
        self.max_shells = max_shells
        self.max_hf_cycles = max_hf_cycles
        self.max_dft_cycles = max_dft_cycles

        self._check_active_atoms()
        self.localized_system = None
        self.two_e_cross = None
        self.dft_potential = None
        self.electron = None
        self.v_emb = None

        if force_unrestricted:
            logger.debug("Forcing unrestricted SCF")
            self._restricted_scf = False
        elif self.charge % 2 == 1:
            logger.debug("Open shells, using unrestricted SCF.")
            self._restricted_scf = False
        else:
            logger.debug("Closed shells, using restricted SCF.")
            self._restricted_scf = True

        self.embed(init_huzinaga_rhf_with_mu=init_huzinaga_rhf_with_mu)

        logger.debug("Driver initialisation complete.")

        if return_dict:
            self.return_dictionary()

    def _build_mol(self) -> gto.mole:
        """Function to build PySCF molecule.

        Returns:
            full_mol (gto.mol): built PySCF molecule object
        """
        logger.debug("Constructing molecule.")
        if os.path.exists(self.geometry):
            # geometry is an xyz file
            full_mol = gto.Mole(
                atom=self.geometry,
                basis=self.basis,
                charge=self.charge,
                unit=self.unit,
                spin=self.spin,
            ).build()
        else:
            logger.info(
                "Input geometry is not an existing file. Assumng raw xyz input."
            )
            logger.info("Input geometry: %s", self.geometry)
            # geometry is raw xyz string
            full_mol = gto.Mole(
                atom=self.geometry[3:],
                basis=self.basis,
                charge=self.charge,
                spin=self.spin,
                unit=self.unit,
            ).build()
        logger.debug("Molecule built.")
        return full_mol

    @cached_property
    def _global_hf(self) -> StreamObject:
        """Run full system Hartree-Fock."""
        logger.debug("Running full system HF.")
        mol_full = self._build_mol()
        # run Hartree-Fock
        global_hf = scf.UHF(mol_full) if not self._restricted_scf else scf.RHF(mol_full)
        global_hf.conv_tol = self.convergence
        global_hf.max_memory = self.max_ram_memory
        global_hf.verbose = self.pyscf_print_level
        global_hf.max_cycle = self.max_hf_cycles
        global_hf.kernel()
        logger.info(f"Global HF: {global_hf.e_tot}")

        return global_hf

    @cached_property
    def _global_ccsd(self) -> StreamObject:
        """Function to run full molecule CCSD calculation."""
        logger.debug("Running full system CC.")
        # run CCSD after HF

        global_cc = cc.CCSD(self._global_hf)
        global_cc.conv_tol = self.convergence
        global_cc.verbose = self.pyscf_print_level
        global_cc.max_memory = self.max_ram_memory
        global_cc.run()
        logger.info(f"Global CCSD: {global_cc.e_tot}")

        return global_cc

    @cached_property
    def _global_fci(self) -> StreamObject:
        """Function to run full molecule FCI calculation.

        WARNING: FACTORIAL SCALING IN BASIS STATES!
        """
        logger.debug("Running full system FCI.")
        # run FCI after HF
        global_fci = fci.FCI(self._global_hf)
        global_fci.conv_tol = self.convergence
        global_fci.verbose = self.pyscf_print_level
        global_fci.max_memory = self.max_ram_memory
        global_fci.run()
        logger.info(f"Global FCI: {global_fci.e_tot}")

        return global_fci

    @cached_property
    def _global_ks(self):
        """Method to run full cheap molecule RKS DFT calculation.

        Note this is necessary to perform localization procedure.
        """
        logger.debug("Running full system RKS DFT.")
        mol_full = self._build_mol()

        global_ks = scf.RKS(mol_full) if self._restricted_scf else scf.UKS(mol_full)
        global_ks.conv_tol = self.convergence
        global_ks.xc = self.xc_functional
        global_ks.max_memory = self.max_ram_memory
        global_ks.verbose = self.pyscf_print_level
        global_ks.max_cycle = self.max_dft_cycles
        global_ks.kernel()
        logger.info(f"Global RKS: {global_ks.e_tot}")

        if global_ks.converged is not True:
            logger.warning("(cheap) global DFT calculation has NOT converged!")

        return global_ks

    def _check_active_atoms(self) -> None:
        """Check that the number of active atoms is valid."""
        all_atoms = self._build_mol().natm
        if self.n_active_atoms not in range(1, all_atoms):
            logger.error("Invalid number of active atoms.")
            raise NbedConfigError(
                f"Invalid number of active atoms. Choose a number from 1 to {all_atoms-1}."
            )
        logger.debug("Number of active atoms valid.")

    def _localize(self):
        """Run the localizer class."""
        logger.debug(f"Getting localized system using {self.localization}.")

        localizers = {
            "spade": SPADELocalizer,
            "boys": BOYSLocalizer,
            "ibo": IBOLocalizer,
            "pipek-mezey": PMLocalizer,
        }

        if self.localization == "spade":
            localized_system = localizers[self.localization](
                self._global_ks,
                self.n_active_atoms,
                max_shells=self.max_shells,
            )
        else:
            localized_system = localizers[self.localization](
                self._global_ks,
                self.n_active_atoms,
                occ_cutoff=self.occupied_threshold,
                virt_cutoff=self.virtual_threshold,
            )
        return localized_system

    def _init_local_hf(self) -> Union[scf.uhf.UHF, scf.rhf.RHF]:
        """Function to build embedded HF object for active subsystem.

        Note this function overwrites the total number of electrons to only include active number.

        Returns:
            local_hf (scf.uhf.UHF or scf.rhf.RHF): embedded Hartree-Fock object.
        """
        logger.debug("Constructing localised RHF object.")
        embedded_mol: gto.Mole = self._build_mol()

        # overwrite total number of electrons to only include active system
        if self._restricted_scf:
            embedded_mol.nelectron = 2 * len(self.localized_system.active_MO_inds)
            self.electron = embedded_mol.nelectron
            local_hf: scf.rhf.RHF = scf.RHF(embedded_mol)
        else:
            embedded_mol.nelectron = len(self.localized_system.active_MO_inds) + len(
                self.localized_system.beta_active_MO_inds
            )
            self.electron = embedded_mol.nelectron
            local_hf: scf.uhf.UHF = scf.UHF(embedded_mol)

        local_hf.max_memory = self.max_ram_memory
        local_hf.conv_tol = self.convergence
        local_hf.verbose = self.pyscf_print_level
        local_hf.max_cycle = self.max_hf_cycles

        return local_hf

    def _init_local_ks(self, xc_functional: str) -> Union[dft.uks.UKS, dft.rks.RKS]:
        """Function to build embedded Hartree Fock object for active subsystem.

        Note this function overwrites the total number of electrons to only include active number.

        Args:
            xc_functonal (str): XC functional to use in embedded calculation.

        Returns:
            local_ks (pyscf.dft.rks.RKS or pyscf.dft.uks.UKS): embedded Kohn-Sham DFT object.
        """
        logger.debug("Initialising localised RKS object.")
        embedded_mol: gto.Mole = self._build_mol()

        if self._restricted_scf:
            # overwrite total number of electrons to only include active system
            embedded_mol.nelectron = 2 * len(self.localized_system.active_MO_inds)
            self.electron = embedded_mol.nelectron
            local_ks: dft.rks.RKS = scf.RKS(embedded_mol)
        else:
            embedded_mol.nelectron = len(self.localized_system.active_MO_inds) + len(
                self.localized_system.beta_active_MO_inds
            )
            self.electron = embedded_mol.nelectron
            local_ks: dft.uks.UKS = scf.UKS(embedded_mol)

        local_ks.max_memory = self.max_ram_memory
        local_ks.conv_tol = self.convergence
        local_ks.verbose = self.pyscf_print_level
        local_ks.xc = xc_functional

        return local_ks

    def _subsystem_dft(self) -> None:
        """Function to perform subsystem RKS DFT calculation."""
        logger.debug("Calculating active and environment subsystem terms.")

        def _ks_components(
            ks_system: Localizer,
            subsystem_dm: np.ndarray,
        ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
            """Calculate the components of subsystem energy from a RKS DFT calculation.

            For a given density matrix this function returns the electronic energy, exchange correlation energy and
            J,K, V_xc matrices.

            Args:
                dm_matrix (np.ndarray): density matrix (to calculate all matrices from)

            Returns:
                Energy_elec (float): DFT energy defined by input density matrix
                e_xc (float): exchange correlation energy defined by input density matrix
                J_mat (np.ndarray): J_matrix defined by input density matrix
            """
            logger.debug("Finding subsystem RKS componenets.")
            # It seems that PySCF lumps J and K in the J array
            # need to access the potential for the right subsystem for unrestricted
            logger.debug(subsystem_dm.shape)
            two_e_term = ks_system.get_veff(dm=subsystem_dm)
            j_mat = ks_system.get_j(dm=subsystem_dm)
            # k_mat = np.zeros_like(j_mat) not needed for PySCF.

            # v_xc = two_e_term - j_mat

            if not self._restricted_scf:
                j_tot = j_mat[0] + j_mat[1]
                dm_tot = subsystem_dm[0] + subsystem_dm[1]
            else:
                j_tot = j_mat
                dm_tot = subsystem_dm

            e_act = (
                np.einsum("ij,ji->", ks_system.get_hcore(), dm_tot)
                + 0.5 * (np.einsum("ij,ji->", j_tot, dm_tot))
                + two_e_term.exc
            )

            # if check_E_with_pyscf:
            #     energy_elec_pyscf = self._global_ks.energy_elec(dm=dm_matrix)[0]
            #     if not np.isclose(energy_elec_pyscf, energy_elec):
            #         raise ValueError("Energy calculation incorrect")
            logger.debug("Subsystem RKS components found.")
            return e_act, two_e_term, j_mat

        if not self._restricted_scf:
            dm_act = np.array(
                [self.localized_system.dm_active, self.localized_system.beta_dm_active]
            )
            dm_env = np.array(
                [self.localized_system.dm_enviro, self.localized_system.beta_dm_enviro]
            )
        else:
            dm_act = self.localized_system.dm_active
            dm_env = self.localized_system.dm_enviro

        (e_act, two_e_act, j_act) = _ks_components(self._global_ks, dm_act)
        # logger.debug(e_act, alpha_e_xc_act)
        (e_env, two_e_env, j_env) = _ks_components(self._global_ks, dm_env)
        # logger.debug(alpha_e_env, alpha_e_xc_env, alpha_ecoul_env)
        self.e_act = e_act
        self.e_env = e_env

        # Computing cross subsystem terms
        logger.debug("Calculating two electron cross subsystem energy.")
        total_dm = self.localized_system.dm_active + self.localized_system.dm_enviro

        if not self._restricted_scf:
            total_dm += (
                self.localized_system.beta_dm_active
                + self.localized_system.beta_dm_enviro
            )

        two_e_term_total = self._global_ks.get_veff(dm=total_dm)
        e_xc_total = two_e_term_total.exc

        if self._restricted_scf:
            j_cross = 0.5 * (
                np.einsum("ij,ij", self.localized_system.dm_active, j_env)
                + np.einsum("ij,ij", self.localized_system.dm_enviro, j_act)
            )
        else:
            j_cross = 0.5 * (
                np.einsum("ij,ij", self.localized_system.dm_active, j_env[0])
                + np.einsum("ij,ij", self.localized_system.dm_enviro, j_act[0])
                + np.einsum("ij,ij", self.localized_system.dm_active, j_env[1])
                + np.einsum("ij,ij", self.localized_system.dm_enviro, j_act[1])
                + np.einsum("ij,ij", self.localized_system.beta_dm_active, j_env[1])
                + np.einsum("ij,ij", self.localized_system.beta_dm_enviro, j_act[1])
                + np.einsum("ij,ij", self.localized_system.beta_dm_active, j_env[0])
                + np.einsum("ij,ij", self.localized_system.beta_dm_enviro, j_act[0])
            )

        # Because of projection we expect kinetic term to be zero
        k_cross = 0.0

        xc_cross = e_xc_total - two_e_act.exc - two_e_env.exc

        # overall two_electron cross energy
        self.two_e_cross = j_cross + k_cross + xc_cross

        logger.debug("RKS components")
        logger.debug(f"e_act: {self.e_act}")
        logger.debug(f"e_env: {self.e_env}")
        logger.debug(f"two_e_cross: {self.two_e_cross}")
        logger.debug(f"e_nuc: {self._global_ks.energy_nuc()}")

    @cached_property
    def _env_projector(self) -> np.ndarray:
        """Return a projector onto the environment in orthogonal basis."""
        s_mat = self._global_ks.get_ovlp()
        env_projector_alpha = s_mat @ self.localized_system.dm_enviro @ s_mat

        if self._restricted_scf:
            env_projector = env_projector_alpha

        else:
            env_projector_beta = s_mat @ self.localized_system.beta_dm_enviro @ s_mat
            env_projector = np.array([env_projector_alpha, env_projector_beta])

        return env_projector

    def _run_emb_CCSD(
        self,
        emb_pyscf_scf_rhf: Union[scf.RHF, scf.UHF],
        frozen_orb_list: Optional[list] = None,
    ) -> Tuple[cc.CCSD, float]:
        """Function run CCSD on embedded restricted Hartree Fock object.

        Note emb_pyscf_scf_rhf is RHF object for the active embedded subsystem (defined in localized basis)
        (see get_embedded_rhf method)

        Args:
            emb_pyscf_scf_rhf (scf.RHF): PySCF restricted Hartree Fock object of active embedded subsystem
            frozen_orb_list (List): A path to an .xyz file describing molecular geometry.

        Returns:
            ccsd (cc.CCSD): PySCF CCSD object
            e_ccsd_corr (float): electron correlation CCSD energy
        """
        logger.debug("Starting embedded CCSD calculation.")
        ccsd = cc.CCSD(emb_pyscf_scf_rhf)
        ccsd.conv_tol = self.convergence
        ccsd.max_memory = self.max_ram_memory
        ccsd.verbose = self.pyscf_print_level

        # Set which orbitals are to be frozen
        ccsd.frozen = frozen_orb_list
        e_ccsd_corr, _, _ = ccsd.kernel()
        logger.info(f"Embedded CCSD energy: {e_ccsd_corr}")
        return ccsd, e_ccsd_corr

    def _run_emb_FCI(
        self,
        emb_pyscf_scf_rhf: Union[scf.RHF, scf.UHF],
        frozen_orb_list: Optional[list] = None,
    ) -> fci.FCI:
        """Function run FCI on embedded restricted Hartree Fock object.

        Note emb_pyscf_scf_rhf is RHF object for the active embedded subsystem (defined in localized basis)
        (see get_embedded_rhf method)

        Args:
            emb_pyscf_scf_rhf (scf.RHF): PySCF restricted Hartree Fock object of active embedded subsystem
            frozen_orb_list (List): A path to an .xyz file describing moleclar geometry.

        Returns:
            fci_scf (fci.FCI): PySCF FCI object
        """
        logger.debug("Starting embedded FCI calculation.")
        fci_scf = fci.FCI(emb_pyscf_scf_rhf)
        fci_scf.conv_tol = self.convergence
        fci_scf.verbose = self.pyscf_print_level
        fci_scf.max_memory = self.max_ram_memory

        fci_scf.frozen = frozen_orb_list
        fci_scf.run()
        logger.info(f"FCI embedding energy: {fci_scf.e_tot}")
        return fci_scf

    def _mu_embed(
        self, localized_scf: StreamObject, dft_potential: np.ndarray
    ) -> Tuple[StreamObject, np.ndarray]:
        """Embed using the Mu-shift projector.

        Args:
            active_scf (StreamObject): A PySCF scf method with the correct number of electrons for the active region.
            dft_potential (np.ndarray): Potential calculated from two electron terms in dft.

        Returns:
            np.ndarray: Matrix form of the embedding potential.
            StreamObject: The embedded scf object.
        """
        logger.debug("Running embedded scf calculation.")

        # Modify the energy_elec function to handle different h_cores
        # which we need for different embedding potentials
        if isinstance(localized_scf, (scf.uhf.UHF, dft.uks.UKS)):
            localized_scf.energy_elec = lambda *args: energy_elec(localized_scf, *args)
            v_emb_alpha = (
                self.mu_level_shift * self._env_projector[0]
            ) + dft_potential[0]
            v_emb_beta = (self.mu_level_shift * self._env_projector[1]) + dft_potential[
                1
            ]
            v_emb = np.array([v_emb_alpha, v_emb_beta])

            hcore_std = localized_scf.get_hcore()
            localized_scf.get_hcore = (
                lambda *args: np.array([hcore_std, hcore_std]) + v_emb
            )
        elif isinstance(localized_scf, (scf.rhf.RHF, dft.rks.RKS)):
            # modify hcore to embedded version
            v_emb = (self.mu_level_shift * self._env_projector) + dft_potential

            hcore_std = localized_scf.get_hcore()
            localized_scf.get_hcore = lambda *args: hcore_std + v_emb
        else:
            logger.error(f"Invalid scf object of type {type(localized_scf)}.")
            raise NbedConfigError("Invalid scf object.")

        localized_scf.kernel()
        logger.info(
            f"Embedded scf energy MU_SHIFT: {localized_scf.e_tot}, converged: {localized_scf.converged}"
        )
        self.v_emb = v_emb

        return localized_scf, v_emb

    def _huzinaga_embed(
        self,
        active_scf: StreamObject,
        dft_potential: np.ndarray,
        dmat_initial_guess: bool = None,
    ) -> Tuple[StreamObject, np.ndarray]:
        """Embed using Huzinaga projector.

        Args:
            active_scf (StreamObject): A PySCF scf method with the correct number of electrons for the active region.
            dft_potential (np.ndarray): Potential calculated from two electron terms in dft.
            dmat_initial_guess (bool): If True, use the initial guess for the density matrix.

        Returns:
            np.ndarray: Matrix form of the embedding potential.
            StreamObject: The embedded scf object.
        """
        logger.debug("Starting Huzinaga embedding method.")
        # We need to run our own SCF method here to update the potential.
        if self._restricted_scf:
            total_enviro_dm = self.localized_system.dm_enviro
        else:
            total_enviro_dm = np.array(
                [self.localized_system.dm_enviro, self.localized_system.beta_dm_enviro]
            )

        localized_scf = active_scf
        if isinstance(localized_scf, (dft.rks.RKS, dft.uks.UKS)):
            huz_method = huzinaga_KS
        elif isinstance(localized_scf, (scf.rhf.RHF, scf.uhf.UHF)):
            huz_method = huzinaga_HF

        (
            c_active_embedded,
            mo_embedded_energy,
            dm_active_embedded,
            huzinaga_op_std,
            huz_scf_conv_flag,
        ) = huz_method(
            localized_scf,
            dft_potential,
            total_enviro_dm,
            dm_conv_tol=1e-6,
            dm_initial_guess=dmat_initial_guess,
        )

        logger.debug(f"{c_active_embedded=}")

        # write results to pyscf object
        logger.debug("Writing results to PySCF object.")
        hcore_std = localized_scf.get_hcore()
        v_emb = huzinaga_op_std + dft_potential
        localized_scf.get_hcore = lambda *args: hcore_std + v_emb

        if not self._restricted_scf:
            localized_scf.energy_elec = lambda *args: energy_elec(localized_scf, *args)

        localized_scf.mo_coeff = c_active_embedded
        localized_scf.mo_occ = localized_scf.get_occ(
            mo_embedded_energy, c_active_embedded
        )
        localized_scf.mo_energy = mo_embedded_energy
        localized_scf.e_tot = localized_scf.energy_tot(dm=dm_active_embedded)
        # localized_scf.conv_check = huz_scf_conv_flag
        localized_scf.converged = huz_scf_conv_flag

        logger.info(f"Huzinaga scf energy: {localized_scf.e_tot}")
        return localized_scf, v_emb

    def _delete_spin_environment(
        self,
        method: str,
        n_env_mo: int,
        mo_coeff: np.ndarray,
        mo_energy: np.ndarray,
        mo_occ: np.ndarray,
        projector,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove enironment orbit from embedded rhf object.

        This function removes (in fact deletes completely) the molecular orbitals
        defined by the environment of the localized system

        Args:
            method (str): The localization method used to embed the system. 'huzinaga' or 'mu'.
            n_env_mo (int): The number of molecular orbitals in the environment.
            mo_coeff (np.ndarray): The molecular orbitals.
            mo_energy (np.ndarray): The molecular orbital energies.
            mo_occ (np.ndarray): The molecular orbital occupation numbers.

        Returns:
            embedded_rhf (StreamObject): Returns input, but with environment orbitals deleted
        """
        logger.debug("Deleting environment for spin.")

        if method == "huzinaga":
            overlap = np.einsum(
                "ij, ki -> i",
                mo_coeff.T,
                projector @ mo_coeff,
            )
            overlap_by_size = overlap.argsort()[::-1]
            frozen_enviro_orb_inds = overlap_by_size[:n_env_mo]

        elif method == "mu":
            shift = mo_coeff.shape[1] - n_env_mo
            frozen_enviro_orb_inds = [mo_i for mo_i in range(shift, mo_coeff.shape[1])]

        active_MOs_occ_and_virt_embedded = [
            mo_i
            for mo_i in range(mo_coeff.shape[1])
            if mo_i not in frozen_enviro_orb_inds
        ]

        logger.info(
            f"Orbital indices for embedded system: {active_MOs_occ_and_virt_embedded}"
        )
        logger.info(
            f"Orbital indices removed from embedded system: {frozen_enviro_orb_inds}"
        )

        # delete enviroment orbitals and associated energies
        # overwrites varibles keeping only active part (both occupied and virtual)
        active_mo_coeff = mo_coeff[:, active_MOs_occ_and_virt_embedded]
        active_mo_energy = mo_energy[active_MOs_occ_and_virt_embedded]
        active_mo_occ = mo_occ[active_MOs_occ_and_virt_embedded]

        logger.debug("Spin environment deleted.")
        return active_mo_coeff, active_mo_energy, active_mo_occ

    def _delete_environment(self, method: str, scf: StreamObject) -> StreamObject:
        """Remove enironment orbit from embedded rhf object.

        This function removes (in fact deletes completely) the molecular orbitals
        defined by the environment of the localized system

        Args:
            method (str): The localization method used to embed the system. 'huzinaga' or 'mu'.
            scf (StreamObject): The embedded SCF object.

        Returns:
            StreamObject: Returns input, but with environment orbitals deleted.
        """
        logger.debug("Deleting environment from SCF object.")

        if self._restricted_scf:
            n_env_mos = len(self.localized_system.enviro_MO_inds)
            scf.mo_coeff, scf.mo_energy, scf.mo_occ = self._delete_spin_environment(
                method,
                n_env_mos,
                scf.mo_coeff,
                scf.mo_energy,
                scf.mo_occ,
                self._env_projector,
            )
        else:
            alpha_n_env_mos = len(self.localized_system.enviro_MO_inds)
            beta_n_env_mos = len(self.localized_system.beta_enviro_MO_inds)
            mo_coeff = np.array([None, None])
            mo_energy = np.array([None, None])
            mo_occ = np.array([None, None])
            (mo_coeff[0], mo_energy[0], mo_occ[0],) = self._delete_spin_environment(
                method,
                alpha_n_env_mos,
                scf.mo_coeff[0],
                scf.mo_energy[0],
                scf.mo_occ[0],
                self._env_projector[0],
            )
            (mo_coeff[1], mo_energy[1], mo_occ[1]) = self._delete_spin_environment(
                method,
                beta_n_env_mos,
                scf.mo_coeff[1],
                scf.mo_energy[1],
                scf.mo_occ[1],
                self._env_projector[1],
            )

            # Need to do it this way or there are broadcasting issues
            scf.mo_coeff = mo_coeff  # np.array([mo_coeff[0], mo_coeff[1]])
            scf.mo_energy = mo_energy  # np.array([mo_energy[0], mo_energy[1]])
            scf.mo_occ = mo_occ  # np.array([mo_occ[0], mo_occ[1]])

        logger.debug("Environment deleted.")
        return scf

    def _dft_in_dft(self, xc_func: str, embedding_method: Callable):
        """Return energy of DFT in DFT embedding.

        Note run_mu_shift (bool) and run_huzinaga (bool) flags define which method to use (can be both)
        This is done when object is initialized.

        Args:
            xc_func (str): XC functional to use in active region.
            embedding_method (callable): Embedding method to use (mu or huzinaga).

        Returns:
            float: Energy of DFT in embedding.
        """
        result = {}
        e_nuc = self._global_ks.energy_nuc()

        local_rks_same_functional = self._init_local_ks(xc_func)
        hcore_std = local_rks_same_functional.get_hcore()
        result["scf_dft"], result["v_emb_dft"] = embedding_method(
            local_rks_same_functional, self.dft_potential
        )

        if not self._restricted_scf:
            y_emb_alpha, y_emb_beta = result["scf_dft"].make_rdm1()

            # calculate correction
            result["dft_correction"] = np.einsum(
                "ij,ij",
                result["v_emb_dft"][0],
                (y_emb_alpha - self.localized_system.dm_active),
            )

            result["dft_correction_beta"] = np.einsum(
                "ij,ij",
                result["v_emb_dft"][1],
                (y_emb_beta - self.localized_system.beta_dm_active),
            )

            veff = result["scf_dft"].get_veff(dm=[y_emb_alpha, y_emb_beta])

            rks_e_elec = (
                veff.exc
                + veff.ecoul
                + np.einsum(
                    "ij,ij",
                    hcore_std,
                    y_emb_alpha,
                )
                + np.einsum(
                    "ij,ij",
                    hcore_std,
                    y_emb_beta,
                )
            )

        else:
            y_emb = result["scf_dft"].make_rdm1()

            # calculate correction
            result["dft_correction"] = np.einsum(
                "ij,ij",
                result["v_emb_dft"],
                (y_emb - self.localized_system.dm_active),
            )
            veff = result["scf_dft"].get_veff(dm=y_emb)
            result["dft_correction_beta"] = 0
            rks_e_elec = veff.exc + veff.ecoul + np.einsum("ij,ij", hcore_std, y_emb)

        result["e_rks"] = (
            rks_e_elec
            + self.e_env
            + self.two_e_cross
            + result["dft_correction"]
            + result["dft_correction_beta"]
            + e_nuc
        )
        result["rks_e_elec"] = rks_e_elec

        return result

    def embed(self, init_huzinaga_rhf_with_mu=False):
        """Run embedded scf calculation.

        Note run_mu_shift (bool) and run_huzinaga (bool) flags define which method to use (can be both)
        This is done when object is initialized.
        """
        logger.debug("Embedding molecule.")
        self.localized_system = self._localize()
        logger.info("Indices of embedded electrons:")
        logger.info(self.localized_system.active_MO_inds)
        logger.info(self.localized_system.enviro_MO_inds)

        e_nuc = self._global_ks.energy_nuc()

        # Run subsystem DFT (calls localized rks)
        self._subsystem_dft()
        logger.debug("Getting global DFT potential to optimize embedded calc in.")

        total_dm = self.localized_system.dm_active + self.localized_system.dm_enviro
        if not self._restricted_scf:
            total_dm += (
                self.localized_system.beta_dm_active
                + self.localized_system.beta_dm_enviro
            )

        g_act_and_env = self._global_ks.get_veff(dm=total_dm)

        self.total_dm = total_dm
        self.g_act_and_env = g_act_and_env

        if self._restricted_scf:
            g_act = self._global_ks.get_veff(dm=self.localized_system.dm_active)
        else:
            g_act = self._global_ks.get_veff(
                dm=[
                    self.localized_system.dm_active,
                    self.localized_system.beta_dm_active,
                ]
            )
        self.g_act = g_act

        self.dft_potential = g_act_and_env - g_act
        logger.info(f"DFT potential average {np.mean(self.dft_potential)}.")

        # To add a projector, put it in this dict with a function
        # if we want any more we could consider adding classes
        # or using a match statement for >python3.10
        embeddings: Dict[str, callable] = {
            "huzinaga": self._huzinaga_embed,
            "mu": self._mu_embed,
        }
        # If only one projector is specified, remove the others.
        if self.projector != "both":
            embeddings = {self.projector: embeddings[self.projector]}

        # This is reverse so that huz can be initialised with mu
        for name in sorted(embeddings, reverse=True):
            logger.debug(f"Runnning embedding with {name} projector.")
            setattr(self, "_" + name, {})
            result = getattr(self, "_" + name)

            embedding_method: callable = embeddings[name]

            local_hf = self._init_local_hf()

            if init_huzinaga_rhf_with_mu and (name == "huzinaga"):
                logger.debug("Initializing huzinaga with mu-shift.")
                # seed huzinaga calc with mu result!
                result["scf"], result["v_emb"] = embedding_method(
                    local_hf,
                    self.dft_potential,
                    dmat_initial_guess=self._mu["scf"].make_rdm1(),
                )
            else:
                result["scf"], result["v_emb"] = embedding_method(
                    local_hf, self.dft_potential
                )

            result["mo_energies_emb_pre_del"] = local_hf.mo_energy
            result["scf"] = self._delete_environment(name, result["scf"])
            result["mo_energies_emb_post_del"] = local_hf.mo_energy

            logger.info(f"V emb mean {name}: {np.mean(result['v_emb'])}")

            # calculate correction
            if self._restricted_scf:
                result["correction"] = np.einsum(
                    "ij,ij", result["v_emb"], self.localized_system.dm_active
                )
                result["beta_correction"] = 0
            else:
                result["correction"] = np.einsum(
                    "ij,ij", result["v_emb"][0], self.localized_system.dm_active
                )
                result["beta_correction"] = np.einsum(
                    "ij,ij", result["v_emb"][1], self.localized_system.beta_dm_active
                )

            result["e_rhf"] = (
                result["scf"].e_tot
                + self.e_env
                + self.two_e_cross
                - result["correction"]
                - result["beta_correction"]
            )
            logger.info(f"RHF energy: {result['e_rhf']}")

            # classical energy
            result["classical_energy"] = (
                self.e_env
                + self.two_e_cross
                + e_nuc
                - result["correction"]
                - result["beta_correction"]
            )
            logger.debug(f"Classical energy: {result['classical_energy']}")

            # Virtual localization
            if self.run_virtual_localization is True:
                logger.debug("Performing virtual localization.")
                self.localized_system.localize_virtual(result["scf"])

            # Calculate ccsd or fci energy
            if self.run_ccsd_emb is True:
                logger.debug("Performing CCSD-in-DFT embedding.")
                ccsd_emb, e_ccsd_corr = self._run_emb_CCSD(
                    result["scf"], frozen_orb_list=None
                )
                result["e_ccsd"] = (
                    ccsd_emb.e_hf
                    - ccsd_emb.e_corr
                    + self.e_env
                    + self.two_e_cross
                    - result["correction"]
                    - result["beta_correction"]
                )
                result["ccsd_emb"] = ccsd_emb.e_hf - e_ccsd_corr - e_nuc

                logger.info(f"CCSD Energy {name}:\t{result['e_ccsd']}")

            if self.run_fci_emb is True:
                logger.debug("Performing FCI-in-DFT embedding.")
                fci_emb = self._run_emb_FCI(result["scf"], frozen_orb_list=None)
                result["e_fci"] = (
                    (fci_emb.e_tot)
                    + self.e_env
                    + self.two_e_cross
                    - result["correction"]
                    - result["beta_correction"]
                )
                logger.info(f"FCI Energy {name}:\t{result['e_fci']}")

                result["fci_emb"] = fci_emb.e_tot - e_nuc
            result["hf_emb"] = result["scf"].e_tot - e_nuc
            result["nuc"] = e_nuc

            if self.run_dft_in_dft is True:
                did = self._dft_in_dft(self._global_ks.xc, embedding_method)
                result["e_dft_in_dft"] = did["e_rks"]
                result["emb_dft"] = did["rks_e_elec"]

        if self.projector == "both":
            logger.warning(
                "Outputting both mu and huzinaga embedding results as tuple."
            )
            self.embedded_scf = (
                self._mu["scf"],
                self._huzinaga["scf"],
            )
            self.classical_energy = (
                self._mu["classical_energy"],
                self._huzinaga["classical_energy"],
            )
        elif self.projector == "mu":
            self.embedded_scf = self._mu["scf"]
            self.classical_energy = self._mu["classical_energy"]
        elif self.projector == "huzinaga":
            self.embedded_scf = self._huzinaga["scf"]
            self.classical_energy = self._huzinaga["classical_energy"]

        logger.info("Embedding complete.")
