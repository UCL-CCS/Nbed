"""Module containg the NbedDriver Class."""

import logging
import os
from copy import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy as sp
from cached_property import cached_property
from pyscf import cc, fci, gto, scf
from pyscf.lib import StreamObject

from nbed.exceptions import NbedConfigError

from .localizers import (
    BOYSLocalizer,
    IBOLocalizer,
    Localizer,
    PMLocalizer,
    SPADELocalizer,
)
from .scf import huzinaga_RHF

logger = logging.getLogger(__name__)


class NbedDriver:
    """Function to return the embedding Qubit Hamiltonian.

    Args:
        geometry (str): Path to .xyz file containing molecular geometry or raw xyz string.
        n_active_atoms (int): The number of atoms to include in the active region.
        basis (str): The name of an atomic orbital basis set to use for chemistry calculations.
        xc_functonal (str): The name of an Exchange-Correlation functional to be used for DFT.
        projector (str):
        localization (str): Orbital localization method to use. One of 'spade', 'pipek-mezey', 'boys' or 'ibo'.
        convergence (float): The convergence tolerance for energy calculations.
        charge (int): Charge of molecular species
        mu_level_shift (float): Level shift parameter to use for mu-projector.
        run_ccsd_emb (bool): Whether or not to find the CCSD energy of embbeded system for reference.
        run_fci_emb (bool): Whether or not to find the FCI energy of embbeded system for reference.
        max_ram_memory (int): Amount of RAM memery in MB available for PySCF calculation
        pyscf_print_level (int): Amount of information PySCF prints
        unit (str): molecular geometry unit 'Angstrom' or 'Bohr'
        max_hf_cycles (int): max number of Hartree-Fock iterations allowed (for global and local HFock)
        _init_huzinaga_rhf_with_mu (bool): Hidden flag to seed huzinaga RHF with mu shift result (for developers only)

    Attributes:
        _global_fci (StreamObject): A Qubit Hamiltonian of some kind
        e_act (float): Active energy from subsystem DFT calculation
        e_env (float): Environment energy from subsystem DFT calculation
        two_e_cross (flaot): two electron energy from cross terms (includes exchange correlation
                             and Coloumb contribution) of subsystem DFT calculation
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
        mu_level_shift: Optional[float] = 1e6,
        run_ccsd_emb: Optional[bool] = False,
        run_fci_emb: Optional[bool] = False,
        run_virtual_localization: Optional[bool] = False,
        max_ram_memory: Optional[int] = 4000,
        pyscf_print_level: int = 1,
        savefile: Optional[Path] = None,
        unit: Optional[str] = "angstrom",
        occupied_threshold: Optional[float] = 0.95,
        virtual_threshold: Optional[float] = 0.95,
        _init_huzinaga_rhf_with_mu: bool = False,
        max_hf_cycles: int =50
    ):
        """Initialise class."""
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
            raise NbedConfigError("Invalid config.")

        self.geometry = geometry
        self.n_active_atoms = n_active_atoms
        self.basis = basis.lower()
        self.xc_functional = xc_functional.lower()
        self.projector = projector.lower()
        self.localization = localization.lower()
        self.convergence = convergence
        self.charge = charge
        self.mu_level_shift = mu_level_shift
        self.run_ccsd_emb = run_ccsd_emb
        self.run_fci_emb = run_fci_emb
        self.run_virtual_localization = run_virtual_localization
        self.max_ram_memory = max_ram_memory
        self.pyscf_print_level = pyscf_print_level
        self.savefile = savefile
        self.unit = unit
        self.occupied_threshold = occupied_threshold
        self.virtual_threshold = virtual_threshold
        self.max_hf_cycles = max_hf_cycles

        self._check_active_atoms()

        self.embed(init_huzinaga_rhf_with_mu=_init_huzinaga_rhf_with_mu)

    def _build_mol(self) -> gto.mole:
        """Function to build PySCF molecule.

        Returns:
            full_mol (gto.mol): built PySCF molecule object
        """
        logger.debug("Construcing molecule.")
        if os.path.exists(self.geometry):
            # geometry is an xyz file
            full_mol = gto.Mole(
                atom=self.geometry, basis=self.basis, charge=self.charge, unit=self.unit
            ).build()
        else:
            # geometry is raw xyz string
            full_mol = gto.Mole(
                atom=self.geometry[3:],
                basis=self.basis,
                charge=self.charge,
                unit=self.unit,
            ).build()
        return full_mol

    @cached_property
    def _global_hf(self) -> StreamObject:
        """Run full system Hartree-Fock."""
        mol_full = self._build_mol()
        # run Hartree-Fock
        global_hf = scf.RHF(mol_full)
        global_hf.conv_tol = self.convergence
        global_hf.max_memory = self.max_ram_memory
        global_hf.verbose = self.pyscf_print_level
        global_hf.max_cycle = self.max_hf_cycles
        global_hf.kernel()
        logger.info(f"global HF: {global_hf.e_tot}")
        return global_hf

    @cached_property
    def _global_fci(self) -> StreamObject:
        """Function to run full molecule FCI calculation. FACTORIAL SCALING IN BASIS STATES!"""
        # run FCI after HF
        global_fci = fci.FCI(self._global_hf)
        global_fci.conv_tol = self.convergence
        global_fci.verbose = self.pyscf_print_level
        global_fci.max_memory = self.max_ram_memory
        global_fci.run()
        logger.info(f"global FCI: {global_fci.e_tot}")

        return global_fci

    @cached_property
    def _global_rks(self):
        """Method to run full cheap molecule RKS DFT calculation.

        Note this is necessary to perform localization procedure.
        """
        mol_full = self._build_mol()

        global_rks = scf.RKS(mol_full)
        global_rks.conv_tol = self.convergence
        global_rks.xc = self.xc_functional
        global_rks.max_memory = self.max_ram_memory
        global_rks.verbose = self.pyscf_print_level
        global_rks.kernel()
        logger.info(f"global RKS {global_rks.e_tot}")

        return global_rks

    def _check_active_atoms(self):
        """Check that the number of active atoms is valid."""
        all_atoms = self._build_mol().natm
        if self.n_active_atoms not in range(1, all_atoms):
            raise NbedConfigError(
                f"Invalid number of active atoms. Choose a number between 0 and {all_atoms}."
            )

    def localize(self):
        """Run the localizer class."""
        logger.debug(f"Getting localized system using {self.localization}.")

        localizers = {
            "spade": SPADELocalizer,
            "boys": BOYSLocalizer,
            "ibo": IBOLocalizer,
            "pipek-mezey": PMLocalizer,
        }

        # Should already be validated.
        localized_system = localizers[self.localization](
            self._global_rks,
            self.n_active_atoms,
            occ_cutoff=self.occupied_threshold,
            virt_cutoff=self.virtual_threshold,
            run_virtual_localization=self.run_virtual_localization,
        )
        return localized_system

    def _init_local_rhf(self) -> scf.RHF:
        """Function to build embedded restricted Hartree Fock object for active subsystem.

        Note this function overwrites the total number of electrons to only include active number
        """
        embedded_mol: gto.Mole = self._build_mol()

        # overwrite total number of electrons to only include active system
        embedded_mol.nelectron = 2 * len(self.localized_system.active_MO_inds)

        logger.debug("Define Hartree-Fock object")
        local_rhf: StreamObject = scf.RHF(embedded_mol)
        local_rhf.max_memory = self.max_ram_memory
        local_rhf.conv_tol = self.convergence
        local_rhf.verbose = self.pyscf_print_level
        local_rhf.max_cycle = self.max_hf_cycles

        return local_rhf

    def _subsystem_dft(self):
        """Function to perform subsystem RKS DFT calculation."""
        logger.debug("Calculating active and environment subsystem terms.")

        def _rks_components(
            rks_system: Localizer, subsystem_dm: np.ndarray,
        ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
            """Calculate the components of subsystem energy from a RKS DFT calculation.

            For a given density matrix this function returns the electronic energy, exchange correlation energy and
            J,K, V_xc matrices.

            Args:
                dm_matrix (np.ndarray): density matrix (to calculate all matrices from)

            Returns:
                Energy_elec (float): DFT energy defubed by input density matrix
                e_xc (float): exchange correlation energy defined by input density matrix
                J_mat (np.ndarray): J_matrix defined by input density matrix
            """
            dm_matrix = subsystem_dm
            # It seems that PySCF lumps J and K in the J array
            two_e_term = rks_system.get_veff(dm=dm_matrix)
            j_mat = two_e_term.vj
            # k_mat = np.zeros_like(j_mat)

            e_xc = two_e_term.exc
            # v_xc = two_e_term - j_mat

            energy_elec = (
                np.einsum("ij,ji->", rks_system.get_hcore(), dm_matrix)
                + two_e_term.ecoul
                + two_e_term.exc
            )

            # if check_E_with_pyscf:
            #     energy_elec_pyscf = self._global_rks.energy_elec(dm=dm_matrix)[0]
            #     if not np.isclose(energy_elec_pyscf, energy_elec):
            #         raise ValueError("Energy calculation incorrect")

            return energy_elec, e_xc, j_mat

        (self.e_act, e_xc_act, j_act) = _rks_components(
            self._global_rks, self.localized_system.dm_active
        )
        (self.e_env, e_xc_env, j_env) = _rks_components(
            self._global_rks, self.localized_system.dm_enviro
        )
        # Computing cross subsystem terms
        logger.debug("Calculating two electron cross subsystem energy.")

        two_e_term_total = self._global_rks.get_veff(
            dm=self.localized_system.dm_active + self.localized_system.dm_enviro
        )
        e_xc_total = two_e_term_total.exc

        j_cross = 0.5 * (
            np.einsum("ij,ij", self.localized_system.dm_active, j_env)
            + np.einsum("ij,ij", self.localized_system.dm_enviro, j_act)
        )
        # Because of projection
        k_cross = 0.0

        xc_cross = e_xc_total - e_xc_act - e_xc_env

        # overall two_electron cross energy
        self.two_e_cross = j_cross + k_cross + xc_cross

        energy_DFT_components = (
            self.e_act + self.e_env + self.two_e_cross + self._global_rks.energy_nuc()
        )
        logger.info("RKS components")
        logger.info(self.e_act)
        logger.info(self.e_env)
        logger.info(self.two_e_cross)
        logger.info(self._global_rks.energy_nuc())
        if not np.isclose(energy_DFT_components, self._global_rks.e_tot):

            raise ValueError(
                "DFT energy of localized components not matching supersystem DFT"
            )

        return None

    def _init_local_rks(self) -> scf.RKS:
        """Function to build embedded restricted Hartree Fock object for active subsystem.

        Note this function overwrites the total number of electrons to only include active number
        """
        embedded_mol: gto.Mole = self._build_mol()

        # overwrite total number of electrons to only include active system
        embedded_mol.nelectron = 2 * len(self.localized_system.active_MO_inds)

        logger.debug("Define Hartree-Fock object")
        local_rks: StreamObject = scf.RKS(embedded_mol)
        local_rks.max_memory = self.max_ram_memory
        local_rks.conv_tol = self.convergence
        local_rks.verbose = self.pyscf_print_level
        local_rks.xc = self.xc_functional

        return local_rks

    @cached_property
    def _env_projector(self):
        """Return a projector onto the environment in orthogonal basis."""
        s_mat = self._global_rks.get_ovlp()
        env_projector = s_mat @ self.localized_system.dm_enviro @ s_mat
        return env_projector

    def _run_emb_CCSD(
        self, emb_pyscf_scf_rhf: scf.RHF, frozen_orb_list: Optional[list] = None
    ) -> Tuple[cc.CCSD, float]:
        """Function run CCSD on embedded restricted Hartree Fock object.

        Note emb_pyscf_scf_rhf is RHF object for the active embedded subsystem (defined in localized basis)
        (see get_embedded_rhf method)

        Args:
            emb_pyscf_scf_rhf (scf.RHF): PySCF restricted Hartree Fock object of active embedded subsystem
            frozen_orb_list (List): A path to an .xyz file describing moleclar geometry.

        Returns:
            ccsd (cc.CCSD): PySCF CCSD object
            e_ccsd_corr (float): electron correlation CCSD energy
        """
        ccsd = cc.CCSD(emb_pyscf_scf_rhf)
        ccsd.conv_tol = self.convergence
        ccsd.max_memory = self.max_ram_memory
        ccsd.verbose = self.pyscf_print_level

        # Set which orbitals are to be frozen
        ccsd.frozen = frozen_orb_list
        e_ccsd_corr, _, _ = ccsd.kernel()
        return ccsd, e_ccsd_corr

    def _run_emb_FCI(
        self, emb_pyscf_scf_rhf: scf.RHF, frozen_orb_list: Optional[list] = None
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
        fci_scf = fci.FCI(emb_pyscf_scf_rhf)
        fci_scf.conv_tol = self.convergence
        fci_scf.verbose = self.pyscf_print_level
        fci_scf.max_memory = self.max_ram_memory

        fci_scf.frozen = frozen_orb_list
        fci_scf.run()
        return fci_scf

    def _mu_embed(self, localized_rhf: StreamObject) -> np.ndarray:
        """Embed using the Mu-shift projector.

        Args:
            localized_rhf (StreamObject): A PySCF RHF method in the localized basis.

        Returns:
            np.ndarray: Matrix form of the embedding potential.
            StreamObject: The embedded RHF object.
        """
        # modify hcore to embedded version
        v_emb = (self.mu_level_shift * self._env_projector) + self._dft_potential
        hcore_std = localized_rhf.get_hcore()
        localized_rhf.get_hcore = lambda *args: hcore_std + v_emb

        logger.debug("Running embedded RHF calculation.")
        localized_rhf.kernel()
        logger.info(
            f"embedded HF energy MU_SHIFT: {localized_rhf.e_tot}, converged: {localized_rhf.converged}"
        )

        # TODO Can be used for checks
        # dm_active_embedded = localized_rhf.make_rdm1(
        #     mo_coeff=localized_rhf.mo_coeff, mo_occ=localized_rhf.mo_occ
        # )

        return v_emb, localized_rhf

    def _huzinaga_embed(self, localized_rhf: StreamObject) -> np.ndarray:
        """Embed using Huzinaga projector.

        Args:
            localized_rhf (StreamObject): A PySCF RHF method in the localized basis.

        Returns:
            np.ndarray: Matrix form of the embedding potential.
            StreamObject: The embedded RHF object.
        """
        # We need to run manual HF to update
        # Fock matrix with each cycle
        (
            c_active_embedded,
            mo_embedded_energy,
            dm_active_embedded,
            huzinaga_op_std,
            huz_rhf_conv_flag
        ) = huzinaga_RHF(
            localized_rhf,
            self._dft_potential,
            self.localized_system.dm_enviro,
            dm_conv_tol=1e-6,
            dm_initial_guess=None,
        )  # TODO: use dm_active_embedded (use mu answer to initialize!)

        # write results to pyscf object
        hcore_std = localized_rhf.get_hcore()
        v_emb = huzinaga_op_std + self._dft_potential
        localized_rhf.get_hcore = lambda *args: hcore_std + v_emb
        localized_rhf.mo_coeff = c_active_embedded
        localized_rhf.mo_occ = localized_rhf.get_occ(
            mo_embedded_energy, c_active_embedded
        )
        localized_rhf.mo_energy = mo_embedded_energy
        localized_rhf.e_tot = localized_rhf.energy_tot(dm=dm_active_embedded)
        # localized_rhf.conv_check = huz_rhf_conv_flag
        localized_rhf.converged = huz_rhf_conv_flag

        logger.info(f"Huzinaga rhf energy: {localized_rhf.e_tot}")
        return v_emb, localized_rhf

    def _delete_environment(self, embedded_rhf, method: str) -> np.ndarray:
        """Remove enironment orbit from embedded rhf object. This function removes (in fact deletes completely) the
        molecular orbitals defined by the environment (defined by the environment of the localized system)

        Args:
            embedded_rhf (StreamObject): A PySCF RHF method in the localized basis.

        Returns:
            embedded_rhf (StreamObject): Returns input, but with environment orbitals deleted
        """
        n_env_mo = len(self.localized_system.enviro_MO_inds)

        if method == "huzinaga":
            # find <psi  | Proj| psi > =  <psi  |  psi_proj >
            # delete orbs with most overlap (as has large overlap with env)
            overlap = np.einsum('ij, ki -> i', embedded_rhf.mo_coeff.T, self._env_projector @ embedded_rhf.mo_coeff)
            overlap_by_size = overlap.argsort()[::-1]
            frozen_enviro_orb_inds = overlap_by_size[:n_env_mo]

            active_MOs_occ_and_virt_embedded = [
                mo_i
                for mo_i in range(embedded_rhf.mo_coeff.shape[1])
                if mo_i not in frozen_enviro_orb_inds
            ]

        elif method == "mu":
            shift = embedded_rhf.mol.nao - n_env_mo
            frozen_enviro_orb_inds = [
                mo_i for mo_i in range(shift, embedded_rhf.mol.nao)
            ]
            active_MOs_occ_and_virt_embedded = [
                mo_i
                for mo_i in range(embedded_rhf.mo_coeff.shape[1])
                if mo_i not in frozen_enviro_orb_inds
            ]

        else:
            raise ValueError("Must use mu or huzinaga flag.")

        logger.info(
            f"Orbital indices for embedded system {active_MOs_occ_and_virt_embedded}"
        )
        logger.info(
            f"Orbital indices removed from embedded system {frozen_enviro_orb_inds}"
        )

        # delete enviroment orbitals and associated energies
        # overwrites varibles keeping only active part (both occupied and virtual)
        embedded_rhf.mo_coeff = embedded_rhf.mo_coeff[
            :, active_MOs_occ_and_virt_embedded
        ]
        embedded_rhf.mo_energy = embedded_rhf.mo_energy[
            active_MOs_occ_and_virt_embedded
        ]
        embedded_rhf.mo_occ = embedded_rhf.mo_occ[active_MOs_occ_and_virt_embedded]

        return embedded_rhf

    def embed(self, init_huzinaga_rhf_with_mu=False):
        """Generate embedded Hamiltonian.

        Note run_mu_shift (bool) and run_huzinaga (bool) flags define which method to use (can be both)
        This is done when object is initialized.
        """
        logger.debug("Running orb localization on global DFT object.")
        self.localized_system = self.localize()

        e_nuc = self._global_rks.energy_nuc()

        # Run subsystem DFT (calls localized rks)
        self._subsystem_dft()

        logger.debug("Get global DFT potential to optimize embedded calc in.")
        g_act_and_env = self._global_rks.get_veff(
            dm=(self.localized_system.dm_active + self.localized_system.dm_enviro)
        )
        g_act = self._global_rks.get_veff(dm=self.localized_system.dm_active)
        self._dft_potential = g_act_and_env - g_act
        logger.info(f"DFT potential average {np.mean(self._dft_potential)}")

        embeddings: Dict[str, callable] = {
            "huzinaga": self._huzinaga_embed,
            "mu": self._mu_embed,
        }
        if self.projector not in ["huzinaga", "both"]:
            embeddings.pop("huzinaga")
        if self.projector not in ["mu", "both"]:
            embeddings.pop("mu")

        self._mu = {}
        self._huzinaga = {}
        for name, embedding_method in embeddings.items():
            local_rhf = self._init_local_rhf()
            result = getattr(self, "_" + name)

            if init_huzinaga_rhf_with_mu and (name == 'huzinaga'):
                # seed huzinaga calc with mu result!
                result["v_emb"], result["scf"] = embedding_method(local_rhf,
                                                                  dm_initial_guess=self._mu['scf'].make_rdm1())
            else:
                result["v_emb"], result["scf"] = embedding_method(local_rhf)

            result["scf"] = self._delete_environment(result["scf"], name)

            logger.info(f"V emb mean {name}: {np.mean(result['v_emb'])}")

            # calculate correction
            result["correction"] = np.einsum(
                "ij,ij", result["v_emb"], self.localized_system.dm_active
            )
            result["e_rhf"] = (
                result["scf"].e_tot
                + self.e_env
                + self.two_e_cross
                - result["correction"]
            )

            # classical energy
            result["classical_energy"] = (
                self.e_env + self.two_e_cross + e_nuc - result["correction"]
            )

            # Calculate ccsd or fci energy
            if self.run_ccsd_emb is True:
                ccsd_emb, e_ccsd_corr = self._run_emb_CCSD(
                    result["scf"], frozen_orb_list=None
                )
                result["e_ccsd"] = (
                    ccsd_emb.e_hf
                    + e_ccsd_corr
                    + self.e_env
                    + self.two_e_cross
                    - result["correction"]
                )
                logger.info(f"CCSD Energy {name}:\n\t{result['e_ccsd']}")

            if self.run_fci_emb is True:
                fci_emb = self._run_emb_FCI(result["scf"], frozen_orb_list=None)
                result["e_fci"] = (
                    (fci_emb.e_tot)
                    + self.e_env
                    + self.two_e_cross
                    - result["correction"]
                )
                logger.info(f"FCI Energy {name}:\n\t{result['e_fci']}")

            ## TODO: Could add DFT in DFT check here (and add better DFT method here to for DFT embedding!)
            # local_DFT_obj = _init_local_rks
            # need to add mu and huz projection schemes here!

        if self.projector == "both":
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

        logger.info(f"num e emb: {2 * len(self.localized_system.active_MO_inds)}")
        logger.info(self.localized_system.active_MO_inds)
        logger.info(self.localized_system.enviro_MO_inds)
