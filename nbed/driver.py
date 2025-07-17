"""Module containg the NbedDriver Class."""

import logging
from functools import cached_property
from json import dump as jdump
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from pyscf import cc, dft, fci, gto, qmmm, scf
from pyscf.lib import StreamObject

from nbed.localizers import (
    BOYSLocalizer,
    ConcentricLocalizer,
    IBOLocalizer,
    OccupiedLocalizer,
    PMLocalizer,
    SPADELocalizer,
)

from .config import LocalizerEnum, NbedConfig, ProjectorEnum
from .ham_builder import HamiltonianBuilder
from .scf import energy_elec
from .scf.huzinaga_scf import huzinaga_scf

# Create the Logger
logger = logging.getLogger(__name__)


class NbedDriver:
    """Function to return the embedding Qubit Hamiltonian.

    Args:
        config (NbedConfig): A validated config model.

    Attributes:
        _global_fci (StreamObject): A Qubit Hamiltonian of some kind
        e_act (float): Active energy from subsystem DFT calculation
        e_env (float): Environment energy from subsystem DFT calculation
        two_e_cross (float): two electron energy from cross terms (includes exchange correlation
                             and Coulomb contribution) of subsystem DFT calculation
        molecular_ham (InteractionOperator): molecular Hamiltonian for active subsystem (projection using mu shift operator)
        classical_energy (float): environment correction energy to obtain total energy (for mu shift method)
        molecular_ham (InteractionOperator): molecular Hamiltonian for active subsystem (projection using huzianga operator)
    """

    def __init__(self, config: NbedConfig):
        """Initialise NbedDriver."""
        logger.debug("Initialising NbedDriver with config:")
        logger.debug(config.model_dump_json())
        self.config = config
        self.localized_system: OccupiedLocalizer
        self.two_e_cross: np.typing.NDArray
        self.electron: int
        self._mu: dict = None
        self._huzinaga: dict = None

        self._restricted_scf = False
        # if config.force_unrestricted:
        #     logger.debug("Forcing unrestricted SCF")
        #     self._restricted_scf = False
        # elif self.config.charge % 2 == 1 or self.config.spin != 0:
        #     logger.debug("Open shells, using unrestricted SCF.")
        #     self._restricted_scf = False
        # else:
        #     logger.debug("Closed shells, using restricted SCF.")
        #     self._restricted_scf = True

        # if we have values for all three, assume we want to run qmmm
        self.run_qmmm = None not in [
            config.mm_charges,
            config.mm_coords,
            config.mm_radii,
        ]

    def _build_mol(self) -> gto.mole:
        """Function to build PySCF molecule.

        Returns:
            full_mol (gto.mol): built PySCF molecule object
        """
        logger.debug("Constructing molecule.")
        logger.info("Molecule input geometry: %s", self.config.geometry)
        # geometry is raw xyz string
        full_mol = gto.Mole(
            atom=self.config.geometry[3:],
            basis=self.config.basis,
            charge=self.config.charge,
            unit=self.config.unit,
            spin=self.config.spin,
        ).build()
        logger.debug("Molecule built.")
        return full_mol

    @cached_property
    def _global_hf(self, **hf_kwargs) -> StreamObject:
        """Run full system Hartree-Fock."""
        logger.debug("Running full system HF.")
        mol_full = self._build_mol()
        # run Hartree-Fock
        global_hf = scf.UHF(mol_full, **hf_kwargs)
        global_hf.conv_tol = self.config.convergence
        global_hf.max_memory = self.config.max_ram_memory
        global_hf.max_cycle = self.config.max_hf_cycles
        global_hf.kernel()
        logger.info(f"Global HF: {global_hf.e_tot}")

        return global_hf

    @cached_property
    def _global_ccsd(self, **ccsd_kwargs) -> StreamObject:
        """Function to run full molecule CCSD calculation."""
        logger.debug("Running full system CC.")
        # run CCSD after HF

        global_cc = cc.CCSD(self._global_hf, **ccsd_kwargs)
        global_cc.conv_tol = self.config.convergence
        global_cc.max_memory = self.config.max_ram_memory
        global_cc.run()
        logger.info(f"Global CCSD: {global_cc.e_tot}")

        return global_cc

    @cached_property
    def _global_fci(self, **fci_kwargs) -> StreamObject:
        """Function to run full molecule FCI calculation.

        WARNING: FACTORIAL SCALING IN BASIS STATES!
        """
        logger.debug("Running full system FCI.")
        # run FCI after HF
        global_fci = fci.FCI(self._global_hf, **fci_kwargs)
        global_fci.conv_tol = self.config.convergence
        global_fci.max_memory = self.config.max_ram_memory
        global_fci.run()
        logger.info(f"Global FCI: {global_fci.e_tot}")

        return global_fci

    @cached_property
    def _global_ks(self, **ks_kwargs) -> StreamObject:
        """Method to run full cheap molecule USK DFT calculation.

        Note this is necessary to perform localization procedure.
        """
        logger.debug("Running full system KS DFT.")
        mol_full = self._build_mol()
        global_ks = dft.UKS(mol_full, **ks_kwargs)
        logger.debug(f"{type(global_ks)=}")
        global_ks.conv_tol = self.config.convergence
        global_ks.xc = self.config.xc_functional
        global_ks.max_memory = self.config.max_ram_memory
        global_ks.max_cycle = self.config.max_dft_cycles

        if self.run_qmmm:
            logger.debug(
                "QM/MM: running full system KS DFT in presence of point charges."
            )
            global_ks = qmmm.mm_charge(
                global_ks,
                self.config.mm_coords,
                self.config.mm_charges,
                self.config.mm_radii,
            )

        global_ks.kernel()
        logger.debug(f"{global_ks.mo_coeff.shape=}")
        logger.debug(f"{global_ks.mo_occ.shape=}")
        logger.debug(f"{global_ks.get_veff().shape=}")
        logger.debug(f"{global_ks.get_hcore().shape=}")
        logger.info(f"Global USK: {global_ks.e_tot}")

        if global_ks.converged is not True:
            logger.warning("(cheap) global DFT calculation has NOT converged!")

        return global_ks

    def _localize(self) -> OccupiedLocalizer:
        """Run the localizer class."""
        logger.debug(f"Getting localized system using {self.config.localization}.")

        match self.config.localization:
            case LocalizerEnum.SPADE:
                return SPADELocalizer(
                    self._global_ks,
                    self.config.n_active_atoms,
                    max_shells=self.config.max_shells,
                    n_mo_overwrite=self.n_mo_overwrite,
                )
            case LocalizerEnum.BOYS:
                return BOYSLocalizer(
                    self._global_ks,
                    self.config.n_active_atoms,
                    occ_cutoff=self.config.occupied_threshold,
                    virt_cutoff=self.config.virtual_threshold,
                )
            case LocalizerEnum.IBO:
                return IBOLocalizer(
                    self._global_ks,
                    self.config.n_active_atoms,
                    occ_cutoff=self.config.occupied_threshold,
                    virt_cutoff=self.config.virtual_threshold,
                )
            case LocalizerEnum.PM:
                return PMLocalizer(
                    self._global_ks,
                    self.config.n_active_atoms,
                    occ_cutoff=self.config.occupied_threshold,
                    virt_cutoff=self.config.virtual_threshold,
                )

    def _init_local_hf(self) -> Union[scf.uhf.UHF, scf.ROHF]:
        """Function to build embedded HF object for active subsystem.

        Note this function overwrites the total number of electrons to only include active number.

        Returns:
            local_hf (scf.uhf.UHF or scf.ROHF): embedded Hartree-Fock object.
        """
        logger.debug("Constructing localised HF object.")
        embedded_mol: gto.Mole = self._build_mol()

        # overwrite total number of electrons to only include active system
        if self.localized_system.beta_active_MO_inds is None:
            embedded_mol.nelectron = 2 * len(self.localized_system.active_MO_inds)
            embedded_mol.spin = 0
            self._electron = embedded_mol.nelectron
            local_hf: scf.UHF = scf.UHF(embedded_mol)
        else:
            embedded_mol.nelectron = len(self.localized_system.active_MO_inds) + len(
                self.localized_system.beta_active_MO_inds
            )
            embedded_mol.spin = len(self.localized_system.active_MO_inds) - len(
                self.localized_system.beta_active_MO_inds
            )
            self._electron = embedded_mol.nelectron
            local_hf: scf.uhf.UHF = scf.UHF(embedded_mol)
        logger.debug(f"{embedded_mol.nelectron=}")
        logger.debug(f"{embedded_mol.spin=}")

        if self.run_qmmm:
            logger.debug("QM/MM: running local SCF in presence of point charges.")
            local_hf = qmmm.mm_charge(
                local_hf,
                self.config.mm_coords,
                self.config.mm_charges,
                self.config.mm_radii,
            )

        local_hf.max_memory = self.config.max_ram_memory
        local_hf.conv_tol = self.config.convergence
        local_hf.max_cycle = self.config.max_hf_cycles

        return local_hf

    def _init_local_ks(self, xc_functional: str) -> Union[dft.uks.UKS, dft.ROKS]:
        """Function to build embedded Hartree Fock object for active subsystem.

        Note this function overwrites the total number of electrons to only include active number.

        Args:
            xc_functional (str): XC functional to use in embedded calculation.

        Returns:
            local_ks (pyscf.dft.RKS or pyscf.dft.uks.UKS): embedded Kohn-Sham DFT object.
        """
        logger.debug("Initialising localised RKS object.")
        embedded_mol: gto.Mole = self._build_mol()

        if self.localized_system.beta_active_MO_inds is None:
            # overwrite total number of electrons to only include active system
            embedded_mol.nelectron = 2 * len(self.localized_system.active_MO_inds)
            self._electron = embedded_mol.nelectron
            local_ks: dft.UKS = dft.UKS(embedded_mol)
        else:
            embedded_mol.nelectron = len(self.localized_system.active_MO_inds) + len(
                self.localized_system.beta_active_MO_inds
            )
            self._electron = embedded_mol.nelectron
            local_ks: dft.uks.UKS = scf.UKS(embedded_mol)
        logger.debug(f"{embedded_mol.nelectron=}")
        logger.debug(f"{embedded_mol.spin=}")

        local_ks.max_memory = self.config.max_ram_memory
        local_ks.conv_tol = self.config.convergence
        local_ks.xc = xc_functional

        return local_ks

    def _subsystem_dft(
        self, global_ks, localized_system
    ) -> tuple[float, float, np.typing.NDArray]:
        """Function to perform subsystem USK DFT calculation."""
        logger.debug("Calculating active and environment subsystem terms.")

        def _ks_components(
            ks_system: dft.KohnShamDFT,
            subsystem_dm: np.ndarray,
        ) -> tuple[float, np.ndarray, np.ndarray]:
            """Calculate the components of subsystem energy from a USK DFT calculation.

            For a given density matrix this function returns the electronic energy, exchange correlation energy and
            J,K, V_xc matrices.

            Args:
                ks_system (pyscf.dft.KohnShamDFT): PySCF Kohn-Sham object
                subsystem_dm (np.ndarray): density matrix (to calculate all matrices from)


            Returns:
                e_act (float): Active region energy.
                two_e_term (npt.NDArray): Two electron potential term
                j_mat (npt.NDArray): J_matrix defined by input density matrix
            """
            logger.debug("Finding subsystem USK componenets.")
            # It seems that PySCF lumps J and K in the J array
            # need to access the potential for the right subsystem for unrestricted
            logger.debug(f"{subsystem_dm.shape=}")
            two_e_term = ks_system.get_veff(dm=subsystem_dm)
            j_mat = ks_system.get_j(dm=subsystem_dm)
            # k_mat = np.zeros_like(j_mat) not needed for PySCF.

            # v_xc = two_e_term - j_mat

            if subsystem_dm.ndim == 3:
                dm_tot = subsystem_dm[0] + subsystem_dm[1]
            else:
                dm_tot = subsystem_dm
            logger.debug(f"{dm_tot.shape=}")

            # e_act = (
            #     np.einsum("ij,ji->", ks_system.get_hcore(), dm_tot)
            #     + 0.5 * (np.einsum("ij,ji->", j_tot, dm_tot))
            #     + two_e_term.exc
            # )
            e_act = (
                np.einsum("ij,ji->", ks_system.get_hcore(), dm_tot)
                + two_e_term.ecoul
                + two_e_term.exc
            )

            # if check_E_with_pyscf:
            #     energy_elec_pyscf = global_ks.energy_elec(dm=dm_matrix)[0]
            #     if not np.isclose(energy_elec_pyscf, energy_elec):
            #         raise ValueError("Energy calculation incorrect")
            logger.debug("Subsystem USK components found.")
            logger.debug(f"{e_act=}")
            logger.debug(f"{two_e_term=}")
            return e_act, two_e_term, j_mat

        if localized_system.beta_dm_active is None:
            logger.debug("Using spinless density matrix.")
            dm_act = localized_system.dm_active
            dm_env = localized_system.dm_enviro
        else:
            dm_act = np.array(
                [
                    localized_system.dm_active,
                    localized_system.beta_dm_active,
                ]
            )
            dm_env = np.array(
                [
                    localized_system.dm_enviro,
                    localized_system.beta_dm_enviro,
                ]
            )

        (e_act, two_e_act, j_act) = _ks_components(global_ks, dm_act)
        # logger.debug(e_act, alpha_e_xc_act)
        (e_env, two_e_env, j_env) = _ks_components(global_ks, dm_env)
        # logger.debug(alpha_e_env, alpha_e_xc_env, alpha_ecoul_env)

        # Computing cross subsystem terms
        logger.debug("Calculating two electron cross subsystem energy.")
        total_dm = localized_system.dm_active + localized_system.dm_enviro

        if localized_system.beta_dm_active is not None:
            total_dm += (
                localized_system.beta_dm_active + localized_system.beta_dm_enviro
            )

        two_e_term_total = global_ks.get_veff(dm=total_dm)
        logger.debug(f"{total_dm.shape=}")
        logger.debug(f"{two_e_term_total.shape=}")
        e_xc_total = two_e_term_total.exc

        if localized_system.beta_dm_active is None:
            j_cross = 0.5 * (
                np.einsum("ij,ij", localized_system.dm_active, j_env)
                + np.einsum("ij,ij", localized_system.dm_enviro, j_act)
            )
        else:
            j_cross = 0.5 * (
                np.einsum("ij,ij", localized_system.dm_active, j_env[0])
                + np.einsum("ij,ij", localized_system.dm_enviro, j_act[0])
                + np.einsum("ij,ij", localized_system.dm_active, j_env[1])
                + np.einsum("ij,ij", localized_system.dm_enviro, j_act[1])
                + np.einsum("ij,ij", localized_system.beta_dm_active, j_env[1])
                + np.einsum("ij,ij", localized_system.beta_dm_enviro, j_act[1])
                + np.einsum("ij,ij", localized_system.beta_dm_active, j_env[0])
                + np.einsum("ij,ij", localized_system.beta_dm_enviro, j_act[0])
            )
        logger.debug(f"{j_cross=}")

        # Because of projection we expect kinetic term to be zero
        k_cross = 0.0

        xc_cross = e_xc_total - two_e_act.exc - two_e_env.exc
        logger.debug(f"{e_xc_total=}")
        logger.debug(f"{two_e_act.exc=}")
        logger.debug(f"{two_e_env.exc=}")

        # overall two_electron cross energy
        two_e_cross = j_cross + k_cross + xc_cross

        logger.debug("USK components")
        logger.debug(f"e_act: {e_act}")
        logger.debug(f"e_env: {e_env}")
        logger.debug(f"two_e_cross: {two_e_cross}")
        logger.debug(f"e_nuc: {global_ks.energy_nuc()}")
        return e_act, e_env, two_e_cross

    @cached_property
    def _env_projector(self) -> np.ndarray:
        """Return a projector onto the environment in orthogonal basis."""
        logger.debug("Getting Environment Projector.")
        s_mat = self._global_ks.get_ovlp()
        logger.debug(f"{s_mat.shape=}")
        env_projector_alpha = s_mat @ self.localized_system.dm_enviro @ s_mat

        if self.localized_system.beta_dm_enviro is None:
            env_projector = env_projector_alpha

        else:
            env_projector_beta = s_mat @ self.localized_system.beta_dm_enviro @ s_mat
            env_projector = np.array([env_projector_alpha, env_projector_beta])
        logger.debug(f"{env_projector.shape=}")
        return env_projector

    def _run_emb_ccsd(
        self,
        emb_pyscf_scf_rhf: Union[scf.ROHF, scf.UHF],
        frozen: Optional[list] = None,
    ) -> tuple[cc.CCSD, float]:
        """Function run CCSD on embedded restricted Hartree Fock object.

        Note emb_pyscf_scf_rhf is ROHF object for the active embedded subsystem (defined in localized basis)
        (see get_embedded_rhf method)

        Args:
            emb_pyscf_scf_rhf (scf.ROHF): PySCF restricted Hartree Fock object of active embedded subsystem
            frozen (List): A path to an .xyz file describing molecular geometry.

        Returns:
            ccsd (cc.CCSD): PySCF CCSD object
            e_ccsd_corr (float): electron correlation CCSD energy
        """
        return run_emb_ccsd(
            emb_pyscf_scf_rhf,
            frozen,
            self.config.convergence,
            self.config.max_ram_memory,
        )

    def _run_emb_fci(
        self,
        emb_pyscf_scf_rhf: Union[scf.ROHF, scf.UHF],
        frozen: Optional[list] = None,
    ) -> fci.FCI:
        """Function run FCI on embedded restricted Hartree Fock object.

        Note emb_pyscf_scf_rhf is ROHF object for the active embedded subsystem (defined in localized basis)
        (see get_embedded_rhf method)

        Args:
            emb_pyscf_scf_rhf (scf.ROHF): PySCF restricted Hartree Fock object of active embedded subsystem
            frozen (List): A path to an .xyz file describing moleclar geometry.

        Returns:
            fci_scf (fci.FCI): PySCF FCI object
        """
        return run_emb_fci(
            emb_pyscf_scf_rhf,
            frozen,
            self.config.convergence,
            self.config.max_ram_memory,
        )

    def _mu_embed(
        self, localized_scf: StreamObject, embedding_potential: np.ndarray
    ) -> tuple[StreamObject, np.ndarray]:
        """Embed using the Mu-shift projector.

        Args:
            localized_scf (StreamObject): A PySCF scf method with the correct number of electrons for the active region.
            embedding_potential (np.ndarray): Potential calculated from two electron terms in dft.

        Returns:
            np.ndarray: Matrix form of the embedding potential.
            StreamObject: The embedded scf object.
        """
        logger.debug("Running mu embedded scf calculation.")

        # Modify the energy_elec function to handle different h_cores
        # which we need for different embedding potentials

        v_emb = (self.config.mu_level_shift * self._env_projector) + embedding_potential
        logger.debug(f"{v_emb.shape=}")

        if v_emb.ndim == 3:
            localized_scf.energy_elec = lambda *args: energy_elec(localized_scf, *args)

        logger.debug(f"{v_emb.shape=}")
        logger.debug(f"{self._env_projector.shape=}")
        logger.debug(f"{embedding_potential.shape=}")
        hcore_std = localized_scf.get_hcore
        logger.debug(f"{hcore_std().shape=}")
        localized_scf.get_hcore = lambda *args: hcore_std(*args) + v_emb
        # veff_std = localized_scf.get_veff
        # localized_scf.get_veff = lambda *args: veff_std(*args) + v_emb
        logger.debug(f"embedded hcore shape {localized_scf.get_hcore().shape}")

        localized_scf.kernel()
        logger.info(
            f"Embedded scf energy MU_SHIFT: {localized_scf.e_tot}, converged: {localized_scf.converged}"
        )

        return localized_scf, v_emb

    def _huzinaga_embed(
        self,
        active_scf: StreamObject,
        embedding_potential: np.ndarray,
        dmat_initial_guess: Optional[tuple[np.ndarray]] = None,
    ) -> tuple[StreamObject, np.ndarray]:
        """Embed using Huzinaga projector.

        Args:
            active_scf (StreamObject): A PySCF scf method with the correct number of electrons for the active region.
            embedding_potential (np.ndarray): Potential calculated from two electron terms in dft.
            dmat_initial_guess (bool): If True, use the initial guess for the density matrix.

        Returns:
            np.ndarray: Matrix form of the embedding potential.
            StreamObject: The embedded scf object.
        """
        logger.debug("Starting Huzinaga embedding method.")
        # We need to run our own SCF method here to update the potential.
        if self.localized_system.beta_dm_enviro is None:
            total_enviro_dm = self.localized_system.dm_enviro
        else:
            total_enviro_dm = np.array(
                [
                    self.localized_system.dm_enviro,
                    self.localized_system.beta_dm_enviro,
                ]
            )
        localized_scf = active_scf
        (
            c_active_embedded,
            mo_embedded_energy,
            dm_active_embedded,
            huzinaga_op_std,
            huz_scf_conv_flag,
        ) = huzinaga_scf(
            localized_scf,
            embedding_potential,
            total_enviro_dm,
            dm_conv_tol=1e-6,
            dm_initial_guess=dmat_initial_guess,
        )

        logger.debug(f"{c_active_embedded=}")

        # write results to pyscf object
        logger.debug("Writing results to PySCF object.")
        hcore_std = localized_scf.get_hcore()
        v_emb = huzinaga_op_std + embedding_potential
        localized_scf.get_hcore = lambda *args: hcore_std + v_emb

        if self.localized_system.spinless is False:
            localized_scf.energy_elec = lambda *args: energy_elec(localized_scf, *args)

        localized_scf.mo_coeff = c_active_embedded
        localized_scf.mo_occ = localized_scf.get_occ(
            mo_embedded_energy, c_active_embedded
        )
        logger.debug(f"{localized_scf.mo_occ=}")
        localized_scf.mo_energy = mo_embedded_energy
        localized_scf.e_tot = localized_scf.energy_tot(dm=dm_active_embedded)
        # localized_scf.conv_check = huz_scf_conv_flag
        localized_scf.converged = huz_scf_conv_flag

        logger.info(f"Embedded scf energy HUZINAGA: {localized_scf.e_tot}")
        return localized_scf, v_emb

    def _delete_spin_environment(
        self,
        projector: ProjectorEnum,
        n_env_mo: int,
        mo_coeff: np.ndarray,
        mo_energy: np.ndarray,
        mo_occ: np.ndarray,
        environment_projector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove enironment orbit from embedded ROHF object.

        This function removes (in fact deletes completely) the molecular orbitals
        defined by the environment of the localized system

        Args:
            projector (ProjectorEnum): The projector used to embed the system.
            n_env_mo (int): The number of molecular orbitals in the environment.
            mo_coeff (np.ndarray): The molecular orbitals.
            mo_energy (np.ndarray): The molecular orbital energies.
            mo_occ (np.ndarray): The molecular orbital occupation numbers.
            environment_projector (np.ndarray): Matrix to project mo_coeff onto environment.

        Returns:
            embedded_rhf (StreamObject): Returns input, but with environment orbitals deleted
        """
        logger.debug("Deleting environment for spin.")
        logger.debug(f"{projector=}")
        logger.debug(f"{n_env_mo=}")
        logger.debug(f"{mo_coeff.shape=}")
        logger.debug(f"{mo_energy=}")
        logger.debug(f"{mo_occ=}")
        logger.debug(f"{environment_projector.shape=}")

        match projector:
            case ProjectorEnum.HUZ:
                # MOs which have the greatest overlap with the
                overlap = np.einsum(
                    "ij, ki -> i",
                    mo_coeff.swapaxes(-1, -2),
                    environment_projector @ mo_coeff,
                )
                overlap_by_size = overlap.argsort()[::-1]
                logger.debug(f"{overlap_by_size=}")
                frozen_enviro_orb_inds = overlap_by_size[:n_env_mo]

            case ProjectorEnum.MU:
                # Orbitals which have been shifted to have energy mu are removed
                shift = mo_coeff.shape[-1] - n_env_mo
                frozen_enviro_orb_inds = [
                    mo_i for mo_i in range(shift, mo_coeff.shape[-1])
                ]

        active_MOs_occ_and_virt_embedded = [
            mo_i
            for mo_i in range(mo_coeff.shape[-1])
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
        active_mo_occ = mo_occ[: len(active_MOs_occ_and_virt_embedded)]

        logger.debug("Spin environment deleted.")
        logger.debug(f"{active_mo_coeff=}")
        logger.debug(f"{active_mo_energy=}")
        logger.debug(f"{active_mo_occ=}")
        return active_mo_coeff, active_mo_energy, active_mo_occ

    def _delete_environment(
        self, projector: ProjectorEnum, scf: StreamObject
    ) -> StreamObject:
        """Remove enironment orbit from embedded ROHF object.

        This function removes (in fact deletes completely) the molecular orbitals
        defined by the environment of the localized system

        Args:
            projector (ProjectorEnum): The projector used to embed the system.
            scf (StreamObject): The embedded SCF object.

        Returns:
            StreamObject: Returns input, but with environment orbitals deleted.
        """
        logger.debug("Deleting environment from SCF object.")

        if self.localized_system.beta_enviro_MO_inds is None:
            n_env_mos = len(self.localized_system.enviro_MO_inds)
            scf.mo_coeff, scf.mo_energy, scf.mo_occ = self._delete_spin_environment(
                projector,
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
            (
                mo_coeff[0],
                mo_energy[0],
                mo_occ[0],
            ) = self._delete_spin_environment(
                projector,
                alpha_n_env_mos,
                scf.mo_coeff[0],
                scf.mo_energy[0],
                scf.mo_occ[0],
                self._env_projector[0],
            )
            (mo_coeff[1], mo_energy[1], mo_occ[1]) = self._delete_spin_environment(
                projector,
                beta_n_env_mos,
                scf.mo_coeff[1],
                scf.mo_energy[1],
                scf.mo_occ[1],
                self._env_projector[1],
            )

            # Need to do it this way or there are broadcasting issues
            scf.mo_coeff = np.array(
                [mo_coeff[0], mo_coeff[1]]
            )  # np.array([mo_coeff[0], mo_coeff[1]])
            scf.mo_energy = np.array(
                [mo_energy[0], mo_energy[1]]
            )  # np.array([mo_energy[0], mo_energy[1]])
            scf.mo_occ = np.array(
                [mo_occ[0], mo_occ[1]]
            )  # np.array([mo_occ[0], mo_occ[1]])

        logger.debug("Environment deleted.")
        return scf

    def _dft_in_dft(self, projection_method: ProjectorEnum) -> dict:
        """Return energy of DFT in DFT embedding.

        Note run_mu_shift (bool) and run_huzinaga (bool) flags define which method to use (can be both)
        This is done when object is initialized.

        Args:
            driver (NbedDriver): A driver object.
            projection_method (callable): Embedding method to use (mu or huzinaga).

        Returns:
            dict: DFT-in-DFT embedding results.
        """
        return dft_in_dft(self, projection_method)

    def embed(
        self,
        init_huzinaga_rhf_with_mu: bool = False,
        n_mo_overwrite: tuple[int | None, int | None] = (None, None),
    ) -> None:
        """Run embedded scf calculation.

        Args:
            init_huzinaga_rhf_with_mu (bool): Will run mu-shift projector even when input projector='huzinaga'.
            n_mo_overwrite (tuple[int, int]): Enforces a specific number of MOs are included in the active region. Used for ACE-of-SPADE reaction path localization.
        """
        logger.debug("Embedding molecule.")
        self.e_nuc = self._global_ks.energy_nuc()

        if n_mo_overwrite is not None and n_mo_overwrite != (None, None):
            logger.debug(
                "Setting n_mo_overwrite with value from embed args %s", n_mo_overwrite
            )
            self.n_mo_overwrite = n_mo_overwrite
        else:
            logger.debug("Setting n_mo_overwrite with value from config.")
            self.n_mo_overwrite = self.config.n_mo_overwrite

        self.localized_system = self._localize()
        logger.info("Indices of embedded electrons:")
        logger.info(self.localized_system.active_MO_inds)
        logger.info(self.localized_system.enviro_MO_inds)

        # Run subsystem DFT (calls localized rks)
        self.e_act, self.e_env, self.two_e_cross = self._subsystem_dft(
            self._global_ks, self.localized_system
        )
        logger.debug("Getting global DFT potential to optimize embedded calc in.")

        total_dm = self.localized_system.dm_active + self.localized_system.dm_enviro
        if self.localized_system.beta_dm_active is not None:
            logger.debug("Adding beta spin density matrix")
            total_dm = np.array(
                [
                    total_dm,
                    self.localized_system.beta_dm_active
                    + self.localized_system.beta_dm_enviro,
                ]
            )

        logger.debug(f"{self._global_ks.get_veff().shape=}")
        logger.debug(f"{self._global_ks.get_veff(dm=total_dm).shape=}")
        g_act_and_env = self._global_ks.get_veff(dm=total_dm)
        logger.debug(f"{total_dm.shape=}")
        logger.debug(f"{g_act_and_env.shape=}")

        if self.localized_system.beta_dm_active is None:
            g_act = self._global_ks.get_veff(dm=self.localized_system.dm_active)
        else:
            g_act = self._global_ks.get_veff(
                dm=np.array(
                    [
                        self.localized_system.dm_active,
                        self.localized_system.beta_dm_active,
                    ]
                )
            )
        embedding_potential = g_act_and_env - g_act
        self.embedding_potential = embedding_potential

        logger.info(f"DFT potential average {np.mean(embedding_potential)}.")

        logger.debug("Beginning Projection.")
        if (
            self.config.projector in [ProjectorEnum.MU, ProjectorEnum.BOTH]
            or init_huzinaga_rhf_with_mu
        ):
            local_hf = self._init_local_hf()
            embedded_scf, v_emb = self._mu_embed(local_hf, embedding_potential)
            self._mu = self.collect_results(embedded_scf, v_emb, ProjectorEnum.MU)

        if self.config.projector in [ProjectorEnum.HUZ, ProjectorEnum.BOTH]:
            local_hf = self._init_local_hf()
            dmat_initial_guess: Optional[tuple[NDArray]] = (
                self._mu["scf"].make_rdm1() if init_huzinaga_rhf_with_mu else None
            )
            embedded_scf, v_emb = self._huzinaga_embed(
                local_hf, embedding_potential, dmat_initial_guess
            )
            self._huzinaga = self.collect_results(
                embedded_scf, v_emb, ProjectorEnum.HUZ
            )

        match self.config.projector:
            case ProjectorEnum.MU:
                self.embedded_scf = self._mu["scf"]
                self.classical_energy = self._mu["classical_energy"]
            case ProjectorEnum.HUZ:
                self.embedded_scf = self._huzinaga["scf"]
                self.classical_energy = self._huzinaga["classical_energy"]
            case ProjectorEnum.BOTH:
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
            case _:
                logger.debug("Projector did not match any know case.")
                logger.warning("Not assigning embedded_scf or classial_energy")

        if filename := self.config.savefile is not None:
            logger.debug("Saving results to file %s", filename)
            with open(filename, "w") as f:
                jdump({"mu": self._mu, "huzinaga": self._huzinaga}, f)

        logger.info("Embedding complete.")

    def collect_results(
        self, embedded_scf: StreamObject, v_emb: NDArray, projector: ProjectorEnum
    ) -> dict:
        """Build a results dictionary.

        Args:
            embedded_scf (StreamObject): An embedded pyscf scf object.
            v_emb (NDArray): Embedding Potential
            projector (ProjectorEnum): Which projector the result should use.

        Returns:
            dict: A dict of results.
        """
        result = {}
        result["scf"] = embedded_scf.copy()
        result["v_emb"] = v_emb
        result["mo_energies_emb_pre_del"] = result["scf"].mo_energy
        result["scf"] = self._delete_environment(projector, result["scf"])
        result["mo_energies_emb_post_del"] = result["scf"].mo_energy

        logger.info(f"V emb mean {projector}: {np.mean(result['v_emb'])}")

        # calculate correction
        if self.localized_system.beta_dm_active is None:
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

        # Virtual localization
        # TODO correlation energy correction???
        if self.config.run_virtual_localization is True:
            logger.debug("Performing virtual localization.")
            result["scf"] = ConcentricLocalizer(
                result["scf"],
                self.config.n_active_atoms,
                max_shells=self.config.max_shells,
            ).localize_virtual()

        result["e_rhf"] = (
            result["scf"].e_tot
            + self.e_env
            + self.two_e_cross
            - result["correction"]
            - result["beta_correction"]
        )
        logger.info(f"ROHF energy: {result['e_rhf']}")

        # classical energy
        result["classical_energy"] = (
            self.e_env
            + self.two_e_cross
            + self.e_nuc
            - result["correction"]
            - result["beta_correction"]
        )
        logger.debug(f"Classical energy: {result['classical_energy']}")

        # Calculate ccsd or fci energy
        if self.config.run_ccsd_emb is True:
            logger.debug("Performing CCSD-in-DFT embedding.")
            ccsd_emb, e_ccsd_corr = self._run_emb_ccsd(result["scf"])
            result["e_ccsd"] = (
                ccsd_emb.e_tot
                + self.e_env
                + self.two_e_cross
                - result["correction"]
                - result["beta_correction"]
            )
            result["ccsd_emb"] = ccsd_emb.e_tot - self.e_nuc

            logger.info(f"CCSD Energy {projector}:\t{result['e_ccsd']}")

        if self.config.run_fci_emb is True:
            logger.debug("Performing FCI-in-DFT embedding.")
            fci_emb = self._run_emb_fci(result["scf"])
            result["e_fci"] = (
                (fci_emb.e_tot)
                + self.e_env
                + self.two_e_cross
                - result["correction"]
                - result["beta_correction"]
            )
            logger.info(f"FCI Energy {projector}:\t{result['e_fci']}")

            result["fci_emb"] = fci_emb.e_tot - self.e_nuc
        result["hf_emb"] = result["scf"].e_tot - self.e_nuc

        if self.config.run_dft_in_dft is True:
            did = self._dft_in_dft(projector)
            result.update(did)

        # Build second quantised Hamiltonian
        hb = HamiltonianBuilder(result["scf"], result["classical_energy"])
        result["second_quantised"] = hb.build()

        logger.debug(f"Found result for {projector}")
        logger.debug(result)

        return result


def run_emb_fci(
    emb_pyscf_scf_rhf: Union[scf.ROHF, scf.UHF],
    frozen: Optional[list] = None,
    convergence: Optional[float] = 1e-6,
    max_ram_memory: Optional[int] = 4000,
) -> fci.FCI:
    """Function run FCI on embedded restricted Hartree Fock object.

    Note emb_pyscf_scf_rhf is ROHF object for the active embedded subsystem (defined in localized basis)
    (see get_embedded_rhf method)

    Args:
        emb_pyscf_scf_rhf (scf.ROHF): PySCF restricted Hartree Fock object of active embedded subsystem
        frozen (List): A path to an .xyz file describing moleclar geometry.
        convergence (float): convergence tolerance.
        max_ram_memory (int): Maximum memory allocation for FCI.

    Returns:
        fci_scf (fci.FCI): PySCF FCI object
    """
    logger.debug("Starting embedded FCI calculation.")
    logger.debug(f"{type(emb_pyscf_scf_rhf)=}")
    logger.debug(f"{frozen=}")
    logger.debug(f"{convergence=}")
    logger.debug(f"{max_ram_memory=}")

    if frozen is None:
        fci_scf = fci.FCI(emb_pyscf_scf_rhf)
    else:
        from pyscf import mcscf

        fci_scf = mcscf.CASSCF(
            emb_pyscf_scf_rhf,
            emb_pyscf_scf_rhf.mol.nelec,
            emb_pyscf_scf_rhf.mol.nao - len(frozen),
        )
        fci_scf.sort_mo(
            [i + 1 for i in range(emb_pyscf_scf_rhf.mol.nao) if i not in frozen]
        )
    fci_scf.conv_tol = convergence
    fci_scf.max_memory = max_ram_memory

    # For UHF, PySCF assumes that hcore is spinless and 2D
    # Because we update hcore for embedding, we need to calculate our own h1e term.
    from functools import reduce

    if np.ndim(hcore := emb_pyscf_scf_rhf.get_hcore()) == 3:
        mo = emb_pyscf_scf_rhf.mo_coeff
        h1e = [
            reduce(np.dot, (mo[0].T, hcore[0], mo[0])),
            reduce(np.dot, (mo[1].T, hcore[1], mo[1])),
        ]
        fci_scf.kernel(h1e=h1e)
    else:
        # kernel function default value is passed in
        fci_scf.kernel()
    logger.info(f"FCI embedding energy: {fci_scf.e_tot}")
    return fci_scf


def run_emb_ccsd(
    emb_pyscf_scf_rhf: Union[scf.ROHF, scf.UHF],
    frozen: Optional[list] = None,
    convergence: float = 1e-6,
    max_ram_memory: int = 4000,
) -> tuple[cc.CCSD, float]:
    """Function run CCSD on embedded restricted Hartree Fock object.

    Note emb_pyscf_scf_rhf is ROHF object for the active embedded subsystem (defined in localized basis)
    (see get_embedded_rhf method)

    Args:
        emb_pyscf_scf_rhf (scf.ROHF): PySCF restricted Hartree Fock object of active embedded subsystem
        frozen (List): A path to an .xyz file describing molecular geometry.
        convergence (float): Convergence threshold.
        max_ram_memory (int): Maximum ram to use in solving.

    Returns:
        ccsd (cc.CCSD): PySCF CCSD object
        e_ccsd_corr (float): electron correlation CCSD energy
    """
    logger.debug("Starting embedded CCSD calculation.")
    ccsd = cc.CCSD(emb_pyscf_scf_rhf, frozen=frozen)
    ccsd.conv_tol = convergence
    ccsd.max_memory = max_ram_memory

    e_ccsd_corr, _, _ = ccsd.kernel()
    logger.info(f"Embedded CCSD energy: {e_ccsd_corr}")
    return ccsd, e_ccsd_corr


def dft_in_dft(driver: "NbedDriver", projection_method: ProjectorEnum) -> dict:
    """Return energy of DFT in DFT embedding.

    Note run_mu_shift (bool) and run_huzinaga (bool) flags define which method to use (can be both)
    This is done when object is initialized.

    Args:
        driver (NbedDriver): A driver object.
        projection_method (callable): Embedding method to use (mu or huzinaga).

    Returns:
        dict: DFT-in-DFT embedding results.
    """
    result = {}
    e_nuc = driver._global_ks.energy_nuc()

    local_rks_same_functional = driver._init_local_ks(driver._global_ks.xc)
    hcore_std = local_rks_same_functional.get_hcore()
    match projection_method:
        case ProjectorEnum.MU:
            result["scf_dft"], result["v_emb_dft"] = driver._mu_embed(
                local_rks_same_functional, driver.embedding_potential
            )
        case ProjectorEnum.HUZ:
            result["scf_dft"], result["v_emb_dft"] = driver._huzinaga_embed(
                local_rks_same_functional, driver.embedding_potential
            )
    result["scf_dft"] = driver._delete_environment(projection_method, result["scf_dft"])

    if driver.localized_system.beta_dm_active is not None:
        y_emb_alpha, y_emb_beta = result["scf_dft"].make_rdm1()

        # calculate correction
        result["dft_correction"] = np.einsum(
            "ij,ij",
            result["v_emb_dft"][0],
            (y_emb_alpha - driver.localized_system.dm_active),
        )

        result["dft_correction_beta"] = np.einsum(
            "ij,ij",
            result["v_emb_dft"][1],
            (y_emb_beta - driver.localized_system.beta_dm_active),
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
            (y_emb - driver.localized_system.dm_active),
        )
        veff = result["scf_dft"].get_veff(dm=y_emb)
        result["dft_correction_beta"] = 0
        rks_e_elec = veff.exc + veff.ecoul + np.einsum("ij,ij", hcore_std, y_emb)

    result["e_dft_in_dft"] = (
        rks_e_elec
        + driver.e_env
        + driver.two_e_cross
        + result["dft_correction"]
        + result["dft_correction_beta"]
        + e_nuc
    )
    result["emb_dft"] = rks_e_elec

    return result
