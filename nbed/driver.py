"""Module containg the NbedDriver Class."""

import logging
from functools import cached_property, reduce
from typing import Callable, Optional, Union

import numpy as np
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

from .config import Localizer, NbedConfig, Projector
from .scf import energy_elec, huzinaga_scf

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
        self.dft_potential: np.typing.NDArray
        self.electron: int
        self.v_emb: np.typing.NDArray
        self._mu: dict = None
        self._huzinaga: dict = None

        if config.force_unrestricted:
            logger.debug("Forcing unrestricted SCF")
            self._restricted_scf = False
        elif self.config.charge % 2 == 1 or self.config.spin != 0:
            logger.debug("Open shells, using unrestricted SCF.")
            self._restricted_scf = False
        else:
            logger.debug("Closed shells, using restricted SCF.")
            self._restricted_scf = True

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
        global_hf = (
            scf.UHF(mol_full, **hf_kwargs)
            if not self._restricted_scf
            else scf.RHF(mol_full)
        )
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
        """Method to run full cheap molecule RKS DFT calculation.

        Note this is necessary to perform localization procedure.
        """
        logger.debug("Running full system KS DFT.")
        mol_full = self._build_mol()
        global_ks = (
            dft.RKS(mol_full, **ks_kwargs)
            if self._restricted_scf
            else dft.UKS(mol_full, **ks_kwargs)
        )
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
        logger.info(f"Global RKS: {global_ks.e_tot}")

        if global_ks.converged is not True:
            logger.warning("(cheap) global DFT calculation has NOT converged!")

        return global_ks

    def _localize(self) -> OccupiedLocalizer:
        """Run the localizer class."""
        logger.debug(f"Getting localized system using {self.config.localization}.")

        match self.config.localization:
            case Localizer.SPADE:
                return SPADELocalizer(
                    self._global_ks,
                    self.config.n_active_atoms,
                    max_shells=self.config.max_shells,
                    n_mo_overwrite=self.config.n_mo_overwrite,
                )
            case Localizer.BOYS:
                return BOYSLocalizer(
                    self._global_ks,
                    self.config.n_active_atoms,
                    occ_cutoff=self.config.occupied_threshold,
                    virt_cutoff=self.config.virtual_threshold,
                )
            case Localizer.IBO:
                return IBOLocalizer(
                    self._global_ks,
                    self.config.n_active_atoms,
                    occ_cutoff=self.config.occupied_threshold,
                    virt_cutoff=self.config.virtual_threshold,
                )
            case Localizer.PM:
                return PMLocalizer(
                    self._global_ks,
                    self.config.n_active_atoms,
                    occ_cutoff=self.config.occupied_threshold,
                    virt_cutoff=self.config.virtual_threshold,
                )

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
            embedded_mol.spin = 0
            self._electron = embedded_mol.nelectron
            local_hf: scf.rhf.RHF = scf.RHF(embedded_mol)
        else:
            embedded_mol.nelectron = len(self.localized_system.active_MO_inds) + len(
                self.localized_system.beta_active_MO_inds
            )
            embedded_mol.spin = len(self.localized_system.active_MO_inds) - len(
                self.localized_system.beta_active_MO_inds
            )
            self._electron = embedded_mol.nelectron
            local_hf: scf.uhf.UHF = scf.UHF(embedded_mol)

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

    def _init_local_ks(self, xc_functional: str) -> Union[dft.uks.UKS, dft.rks.RKS]:
        """Function to build embedded Hartree Fock object for active subsystem.

        Note this function overwrites the total number of electrons to only include active number.

        Args:
            xc_functional (str): XC functional to use in embedded calculation.

        Returns:
            local_ks (pyscf.dft.rks.RKS or pyscf.dft.uks.UKS): embedded Kohn-Sham DFT object.
        """
        logger.debug("Initialising localised RKS object.")
        embedded_mol: gto.Mole = self._build_mol()

        if self._restricted_scf:
            # overwrite total number of electrons to only include active system
            embedded_mol.nelectron = 2 * len(self.localized_system.active_MO_inds)
            self._electron = embedded_mol.nelectron
            local_ks: dft.rks.RKS = scf.RKS(embedded_mol)
        else:
            embedded_mol.nelectron = len(self.localized_system.active_MO_inds) + len(
                self.localized_system.beta_active_MO_inds
            )
            self._electron = embedded_mol.nelectron
            local_ks: dft.uks.UKS = scf.UKS(embedded_mol)

        local_ks.max_memory = self.config.max_ram_memory
        local_ks.conv_tol = self.config.convergence
        local_ks.xc = xc_functional

        return local_ks

    def _subsystem_dft(self) -> None:
        """Function to perform subsystem RKS DFT calculation."""
        logger.debug("Calculating active and environment subsystem terms.")

        def _ks_components(
            ks_system: dft.KohnShamDFT,
            subsystem_dm: np.ndarray,
        ) -> tuple[float, np.ndarray, np.ndarray]:
            """Calculate the components of subsystem energy from a RKS DFT calculation.

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
            logger.debug("Finding subsystem RKS componenets.")
            # It seems that PySCF lumps J and K in the J array
            # need to access the potential for the right subsystem for unrestricted
            logger.debug(subsystem_dm.shape)
            two_e_term = ks_system.get_veff(dm=subsystem_dm)
            j_mat = ks_system.get_j(dm=subsystem_dm)
            # k_mat = np.zeros_like(j_mat) not needed for PySCF.

            # v_xc = two_e_term - j_mat

            if not self._restricted_scf:
                dm_tot = subsystem_dm[0] + subsystem_dm[1]
            else:
                dm_tot = subsystem_dm

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
            #     energy_elec_pyscf = self._global_ks.energy_elec(dm=dm_matrix)[0]
            #     if not np.isclose(energy_elec_pyscf, energy_elec):
            #         raise ValueError("Energy calculation incorrect")
            logger.debug("Subsystem RKS components found.")
            logger.debug(f"{e_act=}")
            logger.debug(f"{two_e_term=}")
            return e_act, two_e_term, j_mat

        if not self._restricted_scf:
            dm_act = np.array(
                [
                    self.localized_system.dm_active,
                    self.localized_system.beta_dm_active,
                ]
            )
            dm_env = np.array(
                [
                    self.localized_system.dm_enviro,
                    self.localized_system.beta_dm_enviro,
                ]
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
        logger.debug(f"{j_cross=}")

        # Because of projection we expect kinetic term to be zero
        k_cross = 0.0

        xc_cross = e_xc_total - two_e_act.exc - two_e_env.exc
        logger.debug(f"{e_xc_total=}")
        logger.debug(f"{two_e_act.exc=}")
        logger.debug(f"{two_e_env.exc=}")

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
        frozen: Optional[list] = None,
    ) -> tuple[cc.CCSD, float]:
        """Function run CCSD on embedded restricted Hartree Fock object.

        Note emb_pyscf_scf_rhf is RHF object for the active embedded subsystem (defined in localized basis)
        (see get_embedded_rhf method)

        Args:
            emb_pyscf_scf_rhf (scf.RHF): PySCF restricted Hartree Fock object of active embedded subsystem
            frozen (List): A path to an .xyz file describing molecular geometry.

        Returns:
            ccsd (cc.CCSD): PySCF CCSD object
            e_ccsd_corr (float): electron correlation CCSD energy
        """
        logger.debug("Starting embedded CCSD calculation.")
        ccsd = cc.CCSD(emb_pyscf_scf_rhf, frozen=frozen)
        ccsd.conv_tol = self.config.convergence
        ccsd.max_memory = self.config.max_ram_memory

        e_ccsd_corr, _, _ = ccsd.kernel()
        logger.info(f"Embedded CCSD energy: {e_ccsd_corr}")
        return ccsd, e_ccsd_corr

    def _run_emb_FCI(
        self,
        emb_pyscf_scf_rhf: Union[scf.RHF, scf.UHF],
        frozen: Optional[list] = None,
    ) -> fci.FCI:
        """Function run FCI on embedded restricted Hartree Fock object.

        Note emb_pyscf_scf_rhf is RHF object for the active embedded subsystem (defined in localized basis)
        (see get_embedded_rhf method)

        Args:
            emb_pyscf_scf_rhf (scf.RHF): PySCF restricted Hartree Fock object of active embedded subsystem
            frozen (List): A path to an .xyz file describing moleclar geometry.

        Returns:
            fci_scf (fci.FCI): PySCF FCI object
        """
        logger.debug("Starting embedded FCI calculation.")
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
        fci_scf.conv_tol = self.config.convergence
        fci_scf.max_memory = self.config.max_ram_memory

        # For UHF, PySCF assumes that hcore is spinless and 2D
        # Because we update hcore for embedding, we need to calculate our own h1e term.
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

    def _mu_embed(
        self, localized_scf: StreamObject, dft_potential: np.ndarray
    ) -> tuple[StreamObject, np.ndarray]:
        """Embed using the Mu-shift projector.

        Args:
            localized_scf (StreamObject): A PySCF scf method with the correct number of electrons for the active region.
            dft_potential (np.ndarray): Potential calculated from two electron terms in dft.

        Returns:
            np.ndarray: Matrix form of the embedding potential.
            StreamObject: The embedded scf object.
        """
        logger.debug("Running embedded scf calculation.")

        # Modify the energy_elec function to handle different h_cores
        # which we need for different embedding potentials
        v_emb = (self.config.mu_level_shift * self._env_projector) + dft_potential
        hcore_std = localized_scf.get_hcore()
        localized_scf.get_hcore = lambda *args: hcore_std + v_emb

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
        dmat_initial_guess: Optional[np.ndarray] = None,
    ) -> tuple[StreamObject, np.ndarray]:
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
                [
                    self.localized_system.dm_enviro,
                    self.localized_system.beta_dm_enviro,
                ]
            )

        (
            c_active_embedded,
            mo_embedded_energy,
            dm_active_embedded,
            huzinaga_op_std,
            huz_scf_conv_flag,
        ) = huzinaga_scf(
            active_scf,
            dft_potential,
            total_enviro_dm,
            dm_conv_tol=1e-6,
            dm_initial_guess=dmat_initial_guess,
        )

        logger.debug(f"{c_active_embedded=}")

        # write results to pyscf object
        logger.debug("Writing results to PySCF object.")
        hcore_std = active_scf.get_hcore()
        v_emb = huzinaga_op_std + dft_potential
        active_scf.get_hcore = lambda *args: hcore_std + v_emb

        if not self._restricted_scf:
            active_scf.energy_elec = lambda *args: energy_elec(active_scf, *args)

        active_scf.mo_coeff = c_active_embedded
        active_scf.mo_occ = active_scf.get_occ(mo_embedded_energy, c_active_embedded)
        active_scf.mo_energy = mo_embedded_energy
        active_scf.e_tot = active_scf.energy_tot(dm=dm_active_embedded)
        # localized_scf.conv_check = huz_scf_conv_flag
        active_scf.converged = huz_scf_conv_flag

        logger.info(f"Huzinaga scf energy: {active_scf.e_tot}")
        return active_scf, v_emb

    def _delete_spin_environment(
        self,
        method: str,
        n_env_mo: int,
        mo_coeff: np.ndarray,
        mo_energy: np.ndarray,
        mo_occ: np.ndarray,
        environment_projector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove enironment orbit from embedded rhf object.

        This function removes (in fact deletes completely) the molecular orbitals
        defined by the environment of the localized system

        Args:
            method (str): The localization method used to embed the system. 'huzinaga' or 'mu'.
            n_env_mo (int): The number of molecular orbitals in the environment.
            mo_coeff (np.ndarray): The molecular orbitals.
            mo_energy (np.ndarray): The molecular orbital energies.
            mo_occ (np.ndarray): The molecular orbital occupation numbers.
            environment_projector (np.ndarray): Matrix to project mo_coeff onto environment.

        Returns:
            embedded_rhf (StreamObject): Returns input, but with environment orbitals deleted
        """
        logger.debug("Deleting environment for spin.")

        if method == "huzinaga":
            overlap = np.einsum(
                "ij, ki -> i",
                mo_coeff.T,
                environment_projector @ mo_coeff,
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
            (
                mo_coeff[0],
                mo_energy[0],
                mo_occ[0],
            ) = self._delete_spin_environment(
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

        e_nuc = self._global_ks.energy_nuc()

        self.localized_system = self._localize()
        logger.info("Indices of embedded electrons:")
        logger.info(self.localized_system.active_MO_inds)
        logger.info(self.localized_system.enviro_MO_inds)

        # Run subsystem DFT (calls localized rks)
        self._subsystem_dft()
        logger.debug("Getting global DFT potential to optimize embedded calc in.")

        total_dm = self.localized_system.dm_active + self.localized_system.dm_enviro
        if not self._restricted_scf:
            logger.debug("Adding beta spin density matrix")
            total_dm += (
                self.localized_system.beta_dm_active
                + self.localized_system.beta_dm_enviro
            )

        g_act_and_env = self._global_ks.get_veff(dm=total_dm)
        logger.debug(f"{total_dm.shape=}")
        logger.debug(f"{g_act_and_env.shape=}")

        if self._restricted_scf:
            g_act = self._global_ks.get_veff(dm=self.localized_system.dm_active)
        else:
            g_act = self._global_ks.get_veff(
                dm=[
                    self.localized_system.dm_active,
                    self.localized_system.beta_dm_active,
                ]
            )
        dft_potential = g_act_and_env - g_act

        logger.info(f"DFT potential average {np.mean(dft_potential)}.")

        # The order of these is important for
        # initializing huzinaga with mu

        embedding_methods_to_run = []
        if (
            self.config.projector in [Projector.MU, Projector.BOTH]
            or init_huzinaga_rhf_with_mu
        ):
            logger.debug("Queued $mu$-shift projector method.")
            embedding_methods_to_run.append("mu")

        if self.config.projector in [Projector.HUZ, Projector.BOTH]:
            logger.debug("Queued Huzinaga projector method.")
            embedding_methods_to_run.append("huzinaga")

        logger.debug(f"Embedding methods to run: {embedding_methods_to_run}")

        for projector_name in embedding_methods_to_run:
            result = {}
            logger.debug(f"Runnning embedding with {projector_name} projector.")

            if projector_name == "mu":
                embedding_method = self._mu_embed
            elif projector_name == "huzinaga":
                embedding_method = self._huzinaga_embed

            local_hf = self._init_local_hf()

            if projector_name == "huzinaga" and init_huzinaga_rhf_with_mu:
                dmat_initial_guess = (self._mu["scf"].make_rdm1(),)
                result["scf"], result["v_emb"] = embedding_method(
                    local_hf, dft_potential, dmat_initial_guess
                )
            else:
                result["scf"], result["v_emb"] = embedding_method(
                    local_hf, dft_potential
                )

            result["mo_energies_emb_pre_del"] = result["scf"].mo_energy
            result["scf"] = self._delete_environment(projector_name, result["scf"])
            result["mo_energies_emb_post_del"] = result["scf"].mo_energy

            logger.info(f"V emb mean {projector_name}: {np.mean(result['v_emb'])}")

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

            # Virtual localization
            # TODO correlation energy correction???
            if self.config.run_virtual_localization is True:
                logger.debug("Performing virtual localization.")
                result["scf"] = ConcentricLocalizer(
                    result["scf"],
                    self.config.n_active_atoms,
                    max_shells=self.config.max_shells,
                ).localize_virtual(self._restricted_scf)

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

            # Calculate ccsd or fci energy
            if self.config.run_ccsd_emb is True:
                logger.debug("Performing CCSD-in-DFT embedding.")
                ccsd_emb, e_ccsd_corr = self._run_emb_CCSD(result["scf"], frozen=None)
                result["e_ccsd"] = (
                    ccsd_emb.e_tot
                    + self.e_env
                    + self.two_e_cross
                    - result["correction"]
                    - result["beta_correction"]
                )
                result["ccsd_emb"] = ccsd_emb.e_tot - e_nuc

                logger.info(f"CCSD Energy {projector_name}:\t{result['e_ccsd']}")

            if self.config.run_fci_emb is True:
                logger.debug("Performing FCI-in-DFT embedding.")
                fci_emb = self._run_emb_FCI(result["scf"], frozen=None)
                result["e_fci"] = (
                    (fci_emb.e_tot)
                    + self.e_env
                    + self.two_e_cross
                    - result["correction"]
                    - result["beta_correction"]
                )
                logger.info(f"FCI Energy {projector_name}:\t{result['e_fci']}")

                result["fci_emb"] = fci_emb.e_tot - e_nuc
            result["hf_emb"] = result["scf"].e_tot - e_nuc
            result["nuc"] = e_nuc

            if self.config.run_dft_in_dft is True:
                did = self._dft_in_dft(self._global_ks.xc, embedding_method)
                result["e_dft_in_dft"] = did["e_rks"]
                result["emb_dft"] = did["rks_e_elec"]

            logger.debug(f"{projector_name=}")
            if projector_name == "mu":
                self._mu = result
            elif projector_name == "huzinaga":
                self._huzinaga = result

        match self.config.projector:
            case Projector.MU:
                self.embedded_scf = self._mu["scf"]
                self.classical_energy = self._mu["classical_energy"]
            case Projector.HUZ:
                self.embedded_scf = self._huzinaga["scf"]
                self.classical_energy = self._huzinaga["classical_energy"]
            case Projector.BOTH:
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

        logger.info("Embedding complete.")
