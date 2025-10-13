"""Module containg the NbedDriver Class."""

import logging
from functools import cached_property, partial
from json import dump as jdump
from typing import Any, Literal, Union, assert_never

import numpy as np
from numpy.typing import NDArray
from pyscf import cc, dft, fci, gto, mcscf, qmmm, scf  # type:ignore
from pyscf.lib import NPArrayWithTag  # type:ignore

from nbed.localizers import (
    BOYSLocalizer,
    ConcentricLocalizer,
    IBOLocalizer,
    LocalizedSystem,
    OccupiedLocalizer,
    PAOLocalizer,
    PMLocalizer,
    SPADELocalizer,
)

from .config import (
    NbedConfig,
    OccupiedLocalizerTypes,
    ProjectorTypes,
    VirtualLocalizerTypes,
)
from .ham_builder import HamiltonianBuilder
from .scf import energy_elec
from .scf.huzinaga_scf import huzinaga_scf

type OneSpinMatrix[M: int] = np.ndarray[tuple[M, M], np.dtype[np.floating]]
type TwoSpinMatrix[M: int] = np.ndarray[tuple[Literal[2], M, M], np.dtype[np.floating]]
type AnySpinMatrix[M: int] = Union[OneSpinMatrix[M], TwoSpinMatrix[M]]

type FCISolver = (
    fci.direct_nosym.FCISolver
    | fci.direct_spin0.FCISolver
    | fci.direct_spin0_symm.FCISolver
    | fci.direct_spin1.FCISolver
    | fci.direct_spin1_symm.FCISolver
)

type AnyKS = dft.rks.RKS | dft.uks.UKS

# Create the Logger
logger = logging.getLogger(__name__)


class NbedDriver:
    """Function to return the embedding Qubit Hamiltonian.

    Args:
        config (NbedConfig): A validated config model.

    Attributes:
        _global_fci (scf.hf.SCF): A Qubit Hamiltonian of some kind
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
        self.config = config.model_validate(config)
        self.localized_system: LocalizedSystem
        self.two_e_cross: OneSpinMatrix | TwoSpinMatrix
        self.electron: int
        self.mu: dict = {}
        self.huzinaga: dict = {}
        self.active_geometry = f"{self.config.n_active_atoms}\n\n" + "\n".join(
            self.config.geometry.splitlines()[2 : 2 + self.config.n_active_atoms]
        )
        logger.debug(f"{self.active_geometry=}")
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

    def _build_mol(self) -> gto.Mole:
        """Function to build PySCF molecule.

        Returns:
            full_mol (gto.mol): built PySCF molecule object
        """
        logger.debug("Constructing molecule.")
        logger.debug("Molecule input geometry: %s", self.config.geometry)
        # geometry is raw xyz string
        full_mol = gto.Mole(
            atom=self.config.geometry[2:],
            basis=self.config.basis,
            charge=self.config.charge,
            unit=self.config.unit,
            spin=self.config.spin,
        ).build()
        logger.debug("Molecule built.")
        return full_mol

    @cached_property
    def _global_hf(self, **hf_kwargs) -> scf.hf.SCF:
        """Run full system Hartree-Fock."""
        logger.info("Running full system HF.")
        mol_full = self._build_mol()
        # run Hartree-Fock
        if self.config.restricted_global:
            global_hf = scf.RHF(mol_full, **hf_kwargs)
        else:
            global_hf = scf.UHF(mol_full, **hf_kwargs)
        global_hf.conv_tol = self.config.convergence
        global_hf.max_memory = self.config.max_ram_memory
        global_hf.max_cycle = self.config.max_hf_cycles
        global_hf.verbose = 1
        global_hf.run()
        logger.info(f"Global HF: {global_hf.e_tot}")

        return global_hf

    @cached_property
    def _global_ccsd(self, **ccsd_kwargs) -> cc.ccsd.CCSDBase:
        """Function to run full molecule CCSD calculation."""
        logger.info("Running full system CC.")
        # run CCSD after HF

        global_cc = cc.CCSD(self._global_hf, **ccsd_kwargs)
        global_cc.conv_tol = self.config.convergence
        global_cc.max_memory = self.config.max_ram_memory
        global_cc.verbose = 1
        global_cc.run()
        logger.info(f"Global CCSD: {global_cc.e_tot}")

        return global_cc

    @cached_property
    def _global_fci(self, **fci_kwargs) -> FCISolver:
        """Function to run full molecule FCI calculation.

        WARNING: FACTORIAL SCALING IN BASIS STATES!
        """
        logger.info("Running full system FCI.")
        # run FCI after HF
        global_fci = fci.FCI(self._global_hf, **fci_kwargs)
        global_fci.conv_tol = self.config.convergence
        global_fci.max_memory = self.config.max_ram_memory
        global_fci.verbose = 1

        global_fci.run()
        logger.info(f"Global FCI: {global_fci.e_tot}")

        return global_fci

    @cached_property
    def _global_ks(self, **ks_kwargs) -> AnyKS:
        """Method to run full cheap molecule UKS DFT calculation.

        Note this is necessary to perform localization procedure.
        """
        logger.info("Running full system KS DFT.")
        mol_full = self._build_mol()

        if self.config.restricted_global:
            global_ks = dft.rks.RKS(mol_full, **ks_kwargs)
        else:
            global_ks = dft.uks.UKS(mol_full, **ks_kwargs)

        logger.debug(f"{type(global_ks)=}")
        global_ks.conv_tol = self.config.convergence
        global_ks.xc = self.config.xc_functional
        global_ks.max_memory = self.config.max_ram_memory
        global_ks.max_cycle = self.config.max_dft_cycles
        global_ks.verbose = 1

        if self.run_qmmm:
            logger.debug(
                "QM/MM: running full system KS DFT in presence of point charges."
            )
            global_ks = qmmm.itrf.mm_charge(
                global_ks,
                self.config.mm_coords,
                self.config.mm_charges,
                self.config.mm_radii,
            )  # type: ignore

        global_ks.run()  # type: ignore
        logger.debug(f"{global_ks.mo_coeff.shape=}")  # type: ignore
        logger.debug(f"{global_ks.mo_occ.shape=}")  # type:ignore
        logger.debug(f"{global_ks.get_veff().shape=}")  # type: ignore
        logger.debug(f"{global_ks.get_hcore().shape=}")  # type: ignore
        logger.info(f"Global UKS: {global_ks.e_tot}")  # type: ignore

        if global_ks.converged is not True:  # type: ignore
            logger.warning("(cheap) global DFT calculation has NOT converged!")

        return global_ks  # type: ignore

    def _localize(self) -> LocalizedSystem:
        """Run the localizer class."""
        logger.debug(f"Getting localized system using {self.config.localization}.")

        localizer: OccupiedLocalizer
        match self.config.localization:
            case OccupiedLocalizerTypes.SPADE:
                localizer = SPADELocalizer(
                    self._global_ks,
                    self.config.n_active_atoms,
                    max_shells=self.config.max_shells,
                    n_mo_overwrite=self.n_mo_overwrite,
                )
            case OccupiedLocalizerTypes.BOYS:
                localizer = BOYSLocalizer(
                    self._global_ks,
                    self.config.n_active_atoms,
                    occ_cutoff=self.config.occupied_threshold,
                    virt_cutoff=self.config.virtual_threshold,
                )
            case OccupiedLocalizerTypes.IBO:
                localizer = IBOLocalizer(
                    self._global_ks,
                    self.config.n_active_atoms,
                    occ_cutoff=self.config.occupied_threshold,
                    virt_cutoff=self.config.virtual_threshold,
                )
            case OccupiedLocalizerTypes.PM:
                localizer = PMLocalizer(
                    self._global_ks,
                    self.config.n_active_atoms,
                    occ_cutoff=self.config.occupied_threshold,
                    virt_cutoff=self.config.virtual_threshold,
                )
            case _:
                raise ValueError(
                    "Invalid Localizer in config %s", self.config.localization
                )

        self.localizer = localizer
        return localizer.localize()

    def _init_local_hf(self) -> scf.hf.SCF:
        """Function to build embedded HF object for active subsystem.

        Note this function overwrites the total number of electrons to only include active number.

        Returns:
            local_hf (scf.uhf.UHF or scf.ROHF): embedded Hartree-Fock object.
        """
        logger.debug("Constructing localised HF object.")
        embedded_mol: gto.Mole = self._init_embedded_mol()

        if self.config.restricted_active:
            local_hf = scf.RHF(embedded_mol)
        else:
            local_hf = scf.UHF(embedded_mol)

        logger.debug(f"{embedded_mol.nelectron=}")
        logger.debug(f"{embedded_mol.nelec=}")
        logger.debug(f"{embedded_mol.spin=}")

        if self.run_qmmm:
            logger.debug("QM/MM: running local SCF in presence of point charges.")
            local_hf = qmmm.itrf.mm_charge(
                local_hf,
                self.config.mm_coords,
                self.config.mm_charges,
                self.config.mm_radii,
            )  # type: ignore

        local_hf.run()  # type: ignore
        local_hf.max_memory = self.config.max_ram_memory  # type:ignore
        local_hf.conv_tol = self.config.convergence  # type:ignore
        local_hf.max_cycle = self.config.max_hf_cycles  # type:ignore
        local_hf.verbose = 1  # type:ignore

        return local_hf  # type:ignore

    def _convert_localized_system(self, ls: LocalizedSystem):
        """Convert a Localized system between spin restriction types."""

    def _init_embedded_mol(self) -> gto.Mole:
        """Create a pyscf molecule for the embedded system.

        Returns:
            gto.Mole: An embedded molecule object.
        """
        embedded_mol: gto.Mole = self._build_mol()
        match self.localized_system.active_occ_inds.ndim:
            case 1:
                n_elec = np.count_nonzero(self.localized_system.active_occ_inds)
                logger.debug(f"embedded nelec {n_elec}")

                embedded_mol.nelectron = 2 * n_elec
                embedded_mol.nelec = (n_elec, n_elec)
                embedded_mol.spin = 0
                self._electron = embedded_mol.nelectron
            case 2:
                n_elec_alpha = np.count_nonzero(
                    self.localized_system.active_occ_inds[0, :]
                )
                n_elec_beta = np.count_nonzero(
                    self.localized_system.active_occ_inds[1, :]
                )
                logger.debug(f"embedded nelec {n_elec_alpha, n_elec_beta}")

                embedded_mol.nelectron = n_elec_alpha + n_elec_beta
                embedded_mol.nelec = (n_elec_alpha, n_elec_beta)
                embedded_mol.spin = int(n_elec_alpha - n_elec_beta)
                self._electron = embedded_mol.nelectron
        return embedded_mol

    def _init_local_ks(self, xc_functional: str) -> AnyKS:
        """Function to build embedded Hartree Fock object for active subsystem.

        Note this function overwrites the total number of electrons to only include active number.

        Args:
            xc_functional (str): XC functional to use in embedded calculation.

        Returns:
            local_ks (pyscf.dft.RKS or pyscf.dft.uks.UKS): embedded Kohn-Sham DFT object.
        """
        logger.debug("Initialising localised RKS object.")
        embedded_mol: gto.Mole = self._init_embedded_mol()

        if self.config.restricted_active:
            local_ks = dft.rks.RKS(embedded_mol)
        else:
            local_ks = dft.uks.UKS(embedded_mol)
        logger.debug(f"{embedded_mol.nelectron=}")
        logger.debug(f"{embedded_mol.nelec=}")
        logger.debug(f"{embedded_mol.spin=}")

        local_ks.max_memory = self.config.max_ram_memory
        local_ks.conv_tol = self.config.convergence
        local_ks.xc = xc_functional
        local_ks.verbose = 1

        return local_ks

    def _subsystem_dft(
        self, global_ks: AnyKS, localized_system: LocalizedSystem
    ) -> tuple[float, float, np.typing.NDArray]:
        """Function to perform subsystem UKS DFT calculation."""
        logger.debug("Calculating active and environment subsystem terms.")

        def _ks_components[Shape, DType](
            ks_system: AnyKS,
            subsystem_dm: AnySpinMatrix,
        ) -> tuple[float, NPArrayWithTag, AnySpinMatrix]:
            """Calculate the components of subsystem energy from a UKS DFT calculation.

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
            logger.debug("Finding subsystem UKS componenets.")
            # It seems that PySCF lumps J and K in the J array
            # need to access the potential for the right subsystem for unrestricted
            logger.debug(f"{subsystem_dm.shape=}")
            two_e_term: NPArrayWithTag = ks_system.get_veff(dm=subsystem_dm)
            j_mat = ks_system.get_j(dm=subsystem_dm)

            if subsystem_dm.ndim == 3:
                dm_tot = subsystem_dm[0] + subsystem_dm[1]
            else:
                dm_tot = subsystem_dm
            logger.debug(f"{dm_tot.shape=}")

            e_act: float = (
                np.einsum("ij,ji->", ks_system.get_hcore(), dm_tot)
                + two_e_term.ecoul  # type: ignore
                + two_e_term.exc  # type: ignore
            )

            # if check_E_with_pyscf:
            #     energy_elec_pyscf = global_ks.energy_elec(dm=dm_matrix)[0]
            #     if not np.isclose(energy_elec_pyscf, energy_elec):
            #         raise ValueError("Energy calculation incorrect")
            logger.debug("Subsystem UKS components found.")
            logger.debug(f"{e_act=}")
            logger.debug(f"{two_e_term.shape=}")
            return e_act, two_e_term, j_mat

        dm_act = localized_system.dm_active
        dm_env = localized_system.dm_enviro

        (e_act, two_e_act, j_act) = _ks_components(global_ks, dm_act)
        # logger.debug(e_act, alpha_e_xc_act)
        (e_env, two_e_env, j_env) = _ks_components(global_ks, dm_env)
        # logger.debug(alpha_e_env, alpha_e_xc_env, alpha_ecoul_env)

        # Computing cross subsystem terms
        logger.debug("Calculating two electron cross subsystem energy.")
        total_dm = localized_system.dm_active + localized_system.dm_enviro

        if localized_system.dm_active.ndim == 3:
            total_dm = total_dm[0, :, :] + total_dm[1, :, :]

        two_e_term_total: NPArrayWithTag = global_ks.get_veff(dm=total_dm)
        logger.debug(f"{total_dm.shape=}")
        logger.debug(f"{two_e_term_total.shape=}")
        e_xc_total: float = two_e_term_total.exc  # type: ignore

        match localized_system.dm_active.ndim:
            case 2:
                j_cross = 0.5 * (
                    np.einsum("ij,ij", localized_system.dm_active, j_env)
                    + np.einsum("ij,ij", localized_system.dm_enviro, j_act)
                )
            case 3:
                j_cross = 0.5 * (
                    np.einsum("ij,ij", localized_system.dm_active[0], j_env[0])
                    + np.einsum("ij,ij", localized_system.dm_enviro[0], j_act[0])
                    + np.einsum("ij,ij", localized_system.dm_active[0], j_env[1])
                    + np.einsum("ij,ij", localized_system.dm_enviro[0], j_act[1])
                    + np.einsum("ij,ij", localized_system.dm_active[1], j_env[1])
                    + np.einsum("ij,ij", localized_system.dm_enviro[1], j_act[1])
                    + np.einsum("ij,ij", localized_system.dm_active[1], j_env[0])
                    + np.einsum("ij,ij", localized_system.dm_enviro[1], j_act[0])
                )
            case _:
                raise ValueError("Active density matrix not valid shape.")

        logger.debug(f"{j_cross=}")

        # Because of projection we expect kinetic term to be zero
        k_cross = 0.0

        xc_cross: float = e_xc_total - two_e_act.exc - two_e_env.exc  # type: ignore
        logger.debug(f"{e_xc_total=}")
        logger.debug(f"{two_e_act.exc=}")  # type: ignore
        logger.debug(f"{two_e_env.exc=}")  # type: ignore

        # overall two_electron cross energy
        two_e_cross = j_cross + k_cross + xc_cross

        logger.debug("UKS components")
        logger.debug(f"e_act: {e_act}")
        logger.debug(f"e_env: {e_env}")
        logger.debug(f"two_e_cross: {two_e_cross}")
        logger.debug(f"e_nuc: {global_ks.energy_nuc()}")
        return e_act, e_env, two_e_cross

    @cached_property
    def _env_projector(self) -> OneSpinMatrix | TwoSpinMatrix:
        """Return a projector onto the environment in orthogonal basis."""
        return _env_projector(
            self._global_ks.get_ovlp(), self.localized_system.dm_enviro
        )

    def _run_emb_ccsd(
        self,
        emb_pyscf_scf_rhf: scf.hf.SCF,
        frozen: list[int] | None = None,
    ) -> tuple[cc.ccsd.CCSDBase, float]:
        """Function run CCSD on embedded restricted Hartree Fock object.

        Note emb_pyscf_scf_rhf is ROHF object for the active embedded subsystem (defined in localized basis)
        (see get_embedded_rhf method)

        Args:
            emb_pyscf_scf_rhf (pyscf.scf.hf.SCF): PySCF restricted Hartree Fock object of active embedded subsystem
            frozen (list[int]): A path to an .xyz file describing molecular geometry.

        Returns:
            ccsd (pyscf.cc.ccsd.CCSDBase): PySCF CCSD object
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
        emb_pyscf_scf_rhf: scf.hf.SCF,
        frozen: list[int] | None = None,
    ) -> FCISolver | scf.hf.SCF | mcscf.casci.CASBase:
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
        self, localized_scf: scf.hf.SCF, embedding_potential: np.ndarray
    ) -> tuple[scf.hf.SCF, np.ndarray]:
        """Embed using the Mu-shift projector.

        Args:
            localized_scf (scf.hf.SCF): A PySCF scf method with the correct number of electrons for the active region.
            embedding_potential (np.ndarray): Potential calculated from two electron terms in dft.

        Returns:
            np.ndarray: Matrix form of the embedding potential.
            scf.hf.SCF: The embedded scf object.
        """
        logger.debug("Running mu embedded scf calculation.")

        # Modify the energy_elec function to handle different h_cores
        # which we need for different embedding potentials

        v_emb = (self.config.mu_level_shift * self._env_projector) + embedding_potential
        logger.debug(f"{v_emb.shape=}")

        if v_emb.ndim == 3:
            # localized_scf.energy_elec = lambda **kwargs: energy_elec(
            #     localized_scf, **kwargs
            # )
            localized_scf.energy_elec = partial(energy_elec, localized_scf)

        logger.debug(f"{v_emb.shape=}")
        logger.debug(f"{self._env_projector.shape=}")
        logger.debug(f"{embedding_potential.shape=}")
        hcore_std = localized_scf.get_hcore
        logger.debug(f"{hcore_std().shape=}")
        setattr(localized_scf, "get_hcore_std", hcore_std)
        if not hasattr(localized_scf, "v_emb"):
            setattr(localized_scf, "v_emb", v_emb)
        else:
            raise ValueError("Localized SCF already has v_emb attribute.")

        localized_scf.get_hcore = lambda *args: hcore_std(*args) + v_emb  # type:ignore
        # veff_std = localized_scf.get_veff
        # localized_scf.get_veff = lambda *args: veff_std(*args) + v_emb
        logger.debug(f"embedded hcore shape {localized_scf.get_hcore().shape}")
        localized_scf.kernel()  # type:ignore
        logger.info(
            f"Embedded scf energy MU_SHIFT: {localized_scf.e_tot}, converged: {localized_scf.converged}"
        )

        return localized_scf, v_emb

    def _huzinaga_embed(
        self,
        active_scf: scf.hf.SCF,
        embedding_potential: np.ndarray,
        localized_system: LocalizedSystem,
        dmat_initial_guess: OneSpinMatrix | TwoSpinMatrix | None = None,
    ) -> tuple[scf.hf.SCF, OneSpinMatrix | TwoSpinMatrix]:
        """Embed using Huzinaga projector.

        Args:
            active_scf (scf.hf.SCF): A PySCF scf method with the correct number of electrons for the active region.
            embedding_potential (np.ndarray): Potential calculated from two electron terms in dft.
            localized_system (LocalizedSystem): Dataclass describing the MOs of a localized system.
            dmat_initial_guess (bool): If True, use the initial guess for the density matrix.

        Returns:
            np.ndarray: Matrix form of the embedding potential.
            scf.hf.SCF: The embedded scf object.
        """
        logger.info("Starting Huzinaga embedding method...")
        # We need to run our own SCF method here to update the potential.

        if localized_system.c_loc_virt is not None:
            virtual_projector = np.einsum(
                "...ij,...jk->...ik",
                localized_system.c_loc_virt,
                localized_system.c_loc_virt.swapaxes(-1, -2),
            )
            dm_environment_virtual = (
                np.identity(localized_system.c_loc_virt.shape[-2])
                - localized_system.dm_loc_occ
                - virtual_projector
            )
        else:
            dm_environment_virtual = None

        (
            c_active_embedded,
            mo_embedded_energy,
            dm_active_embedded,
            huzinaga_op_std,
            huz_scf_conv_flag,
        ) = huzinaga_scf(
            active_scf,
            embedding_potential,
            localized_system.dm_enviro,
            dm_environment_virtual=dm_environment_virtual,
            dm_conv_tol=1e-6,
            dm_initial_guess=dmat_initial_guess,
        )

        logger.debug(f"{c_active_embedded=}")

        # write results to pyscf object
        logger.debug("Writing results to PySCF object.")
        hcore_std = active_scf.get_hcore
        v_emb = huzinaga_op_std + embedding_potential
        active_scf.get_hcore = lambda *args: hcore_std(*args) + v_emb  # type:ignore

        if localized_system.dm_active.ndim == 3:
            active_scf.energy_elec = partial(energy_elec, active_scf)

        active_scf.mo_occ = active_scf.get_occ(mo_embedded_energy, c_active_embedded)

        if localized_system.c_loc_virt is not None:
            logger.debug("Overwriting embedded virtuals with result from localizer.")
            logger.debug(f"{np.sum(active_scf.mo_occ, axis=0)}")
            logger.debug(
                f"{c_active_embedded[..., np.sum(active_scf.mo_occ, axis=0)> 0].shape=}"
            )
            logger.debug(f"{localized_system.c_loc_virt.shape=}")
            active_scf.mo_coeff = np.concatenate(
                (
                    c_active_embedded[..., np.sum(active_scf.mo_occ, axis=0) > 0],
                    c_active_embedded[..., np.sum(active_scf.mo_occ, axis=0) == 0][
                        : localized_system.c_loc_virt.shape[-1]
                    ],
                ),
                axis=2,
            )  # type:ignore
            active_scf.mo_occ = active_scf.mo_occ[: active_scf.mo_coeff.shape[-1]]
        else:
            active_scf.mo_coeff = c_active_embedded  # type:ignore

        logger.debug(f"{active_scf.mo_occ=}")
        logger.debug(f"{active_scf.mo_coeff.shape=}")  # type:ignore
        active_scf.mo_energy = mo_embedded_energy  # type:ignore
        active_scf.e_tot = active_scf.energy_tot(dm=dm_active_embedded)
        # active_scf.conv_check = huz_scf_conv_flag
        active_scf.converged = huz_scf_conv_flag

        logger.info(f"Embedded scf energy HUZINAGA: {active_scf.e_tot}")
        return active_scf, v_emb

    def _delete_environment(
        self,
        projector: ProjectorTypes,
        scf_object: scf.hf.SCF,
        localized_system: LocalizedSystem,
        env_projector: NDArray,
    ) -> scf.hf.SCF:
        """Remove enironment orbit from embedded ROHF object.

        This function removes (in fact deletes completely) the molecular orbitals
        defined by the environment of the localized system.

        Args:
            projector (ProjectorTypes): The projector used to embed the system.
            scf_object (scf.hf.SCF): The embedded SCF object.
            localized_system (LocalizedSystem): Occupied Localization results for a molecule.
            env_projector (NDArray): Projector onto the environment region.

        Returns:
            scf.hf.SCF: Returns input, but with environment orbitals deleted.
        """
        logger.debug("Deleting environment from SCF object.")

        match scf_object:
            # Restricted Closed
            case scf.rhf.RHF() | dft.rks.RKS():
                n_env_mos = np.sum(localized_system.enviro_occ_inds, dtype=int)
                logger.debug(f"{n_env_mos=}")
                scf_object.mo_coeff, scf_object.mo_energy, scf_object.mo_occ = (  # type:ignore
                    _delete_spin_environment(  # type:ignore
                        projector,
                        n_env_mos,
                        scf_object.mo_coeff,  # type:ignore
                        scf_object.mo_energy,  # type:ignore
                        scf_object.mo_occ,  # type:ignore
                        env_projector,
                    )
                )
            # Resticted Open
            case scf.rohf.ROHF() | dft.roks.ROKS():
                n_env_mos = [
                    np.sum(localized_system.enviro_occ_inds[0]),
                    np.sum(localized_system.enviro_occ_inds[1]),
                ]
                logger.debug(f"{n_env_mos=}")
                (
                    mo_coeff_alpha,
                    mo_energy_alpha,
                    mo_occ_alpha,
                ) = _delete_spin_environment(
                    projector,
                    n_env_mos[0],
                    scf_object.mo_coeff,  # type:ignore
                    scf_object.mo_energy,  # type:ignore
                    scf_object.mo_occ[0],  # type:ignore
                    env_projector[0],
                )
                (mo_coeff_beta, mo_energy_beta, mo_occ_beta) = _delete_spin_environment(
                    projector,
                    n_env_mos[1],
                    scf_object.mo_coeff,  # type:ignore
                    scf_object.mo_energy,  # type:ignore
                    scf_object.mo_occ[1],  # type:ignore
                    env_projector[1],
                )
                # Need to do it this way or there are broadcasting issues
                scf_object.mo_coeff = np.array(
                    [mo_coeff_alpha, mo_coeff_beta]
                )  # np.array([mo_coeff[0], mo_coeff[1]]) #type:ignore
                scf_object.mo_energy = np.array(
                    [mo_energy_alpha, mo_energy_beta]
                )  # np.array([mo_energy[0], mo_energy[1]]) #type:ignore
                scf_object.mo_occ = np.array(
                    [mo_occ_alpha, mo_occ_beta]
                )  # np.array([mo_occ[0], mo_occ[1]]) #type:ignore            # Unrestrected
            case scf.uhf.UHF() | dft.uks.UKS():
                #
                n_env_mos = [
                    np.sum(localized_system.enviro_occ_inds[0]),
                    np.sum(localized_system.enviro_occ_inds[1]),
                ]
                logger.debug(f"{n_env_mos=}")
                (
                    mo_coeff_alpha,
                    mo_energy_alpha,
                    mo_occ_alpha,
                ) = _delete_spin_environment(
                    projector,
                    n_env_mos[0],
                    scf_object.mo_coeff[0],  # type:ignore
                    scf_object.mo_energy[0],  # type:ignore
                    scf_object.mo_occ[0],  # type:ignore
                    env_projector[0],
                )
                (mo_coeff_beta, mo_energy_beta, mo_occ_beta) = _delete_spin_environment(
                    projector,
                    n_env_mos[1],
                    scf_object.mo_coeff[1],  # type:ignore
                    scf_object.mo_energy[1],  # type:ignore
                    scf_object.mo_occ[1],  # type:ignore
                    env_projector[1],
                )
                # Need to do it this way or there are broadcasting issues
                scf_object.mo_coeff = np.array(
                    [mo_coeff_alpha, mo_coeff_beta]
                )  # np.array([mo_coeff[0], mo_coeff[1]]) #type:ignore
                scf_object.mo_energy = np.array(
                    [mo_energy_alpha, mo_energy_beta]
                )  # np.array([mo_energy[0], mo_energy[1]]) #type:ignore
                scf_object.mo_occ = np.array(
                    [mo_occ_alpha, mo_occ_beta]
                )  # np.array([mo_occ[0], mo_occ[1]]) #type:ignore
            case _:
                logger.error("Combination of C matrix and occupancy shape not valid.")
                raise ValueError(
                    "SCF Object not instance of Restricted or Unrestrictd."
                )

        logger.debug("Environment deleted.")
        return scf_object

    def _dft_in_dft(self, projection_method: ProjectorTypes) -> dict:
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
        logger.info("Beginning embedding...")
        if self.config.virtual_localization is VirtualLocalizerTypes.PROJECTED_AO:
            raise NotImplementedError("PAO not yet fully implemented.")

        self.e_nuc = self._global_ks.energy_nuc()

        if n_mo_overwrite is not None and n_mo_overwrite != (None, None):
            logger.debug(
                "Setting n_mo_overwrite with value from embed args %s", n_mo_overwrite
            )
            self.n_mo_overwrite = n_mo_overwrite
        else:
            logger.debug("Setting n_mo_overwrite with value from config.")
            self.n_mo_overwrite = self.config.n_mo_overwrite

        logger.info("Localizing occupied orbitals...")
        self.localized_system = self._localize()

        logger.info("Projecting out environment...")
        # Run subsystem DFT (calls localized rks)
        self.e_act, self.e_env, self.two_e_cross = self._subsystem_dft(
            self._global_ks, self.localized_system
        )
        logger.debug("Getting global DFT embedding potential.")

        total_dm = self.localized_system.dm_active + self.localized_system.dm_enviro

        g_act_and_env = self._global_ks.get_veff(dm=total_dm)

        g_act = self._global_ks.get_veff(dm=self.localized_system.dm_active)

        embedding_potential = g_act_and_env - g_act
        self.embedding_potential = embedding_potential

        logger.debug(f"DFT potential average {np.mean(embedding_potential)}.")

        # logger.debug("converting localized system")
        # if self.config.restricted_global and isinstance(
        #     self.localized_system, RestrictedLS
        # ):
        #     self.localized_system = UnrestrictedLS.from_spin_components(
        #         self.localized_system, self.localized_system
        #     )
        #     self.embedding_potential = np.array(
        #         [self.embedding_potential, self.embedding_potential]
        #     )

        logger.debug("Beginning Projection.")
        if (
            self.config.projector in [ProjectorTypes.MU, ProjectorTypes.BOTH]
            or init_huzinaga_rhf_with_mu
        ):
            local_hf = self._init_local_hf()

            if self.config.virtual_localization == VirtualLocalizerTypes.PROJECTED_AO:
                raise NotImplementedError(
                    "Projected Atomic Orbitals defined only for Huzinaga projector."
                )

            embedded_scf, v_emb = self._mu_embed(local_hf, embedding_potential)
            self.mu = self.post_embed(embedded_scf, v_emb, ProjectorTypes.MU)

        if self.config.projector in [ProjectorTypes.HUZ, ProjectorTypes.BOTH]:
            local_hf = self._init_local_hf()

            dmat_initial_guess: NDArray | None = (
                self.mu["scf"].make_rdm1() if init_huzinaga_rhf_with_mu else None
            )

            if self.config.virtual_localization == VirtualLocalizerTypes.PROJECTED_AO:
                logger.debug("Updating localized system with PAO virtual orbitals.")
                pao = PAOLocalizer(
                    local_hf,
                    self.config.n_active_atoms,
                    self.localized_system.c_loc_occ,
                    norm_cutoff=self.config.norm_cutoff,
                    overlap_cutoff=self.config.overlap_cutoff,
                )
                pao_mo_coeff = pao.localize_virtual()
                self.localized_system.c_loc_virt = pao_mo_coeff

            embedded_scf, v_emb = self._huzinaga_embed(
                local_hf, embedding_potential, self.localized_system, dmat_initial_guess
            )
            self.huzinaga = self.post_embed(embedded_scf, v_emb, ProjectorTypes.HUZ)

        match self.config.projector:
            case ProjectorTypes.MU:
                self.embedded_scf = self.mu["scf"]
                self.classical_energy = self.mu["classical_energy"]
            case ProjectorTypes.HUZ:
                self.embedded_scf = self.huzinaga["scf"]
                self.classical_energy = self.huzinaga["classical_energy"]
            case ProjectorTypes.BOTH:
                logger.warning(
                    "Outputting both mu and huzinaga embedding results as tuple."
                )
                self.embedded_scf = (
                    self.mu["scf"],
                    self.huzinaga["scf"],
                )
                self.classical_energy = (
                    self.mu["classical_energy"],
                    self.huzinaga["classical_energy"],
                )
            case _:
                logger.error("Projector did not match any know case.")
                logger.warning("Not assigning embedded_scf or classial_energy")
                raise ValueError(
                    "Projector %s did not match any know case.", self.config.projector
                )

        if filename := self.config.savefile is not None:
            logger.debug("Saving results to file %s", filename)
            with open(filename, "w") as f:
                jdump({"mu": self.mu, "huzinaga": self.huzinaga}, f)

        logger.info("Embedding complete.")

    def post_embed(
        self, embedded_scf: scf.hf.SCF, v_emb: NDArray, projector: ProjectorTypes
    ) -> dict:
        """Projector-dependent components of the embedding procedure.

        Args:
            embedded_scf (scf.hf.SCF): An embedded pyscf scf object.
            v_emb (NDArray): Embedding Potential
            projector (ProjectorTypes): Which projector the result should use.

        Returns:
            dict: A dict of results.
        """
        logger.info("Deleting environment orbitals...")
        result: dict[str, Any] = {}
        result["scf"] = embedded_scf.copy()
        result["v_emb"] = v_emb
        result["mo_energies_emb_pre_del"] = result["scf"].mo_energy
        result["scf"] = self._delete_environment(
            projector, result["scf"], self.localized_system, self._env_projector
        )
        result["mo_energies_emb_post_del"] = result["scf"].mo_energy

        logger.info(f"V emb mean {projector}: {np.mean(result['v_emb'])}")

        # calculate correction
        match self.localized_system.dm_active.ndim:
            case 2:
                result["correction"] = np.einsum(
                    "ij,ij", result["v_emb"], self.localized_system.dm_active
                )
                result["beta_correction"] = 0
            case 3:
                result["correction"] = np.einsum(
                    "ij,ij", result["v_emb"][0], self.localized_system.dm_active[0]
                )
                result["beta_correction"] = np.einsum(
                    "ij,ij", result["v_emb"][1], self.localized_system.dm_active[1]
                )

        # Post-embedding Virtual localization
        match self.config.virtual_localization:
            case VirtualLocalizerTypes.CONCENTRIC:
                logger.info("Performing Concentric Localization of virtuals ...")
                result["cl"] = ConcentricLocalizer(
                    result["scf"],
                    self.config.n_active_atoms,
                    max_shells=self.config.max_shells,
                )
                result["scf"] = result["cl"].localize_virtual()
            case VirtualLocalizerTypes.DISABLE:
                logger.debug("Not performing virtual localization.")
            case _:
                logger.debug(
                    f"Driver does not have a method implemented for {self.config.virtual_localization}"
                )

        logger.info("Collcting results...")
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
                (fci_emb.e_tot)  # type:ignore
                + self.e_env  # type:ignore
                + self.two_e_cross
                - result["correction"]
                - result["beta_correction"]
            )  # type:ignore
            logger.info(f"FCI Energy {projector}:\t{result['e_fci']}")

            result["fci_emb"] = fci_emb.e_tot - self.e_nuc
        result["hf_emb"] = result["scf"].e_tot - self.e_nuc

        if self.config.run_dft_in_dft is True:
            did = self._dft_in_dft(projector)
            result.update(did)
        if self.config.build_hamiltonian:
            # Build second quantised Hamiltonian
            hb = HamiltonianBuilder(result["scf"], result["classical_energy"])
            result["second_quantised"] = hb.build()

        logger.debug(f"Found result for {projector}")
        logger.debug(result)

        return result


def run_emb_fci(
    emb_pyscf_scf_rhf: scf.hf.SCF,
    frozen: list | None = None,
    convergence: float | None = 1e-6,
    max_ram_memory: int | None = 4000,
) -> scf.hf.SCF | FCISolver | mcscf.casci.CASBase:
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
        fci_scf = mcscf.CASSCF(
            emb_pyscf_scf_rhf,
            emb_pyscf_scf_rhf.mol.nelec,
            emb_pyscf_scf_rhf.mol.nao - len(frozen),
        )
        fci_scf.sort_mo(  # type:ignore
            [i + 1 for i in range(emb_pyscf_scf_rhf.mol.nao) if i not in frozen]
        )  # type:ignore
        if not isinstance(fci_scf, mcscf.casci.CASBase):
            raise NotImplementedError("Check Embedded FCI parameters.")

    fci_scf.conv_tol = convergence  # type:ignore
    fci_scf.max_memory = max_ram_memory  # type:ignore
    fci_scf.verbose = 1  # type:ignore

    # For UHF, PySCF assumes that hcore is spinless and 2D
    # Because we update hcore for embedding, we need to calculate our own h1e term.
    from functools import reduce

    h_core = emb_pyscf_scf_rhf.get_hcore()
    if np.ndim(h_core) == 3 and frozen is None:
        mo: NDArray = emb_pyscf_scf_rhf.mo_coeff  # type: ignore
        h1e = [
            reduce(np.dot, (mo[0].T, h_core[0], mo[0])),
            reduce(np.dot, (mo[1].T, h_core[1], mo[1])),
        ]
        fci_scf.kernel(h1e=h1e)  # type:ignore
    else:
        # kernel function default value is passed in
        fci_scf.kernel()  # type:ignore
    logger.info(f"FCI embedding energy: {fci_scf.e_tot}")  # type:ignore
    return fci_scf


def run_emb_ccsd(
    emb_pyscf_scf_rhf: scf.hf.SCF,
    frozen: list | None = None,
    convergence: float = 1e-6,
    max_ram_memory: int = 4000,
) -> tuple[cc.ccsd.CCSDBase, float]:
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
    ccsd.verbose = 2

    e_ccsd_corr: float
    e_ccsd_corr, _, _ = ccsd.kernel()  # type:ignore
    logger.info(f"Embedded CCSD energy: {e_ccsd_corr}")
    logger.info(f"CCSD Converged {ccsd.converged}")
    return ccsd, e_ccsd_corr  # type:ignore


def _env_projector(
    s_mat: np.ndarray[tuple[int, int]],
    dm_enviro: np.ndarray[tuple[int, int] | tuple[int, int, int]],
):
    """Return a projector onto the environment in orthogonal basis.

    Args:
        s_mat (np.ndarray): The AO overlap matrix.
        dm_enviro (np.ndarray): Environment Density Matrix

    Returns:
        np.ndaray: Projector onto the environment.
    """
    logger.debug("Getting Environment Projector.")
    logger.debug(f"{s_mat.shape=}")
    env_projector = np.einsum("ij,...jk,kl->...il", s_mat, dm_enviro, s_mat)
    logger.debug(f"{env_projector.shape=}")
    return env_projector


def _delete_spin_environment(
    projector: ProjectorTypes,
    n_env_mo: int,
    mo_coeff: np.ndarray[tuple[int, int]],
    mo_energy: np.ndarray[tuple[int]],
    mo_occ: np.ndarray[tuple[int]],
    environment_projector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove enironment orbit from embedded ROHF object.

    This function removes (in fact deletes completely) the molecular orbitals
    defined by the environment of the localized system

    Args:
        projector (ProjectorTypes): The projector used to embed the system.
        n_env_mo (int): The number of molecular orbitals in the environment.
        mo_coeff (np.ndarray): The molecular orbitals.
        mo_energy (np.ndarray): The molecular orbital energies.
        mo_occ (np.ndarray): The molecular orbital occupation numbers.
        environment_projector (np.ndarray): Matrix to project mo_coeff onto environment.

    Returns:
        embedded_rhf (scf.hf.SCF): Returns input, but with environment orbitals deleted
    """
    logger.debug("Deleting environment for spin.")
    logger.debug(f"{projector=}")
    logger.debug(f"{n_env_mo=}")
    logger.debug(f"{mo_coeff.shape=}")
    logger.debug(f"{mo_energy=}")
    logger.debug(f"{mo_occ=}")
    logger.debug(f"{environment_projector.shape=}")

    frozen_enviro_orb_inds: list[int] = []
    match projector:
        case ProjectorTypes.HUZ:
            # MOs which have the greatest overlap with the
            overlap: NDArray[np.floating] = np.einsum(
                "ij, ki -> i",
                mo_coeff.swapaxes(-1, -2),
                environment_projector @ mo_coeff,
            )
            overlap_by_size: NDArray[np.integer] = overlap.argsort()[::-1]
            logger.debug(f"{overlap_by_size=}")
            frozen_enviro_orb_inds = list(overlap_by_size[:n_env_mo])

        case ProjectorTypes.MU:
            # Orbitals which have been shifted to have energy mu are removed
            shift = mo_coeff.shape[-1] - n_env_mo
            logger.debug(f"{shift=}")
            logger.debug(f"{mo_coeff.shape=}")
            logger.debug(f"{n_env_mo=}")
            frozen_enviro_orb_inds = [mo_i for mo_i in range(shift, mo_coeff.shape[-1])]
        case ProjectorTypes.BOTH:
            raise ValueError("Projector must be specified to delete environment.")
        case _:
            assert_never(projector)

    active_MOs_occ_and_virt_embedded = [
        mo_i for mo_i in range(mo_coeff.shape[-1]) if mo_i not in frozen_enviro_orb_inds
    ]

    logger.info(
        f"Orbital indices for embedded system: {active_MOs_occ_and_virt_embedded}"
    )
    logger.info(
        f"Orbital indices removed from embedded system: {frozen_enviro_orb_inds}"
    )

    # delete enviroment orbitals and associated energies
    # overwrites varibles keeping only active part (both occupied and virtual)
    active_mo_coeff = mo_coeff[..., active_MOs_occ_and_virt_embedded]
    active_mo_energy = mo_energy[..., active_MOs_occ_and_virt_embedded]
    active_mo_occ = mo_occ[active_MOs_occ_and_virt_embedded]

    logger.debug("Spin environment deleted.")
    logger.debug(f"{active_mo_coeff=}")
    logger.debug(f"{active_mo_energy=}")
    logger.debug(f"{active_mo_occ=}")
    return active_mo_coeff, active_mo_energy, active_mo_occ


def dft_in_dft(driver: "NbedDriver", projection_method: ProjectorTypes) -> dict:
    """Return energy of DFT in DFT embedding.

    Note run_mu_shift (bool) and run_huzinaga (bool) flags define which method to use (can be both)
    This is done when object is initialized.

    Args:
        driver (NbedDriver): A driver object.
        projection_method (callable): Embedding method to use (mu or huzinaga).

    Returns:
        dict: DFT-in-DFT embedding results.
    """
    result: dict[str, Any] = {}
    e_nuc = driver._global_ks.energy_nuc()

    scf_object = driver._init_local_ks(driver._global_ks.xc)
    hcore_std = scf_object.get_hcore()
    match projection_method:
        case ProjectorTypes.MU:
            scf_object, v_emb_dft = driver._mu_embed(
                scf_object, driver.embedding_potential
            )
            result["v_emb_dft"] = v_emb_dft
        case ProjectorTypes.HUZ:
            scf_object, v_emb_dft = driver._huzinaga_embed(
                scf_object,
                driver.embedding_potential,
                driver.localized_system,
            )
            result["v_emb_dft"] = v_emb_dft
        case ProjectorTypes.BOTH:
            raise ValueError("Cannot use BOTH projector for DFT-in-DFT.")
        case _:
            assert_never(projection_method)

    result["scf_dft"] = driver._delete_environment(
        projection_method,
        scf_object,
        driver.localized_system,
        driver._env_projector,
    )

    match driver.localized_system.dm_active.ndim:
        case 2:
            y_emb = result["scf_dft"].make_rdm1()  # type:ignore

            # calculate correction
            result["dft_correction"] = np.einsum(
                "ij,ij",
                result["v_emb_dft"],
                (y_emb - driver.localized_system.dm_active),
            )  # type:ignore
            veff = result["scf_dft"].get_veff(dm=y_emb)  # type:ignore
            result["dft_correction_beta"] = 0.0
            rks_e_elec = veff.exc + veff.ecoul + np.einsum("ij,ij", hcore_std, y_emb)

        case 3:
            y_emb_alpha, y_emb_beta = result["scf_dft"].make_rdm1()  # type:ignore

            # calculate correction
            result["dft_correction"] = np.einsum(
                "ij,ij",
                result["v_emb_dft"][0],
                (y_emb_alpha - driver.localized_system.dm_active[0]),
            )

            result["dft_correction_beta"] = np.einsum(
                "ij,ij",
                result["v_emb_dft"][1],
                (y_emb_beta - driver.localized_system.dm_active[1]),
            )

            veff = result["scf_dft"].get_veff(dm=[y_emb_alpha, y_emb_beta])  # type:ignore

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
        case _:
            raise ValueError("Active DM Shape not valid.")

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
