"""Module containg the NbedDriver Class."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy as sp
from cached_property import cached_property
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.ops import QubitOperator
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import bravyi_kitaev, bravyi_kitaev_tree, jordan_wigner
from pyscf import ao2mo, cc, fci, gto, scf
from pyscf.lib import StreamObject

from .embed import rks_veff
from .exceptions import NbedConfigError, NbedDriverError
from .localizers import BOYSLocalizer, IBOLocalizer, PMLocalizer, SPADELocalizer
from .scf import huzinaga_RHF
from .utils import setup_logs

logger = logging.getLogger(__name__)


class NbedDriver(object):
    """Function to return the embedding Qubit Hamiltonian.

    Args:
        geometry (Path): A path to an .xyz file describing moleclar geometry.
        n_active_atoms (int): The number of atoms to include in the active region.
        basis (str): The name of an atomic orbital basis set to use for chemistry calculations.
        xc_functonal (str): The name of an Exchange-Correlation functional to be used for DFT.
        projector (str):
        localization (str): Orbital Localisation method to use. One of 'spade', 'mullikan', 'boys' or 'ibo'.
        convergence (float): The convergence tolerance for energy calculations.
        charge (int): Charge of molecular species
        mu_level_shift (float): Level shift parameter to use for mu-projector.
        run_ccsd_emb (bool): Whether or not to find the CCSD energy of embbeded system for reference.
        run_fci_emb (bool): Whether or not to find the FCI energy of embbeded system for reference.
        max_ram_memory (int): Amount of RAM memery in MB available for PySCF calculation
        pyscf_print_level (int): Amount of information PySCF prints
        qubits (int): The number of qubits available for the output hamiltonian.

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
        geometry: Path,
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
        max_ram_memory: Optional[int] = 4000,
        pyscf_print_level: int = 1,
        qubits: Optional[int] = None,
        savefile: Optional[Path] = None,
    ):

        self.geometry = geometry
        self.n_active_atoms = n_active_atoms
        self.basis = basis.lower()
        self.xc_functional = xc_functional.lower()
        self.output = output.lower()
        self.projector = projector.lower()
        self.localization = localization.lower()
        self.convergence = convergence
        self.charge = charge
        self.mu_level_shift = mu_level_shift
        self.run_ccsd_emb = run_ccsd_emb
        self.run_fci_emb = run_fci_emb
        self.max_ram_memory = max_ram_memory
        self.pyscf_print_level = pyscf_print_level
        self.qubits = qubits
        self.savefile = savefile

        config_valid = True
        if self.projector not in ["mu", "huzinaga", "both"]:
            logger.error(
                "Invalid projector %s selected. Choose from 'mu' or 'huzinzaga'.",
                self.projector,
            )
            config_valid = False

        if self.localization not in ["spade", "ibo", "boys", "mullikan"]:
            logger.error(
                "Invalid localization method %s. Choose from 'ibo','boys','mullikan' or 'spade'.",
                self.localization,
            )
            config_valid = False

        # Attributes
        self.e_act: float = None
        self.e_env: float = None
        self.two_e_cross: float = None

        self.molecular_ham: InteractionOperator = None
        self.classical_energy: float = None

        self.embed()

    def _build_mol(self) -> gto.mole:
        """Function to build PySCF molecule

        Returns:
            full_mol (gto.mol): built PySCF molecule object
        """
        logger.debug("Construcing molecule.")
        full_mol = gto.Mole(
            atom=self.geometry, basis=self.basis, charge=self.charge
        ).build()
        return full_mol

    @property
    def full_system_hamiltonian(self):
        """Build full molecular fermionic Hamiltonian (of whole system)
        Idea is to compare the number of terms to embedded Hamiltonian
        """
        return self.build_molecular_hamiltonian(self._global_HF)

    @cached_property
    def _global_hf(self) -> StreamObject:
        """Run full system Hartree-Fock."""
        mol_full = self._build_mol()
        # run Hartree-Fock
        global_HF = scf.RHF(mol_full)
        global_HF.conv_tol = self.convergence
        global_HF.max_memory = self.max_ram_memory
        global_HF.verbose = self.pyscf_print_level
        global_HF.kernel()

    @cached_property
    def _global_fci(self) -> StreamObject:
        """Function to run full molecule FCI calculation. Note this is very expensive"""

        # run FCI after HF
        global_fci = fci.FCI(self._global_HF)
        global_fci.conv_tol = self.convergence
        global_fci.verbose = self.pyscf_print_level
        global_fci.max_memory = self.max_ram_memory
        global_fci.run()
        print(f"global FCI: {global_fci.e_tot}")

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

        global_rks = self.define_rks_in_newbasis(
            global_rks, self._local_basis_transform
        )

        pyscf_scf_rks = global_rks  # TODO
        hcore_std = pyscf_scf_rks.get_hcore()
        pyscf_scf_rks.get_hcore = (
            lambda *args: self._local_basis_transform.conj().T
            @ hcore_std
            @ self._local_basis_transform
        )

        pyscf_scf_rks.get_veff = (
            lambda mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1: rks_veff(
                pyscf_scf_rks, self._local_basis_transform, dm=dm, check_result=True
            )
        )

        # overwrite C matrix with localised orbitals
        pyscf_scf_rks.mo_coeff = self.localized_system.c_loc_occ_and_virt
        dm_loc = pyscf_scf_rks.make_rdm1(
            mo_coeff=pyscf_scf_rks.mo_coeff, mo_occ=pyscf_scf_rks.mo_occ
        )

        # fock_locbasis = _global_rks.get_hcore() + _global_rks.get_veff(dm=dm_loc)
        fock_locbasis = pyscf_scf_rks.get_fock(dm=dm_loc)

        # orbital_energies_std = _global_rks.mo_energy
        orbital_energies_loc = np.diag(
            pyscf_scf_rks.mo_coeff.conj().T @ fock_locbasis @ pyscf_scf_rks.mo_coeff
        )
        pyscf_scf_rks.mo_energy = orbital_energies_loc

        # # check electronic energy matches standard global calc
        # global_rks_total_energy_loc = pyscf_scf_rks.energy_tot(dm=dm_loc)
        # if not np.isclose(self._global_rks.e_tot, global_rks_total_energy_loc):
        #     raise ValueError(
        #         "electronic energy of standard calculation not matching localized calculation"
        #     )

        # check if mo energies match
        # orbital_energies_std = _global_rks.mo_energy
        # if not np.allclose(orbital_energies_std, orbital_energies_loc):
        #     raise ValueError('orbital energies of standard calc not matching localized calc')

        return pyscf_scf_rks

    @cached_property
    def localized_system(self):
        """Run the localizer class."""
        logger.debug(f"Getting localized system using {self.localization}.")

        localizers = {"spade": SPADELocalizer, 
            "boys": BOYSLocalizer,
            "ibo": IBOLocalizer,
            "pipek-menzy": PMLocalizer
            }

        # Should already be validated.
        localized_system = localizers[self.localization](
            self._global_rks,
            self.n_active_atoms,
            occ_cutoff=0.95,
            virt_cutoff=0.95,
            run_virtual_localization=False,
            
        )
        return localized_system

    @cached_property
    def _local_basis_transform(
        self,
        sanity_check: Optional[bool] = False,
    ) -> np.ndarray:
        """Canonical to Localized Orbital Transform.

        Get operator that changes from standard canonical orbitals (C_matrix standard) to
        localized orbitals (C_matrix_localized)

        Args:
            pyscf_scf (StreamObject): PySCF molecule object
            c_all_localized_and_virt (np.array): C_matrix of localized orbitals (includes occupied and virtual)
            sanity_check (bool): optional flag to check if change of basis is working properly

        Returns:
            matrix_std_to_loc (np.array): Matrix that maps from standard (canonical) MOs to localized MOs
        """
        s_mat = self.pyscf_scf.get_ovlp()
        s_half = sp.linalg.fractional_matrix_power(s_mat, 0.5)
        s_neg_half = sp.linalg.fractional_matrix_power(s_mat, -0.5)

        # find orthogonal orbitals
        ortho_std = s_half @ self.pyscf_scf.mo_coeff
        ortho_loc = s_half @ self.localized_system.c_all_localized_and_virt

        # Build change of basis operator (maps between orthonormal basis (canonical and localized)
        unitary_ORTHO_std_onto_loc = np.einsum("ik,jk->ij", ortho_std, ortho_loc)

        # move back into non orthogonal basis
        matrix_std_to_loc = s_neg_half @ unitary_ORTHO_std_onto_loc @ s_half

        # if sanity_check:
        #     if np.allclose(unitary_ORTHO_std_onto_loc @ ortho_loc, ortho_std) is not True:
        #         raise ValueError(
        #             "Change of basis incorrect... U_ORTHO_std_onto_loc*C_ortho_loc !=  C_ortho_STD"
        #         )

        #     if (
        #         np.allclose(
        #             unitary_ORTHO_std_onto_loc.conj().T @ unitary_ORTHO_std_onto_loc,
        #             np.eye(unitary_ORTHO_std_onto_loc.shape[0]),
        #         )
        #         is not True
        #     ):
        #         raise ValueError("Change of basis (U_ORTHO_std_onto_loc) is not Unitary!")

        # if sanity_check:
        #     if (
        #         np.allclose(
        #             matrix_std_to_loc @ c_all_localized_and_virt, pyscf_scf.mo_coeff
        #         )
        #         is not True
        #     ):
        #         raise ValueError(
        #             "Change of basis incorrect... U_std*C_std !=  C_loc_occ_and_virt"
        #         )

        return matrix_std_to_loc

    def _init_local_rhf(self) -> scf.RHF:
        """Function to build embedded restricted Hartree Fock object for active subsystem

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

        logger.debug("Define Hartree-Fock object in localized basis")
        # TODO: need to check if change of basis here is necessary (START)
        h_core = local_rhf.get_hcore()

        local_rhf.get_hcore = (
            lambda *args: self._local_basis_transform.conj().T
            @ h_core
            @ self._local_basis_transform
        )

        if local_rhf.mo_coeff is not None:
            dm = local_rhf.make_rdm1(local_rhf.mo_coeff, local_rhf.mo_occ)
        else:
            dm = local_rhf.init_guess_by_1e()

        # if pyscf_RHF._eri is None:
        #     pyscf_RHF._eri = pyscf_RHF.mol.intor('int2e', aosym='s8')

        vj, vk = local_rhf.get_jk(mol=local_rhf.mol, dm=dm, hermi=1)
        v_eff = vj - vk * 0.5

        # v_eff = pyscf_obj.get_veff(dm=dm)
        local_rhf.get_veff = (
            lambda *args: self._local_basis_transform.conj().T
            @ v_eff
            @ self._local_basis_transform
        )

        return local_rhf

    def _subsystem_dft(self):
        """Function to perform subsystem RKS DFT calculation"""
        logger.debug("Calculating active and environment subsystem terms.")

        def _rks_components(
            self, dm_matrix: np.ndarray
        ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
            """
            Calculate the components of subsystem energy from a RKS DFT calculation.

            For a given density matrix this function returns the electronic energy, exchange correlation energy and
            J,K, V_xc matrices.

            Args:
                dm_matrix (np.ndarray): density matrix (to calculate all matrices from)
            Returns:
                Energy_elec (float): DFT energy defubed by input density matrix
                e_xc (float): exchange correlation energy defined by input density matrix
                J_mat (np.ndarray): J_matrix defined by input density matrix
            """
            # It seems that PySCF lumps J and K in the J array
            two_e_term = self._global_rks.get_veff(dm=dm_matrix)
            j_mat = two_e_term.vj
            k_mat = np.zeros_like(j_mat)

            e_xc = two_e_term.exc
            v_xc = two_e_term - j_mat

            energy_elec = (
                np.einsum("ij,ji->", self._global_rks.get_hcore(), dm_matrix)
                + two_e_term.ecoul
                + two_e_term.exc
            )

            # if check_E_with_pyscf:
            #     energy_elec_pyscf = self._global_rks.energy_elec(dm=dm_matrix)[0]
            #     if not np.isclose(energy_elec_pyscf, energy_elec):
            #         raise ValueError("Energy calculation incorrect")

            return energy_elec, e_xc, j_mat

        (self.e_act, e_xc_act, j_act) = _rks_components(
            self.localized_system.dm_active,
        )
        (self.e_env, e_xc_env, j_env) = _rks_components(
            self.localized_system.dm_enviro,
        )
        # Computing cross subsystem terms
        logger.debug("Calculating two electron cross subsystem energy.")

        two_e_term_total = self._global_rks.get_veff(dm=self.dm_active + self.dm_enviro)
        e_xc_total = two_e_term_total.exc

        j_cross = 0.5 * (
            np.einsum("ij,ij", self.dm_active, j_env)
            + np.einsum("ij,ij", self.dm_enviro, j_act)
        )
        # Because of projection
        k_cross = 0.0

        xc_cross = e_xc_total - e_xc_act - e_xc_env

        # overall two_electron cross energy
        two_e_cross = j_cross + k_cross + xc_cross

        # if not np.isclose(energy_DFT_components, self._global_rks.e_tot):
        #     energy_DFT_components = (
        #         self.e_act
        #         + self.e_env
        #         + two_e_cross
        #         + self._global_rks.energy_nuc()
        #     )

        #     raise ValueError(
        #         "DFT energy of localized components not matching supersystem DFT"
        #     )

        return None

    @cached_property
    def _orthogonal_projector(self):
        """Return a projector onto the environment in orthogonal basis."""
        # get system matrices
        s_mat = self._global_rks.get_ovlp()
        s_half = sp.linalg.fractional_matrix_power(s_mat, 0.5)

        # 1. Get orthogonal C matrix (localized)
        c_loc_ortho = s_half @ self.localized_system.c_loc_occ_and_virt

        # 2. Define projector that projects MO orbs of subsystem B onto themselves and system A onto zero state!
        #    (do this in orthongoal basis!)
        #    note we only take MO environment indices!
        ortho_proj = np.einsum(
            "ik,jk->ij",
            c_loc_ortho[:, self.localized_system.enviro_MO_inds],
            c_loc_ortho[:, self.localized_system.enviro_MO_inds],
        )
        return ortho_proj

    def _run_emb_CCSD(
        self, emb_pyscf_scf_rhf: scf.RHF, frozen_orb_list: Optional[list] = None
    ) -> Tuple[cc.CCSD, float]:
        """Function run CCSD on embedded restricted Hartree Fock object

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
        self, emb_pyscf_scf_rhf: gto.Mole, frozen_orb_list: Optional[list] = None
    ) -> Tuple[fci.FCI]:
        """Function run FCI on embedded restricted Hartree Fock object

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
        # Get Projector
        s_mat = self.pyscf_scf.get_ovlp()
        s_half = sp.linalg.fractional_matrix_power(s_mat, 0.5)
        enviro_projector = s_half @ self._orthogonal_projector @ s_half

        # run SCF
        v_emb = (self.mu_level_shift * enviro_projector) + self._dft_potential
        hcore_std = localized_rhf.get_hcore()
        localized_rhf.get_hcore = lambda *args: hcore_std + v_emb

        logger.debug("Running embedded RHF calculation.")
        localized_rhf.kernel()
        print(
            f"embedded HF energy MU_SHIFT: {localized_rhf.e_tot}, converged: {embedded_rhf.converged}"
        )

        dm_active_embedded = localized_rhf.make_rdm1(
            mo_coeff=localized_rhf.mo_coeff, mo_occ=localized_rhf.mo_occ
        )

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
        ) = huzinaga_RHF(
            localized_rhf,
            self._dft_potential,
            self._orthogonal_projector,
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

        return v_emb, localized_rhf

    def build_molecular_hamiltonian(
        self,
        scf_method: StreamObject,
    ) -> InteractionOperator:
        """Returns second quantized fermionic molecular Hamiltonian

        Args:
            scf_method (StreamObject): A pyscf self-consistent method.
            frozen_indices (list[int]): A list of integer indices of frozen moleclar orbitals.

        Returns:
            molecular_hamiltonian (InteractionOperator): fermionic molecular Hamiltonian
        """

        # C_matrix containing orbitals to be considered
        # if there are any environment orbs that have been projected out... these should NOT be present in the
        # scf_method.mo_coeff array (aka columns should be deleted!)
        c_matrix_active = scf_method.mo_coeff
        n_orbs = c_matrix_active.shape[1]

        # one body terms
        one_body_integrals = (
            c_matrix_active.T @ scf_method.get_hcore() @ c_matrix_active
        )

        two_body_compressed = ao2mo.kernel(scf_method.mol, c_matrix_active)

        # get electron repulsion integrals
        eri = ao2mo.restore(1, two_body_compressed, n_orbs)  # no permutation symmetry

        # Openfermion uses pysicist notation whereas pyscf uses chemists
        two_body_integrals = np.asarray(eri.transpose(0, 2, 3, 1), order="C")

        core_constant, one_body_ints_reduced, two_body_ints_reduced = (
            0,
            one_body_integrals,
            two_body_integrals,
        )
        # core_constant, one_body_ints_reduced, two_body_ints_reduced = get_active_space_integrals(
        #                                                                                        one_body_integrals,
        #                                                                                        two_body_integrals,
        #                                                                                        occupied_indices=None,
        #                                                                                        active_indices=active_mo_inds
        #                                                                                         )

        logger.debug(f"core constant: {core_constant}")

        one_body_coefficients, two_body_coefficients = spinorb_from_spatial(
            one_body_ints_reduced, two_body_ints_reduced
        )

        molecular_hamiltonian = InteractionOperator(
            core_constant, one_body_coefficients, 0.5 * two_body_coefficients
        )

        return molecular_hamiltonian

    def embed(self):
        """Generate embedded Hamiltonian

        Note run_mu_shift (bool) and run_huzinaga (bool) flags define which method to use (can be both)
        This is done when object is initialized
        """
        e_nuc = self._global_rks.mol.energy_nuc()

        self._subsystem_dft(self._global_rks)

        logger.debug("Get global DFT potential to optimize embedded calc in.")
        g_act_and_env = self._global_rks.get_veff(
            dm=(self.localized_system.dm_active + self.localized_system.dm_enviro)
        )
        g_act = self._global_rks.get_veff(dm=self.localized_system.dm_active)
        self._dft_potential = g_act_and_env - g_act

        # Initialise here, cause we're going to overwrite properties.
        local_rhf = self._init_local_rhf()

        if self.run_mu_shift is True:
            v_emb, embedded_rhf = self._mu_embed(local_rhf)
        elif self.run_huzinaga is True:
            v_emb, embedded_rhf = self._huzinaga_embed(local_rhf)

        else:
            logger.error("Neither ")
            raise NbedDriverError("No embedding method selected.")

        # delete enviroment orbitals:
        shift = self._global_rks.mol.nao - len(self.localized_system.enviro_MO_inds)
        frozen_enviro_orb_inds = [
            mo_i for mo_i in range(shift, self._global_rks.mol.nao)
        ]
        active_MO_inds = [
            mo_i
            for mo_i in range(embedded_rhf.mo_coeff.shape[1])
            if mo_i not in frozen_enviro_orb_inds
        ]

        embedded_rhf.mo_coeff = embedded_rhf.mo_coeff[:, active_MO_inds]
        embedded_rhf.mo_energy = embedded_rhf.mo_energy[active_MO_inds]
        embedded_rhf.mo_occ = embedded_rhf.mo_occ[active_MO_inds]

        # calculate correction
        wf_correction = np.einsum("ij,ij", v_emb, self.localized_system.dm_active)

        # classical energy
        self.classical_energy = self.e_env + self.two_e_cross + e_nuc - wf_correction

        # Hamiltonian
        self.molecular_ham = self.molecular_hamiltonian(embedded_rhf)

        # Calculate ccsd or fci energy
        if self.run_ccsd_emb is True:
            ccsd_emb, e_ccsd_corr = self._run_emb_CCSD(
                embedded_rhf, frozen_orb_list=None
            )
            e_wf_emb = (
                ccsd_emb.e_hf
                + e_ccsd_corr
                + self.e_env
                + self.two_e_cross
                - wf_correction
            )
            print("CCSD Energy MU shift:\n\t%s", e_wf_emb)

        if self.run_fci_emb is True:
            fci_emb = self._run_emb_FCI(self.embedded_rhf, frozen_orb_list=None)
            e_wf_fci_emb = (
                (fci_emb.e_tot) + self.e_env + self.two_e_cross - wf_correction
            )
            print("FCI Energy MU shift:\n\t%s", e_wf_fci_emb)

        print(f"num e emb: {2 * len(self.localized_system.active_MO_inds)}")
        print(self.localized_system.active_MO_inds)
        print(self.localized_system.enviro_MO_inds)

        return embedded_rhf


import nbed

ham = nbed.nbed()
ham

qiskit_vqe(hamiltonian=nbed.nbed(geometry, active_atoms)["huzinaga"])
