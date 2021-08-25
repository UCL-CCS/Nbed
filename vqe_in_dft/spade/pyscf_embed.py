"""
PySCF subclasses of Embed
"""
from typing import Tuple
import numpy as np
from pyscf import gto, dft, scf, lib, mp, cc

from .embed import Embed


class PySCFEmbed(Embed):
    """Class with embedding methods using PySCF."""

    def run_mean_field(self, v_emb=None):
        """
        Runs mean-field calculation with PySCF.
        If 'level' is not provided, it runs the a calculation at the level
        given by the 'low_level' key in self.keywords. HF otherwise.

        Parameters
        ----------
        v_emb : numpy.array or list of numpy.array (None)
            Embedding potential.
        """
        # Create molecule from keywords
        self._mol = gto.mole.Mole()
        self._mol.verbose = self.keywords["print_level"]
        # self._mol.output = self.keywords['driver_output']
        self._mol.atom = self.keywords["geometry"]
        self._mol.max_memory = self.keywords["memory"]
        self._mol.basis = self.keywords["basis"]
        self._mol.build()

        reference_methods = {
            "hf": {
                "rhf": scf.RHF,
                "uhf": scf.UHF,
                "rohf": scf.ROHF,
            },
            "_": {"rhf": dft.RKS, "uhf": dft.UKS, "rohf": dft.ROKS},
        }
        # Alias the keywords for neatness
        llr = self.keywords["low_level_reference"].lower()
        hlr = self.keywords["high_level_reference"].lower()

        if v_emb is None:  # low-level/environment calculation
            self._mol.output = self.keywords["driver_output"]
            ref = reference_methods.get(
                self.keywords["low_level"], reference_methods["_"]
            )
            ref = ref[llr]
            self._mean_field = ref(self._mol)

            if self.keywords["low_level"] == "hf":
                self.e_xc = 0.0

            self._mean_field.conv_tol = self.keywords["e_convergence"]
            self._mean_field.xc = self.keywords["low_level"]
            self._mean_field.kernel()

            # It seems like these two aren't used
            self.v_xc_total = self._mean_field.get_veff()
            self.e_xc_total = self._mean_field.get_veff().exc

        # If an embedding potential is provided, run the high-level calculation
        else:
            ref = reference_methods["hf"][hlr]
            self._mean_field = ref(self._mol)
            if llr == "rhf":
                self._mol.nelectron = 2 * self.n_act_mos
                self._mean_field.get_hcore = lambda *args: v_emb + self.h_core
            elif llr in ["rohf", "uhf"]:
                self._mol.nelectron = self.n_act_mos + self.beta_n_act_mos
                self._mean_field.get_vemb = lambda *args: v_emb
            self._mean_field.conv_tol = self.keywords["e_convergence"]
            self._mean_field.kernel()

        if llr == "rhf":
            docc = (self._mean_field.mo_occ == 2).sum()
            self.occupied_orbitals = self._mean_field.mo_coeff[:, :docc]
            self.j, self.k = self._mean_field.get_jk()
            self.v_xc_total = self._mean_field.get_veff() - self.j
        else:
            if (llr == "uhf" and v_emb is None) or (hlr == "uhf" and v_emb is not None):
                n_alpha = (self._mean_field.mo_occ[0] == 1).sum()
                n_beta = (self._mean_field.mo_occ[1] == 1).sum()
                self.alpha_occupied_orbitals = self._mean_field.mo_coeff[0, :, :n_alpha]
                self.beta_occupied_orbitals = self._mean_field.mo_coeff[1, :, :n_beta]
            if (llr == "rohf" and v_emb is None) or (
                hlr == "rohf" and v_emb is not None
            ):
                n_beta = (self._mean_field.mo_occ == 2).sum()
                n_alpha = n_beta + (self._mean_field.mo_occ == 1).sum()
                self.alpha_occupied_orbitals = self._mean_field.mo_coeff[:, :n_alpha]
                self.beta_occupied_orbitals = self._mean_field.mo_coeff[:, :n_beta]

            j, k = self._mean_field.get_jk()
            self.alpha_j, self.beta_j = j[0:2]
            self.alpha_k, self.beta_k = k[0:2]
            self.alpha_v_xc_total = (
                self._mean_field.get_veff()[0] - self.alpha_j - self.beta_j
            )
            self.beta_v_xc_total = (
                self._mean_field.get_veff()[1] - self.alpha_j - self.beta_j
            )

        self.alpha = 0.0
        self._n_basis_functions = self._mol.nao
        self.nre = self._mol.energy_nuc()
        self.ao_overlap = self._mean_field.get_ovlp(self._mol)
        self.h_core = self._mean_field.get_hcore(self._mol)
        print(f"{self._mean_field.e_tot=}")
        return None

    def count_active_aos(self, basis: str = None) -> int:
        """
        Computes the number of AOs from active atoms.

        Parameters
        ----------
        basis : str
            Name of basis set from which to count active AOs.

        Returns
        -------
            self.n_active_aos : int
                Number of AOs in the active atoms.
        """
        if basis is None:
            self.n_active_aos = self._mol.aoslice_nr_by_atom()[
                self.keywords["n_active_atoms"] - 1
            ][3]
        else:
            self._projected_mol = gto.mole.Mole()
            self._projected_mol.atom = self.keywords["geometry"]
            self._projected_mol.basis = basis
            self._projected_mf = scf.RHF(self._projected_mol)
            self.n_active_aos = self._projected_mol.aoslice_nr_by_atom()[
                self.keywords["n_active_atoms"] - 1
            ][3]
        return self.n_active_aos

    def basis_projection(self, orbitals: np.ndarray) -> np.ndarray:
        """
        Defines a projection of orbitals in one basis onto another.

        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients to be projected.
        projection_basis : str
            Name of basis set onto which orbitals are to be projected.

        Returns
        -------
        projected_orbitals : numpy.array
            MO coefficients of orbitals projected onto projection_basis.
        """
        self.projected_overlap = self._projected_mf.get_ovlp(self._mol)[
            : self.n_active_aos, : self.n_active_aos
        ]
        self.overlap_two_basis = gto.intor_cross(
            "int1e_ovlp_sph", self._mol, self._projected_mol
        )[: self.n_active_aos, :]
        projected_orbitals = (
            np.linalg.inv(self.projected_overlap) @ self.overlap_two_basis @ orbitals
        )
        return projected_orbitals

    def closed_shell_subsystem(
        self, subsystem_orbitals: np.ndarray
    ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the potential matrices J, K, and V and subsystem energies.

        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients of subsystem.

        Returns
        -------
        e : float
            Total energy of subsystem.
        e_xc : float
            DFT Exchange-correlation energy of subsystem.
        j : numpy.array
            Coulomb matrix of subsystem.
        k : numpy.array
            Exchange matrix of subsystem.
        v_xc : numpy.array
            Kohn-Sham potential matrix of subsystem.
        """
        density = 2.0 * subsystem_orbitals @ subsystem_orbitals.T
        # It seems that PySCF lumps J and K in the J array
        j = self._mean_field.get_j(dm=density)
        k = np.zeros([self._n_basis_functions, self._n_basis_functions])
        two_e_term = self._mean_field.get_veff(self._mol, density)
        e_xc = two_e_term.exc
        v_xc = two_e_term - j

        # Energy
        e = self.trace(density, self.h_core + j / 2) + e_xc
        return e, e_xc, j, k, v_xc

    def open_shell_subsystem(
        self, alpha_orbitals: np.ndarray, beta_orbitals: np.ndarray
    ) -> Tuple[
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Computes the potential matrices J, K, and V and subsystem
        energies for open shell cases.

        Parameters
        ----------
        alpha_orbitals : numpy.array
            Alpha MO coefficients.
        beta_orbitals : numpy.array
            Beta MO coefficients.

        Returns
        -------
        e : float
            Total energy of subsystem.
        e_xc : float
            Exchange-correlation energy of subsystem.
        alpha_j : numpy.array
            Alpha Coulomb matrix of subsystem.
        beta_j : numpy.array
            Beta Coulomb matrix of subsystem.
        alpha_k : numpy.array
            Alpha Exchange matrix of subsystem.
        beta_k : numpy.array
            Beta Exchange matrix of subsystem.
        alpha_v_xc : numpy.array
            Alpha Kohn-Sham potential matrix of subsystem.
        beta_v_xc : numpy.array
            Beta Kohn-Sham potential matrix of subsystem.
        """
        alpha_density = alpha_orbitals @ alpha_orbitals.T
        beta_density = beta_orbitals @ beta_orbitals.T

        # J and K
        j = self._mean_field.get_j(dm=[alpha_density, beta_density])
        alpha_j = j[0]
        beta_j = j[1]
        alpha_k = np.zeros([self._n_basis_functions, self._n_basis_functions])
        beta_k = np.zeros([self._n_basis_functions, self._n_basis_functions])
        two_e_term = self._mean_field.get_veff(self._mol, [alpha_density, beta_density])
        e_xc = two_e_term.exc
        alpha_v_xc = two_e_term[0] - (j[0] + j[1])
        beta_v_xc = two_e_term[1] - (j[0] + j[1])

        # Energy
        e = (
            self.dot(self.h_core, alpha_density + beta_density)
            + 0.5 * (self.dot(alpha_j + beta_j, alpha_density + beta_density))
            + e_xc
        )

        return e, e_xc, alpha_j, beta_j, alpha_k, beta_k, alpha_v_xc, beta_v_xc

    def correlation_energy(
        self,
        span_orbitals=None,
        kernel_orbitals=None,
        span_orbital_energies=None,
        kernel_orbital_energies=None,
    ):
        """
        Computes the correlation energy for the current set of active
        virtual orbitals.

        Parameters
        ----------
        span_orbitals : numpy.array
            Orbitals transformed by the span of the previous shell.
        kernel_orbitals : numpy.array
            Orbitals transformed by the kernel of the previous shell.
        span_orbital_energies : numpy.array
            Orbitals energies of the span orbitals.
        kernel_orbital_energies : numpy.array
            Orbitals energies of the kernel orbitals.

        Returns
        -------
        correlation_energy : float
            Correlation energy of the span_orbitals.
        """

        shift = self._n_basis_functions - self.n_env_mos
        if span_orbitals is None:
            # If not using CL orbitals, just freeze the level-shifted MOs
            frozen_orbitals = [i for i in range(shift, self._n_basis_functions)]
        else:
            # Preparing orbitals and energies for CL shell
            effective_orbitals = np.hstack((span_orbitals, kernel_orbitals))
            orbital_energies = np.concatenate(
                (span_orbital_energies, kernel_orbital_energies)
            )
            frozen_orbitals = [
                i
                for i in range(
                    self.n_act_mos + span_orbitals.shape[1], self._n_basis_functions
                )
            ]
            orbitals = np.hstack(
                (
                    self.occupied_orbitals,
                    effective_orbitals,
                    self._mean_field.mo_coeff[:, shift:],
                )
            )
            orbital_energies = np.concatenate(
                (
                    self._mean_field.mo_energy[: self.n_act_mos],
                    orbital_energies,
                    self._mean_field.mo_energy[shift:],
                )
            )
            # Replace orbitals in the mean_field obj by the CL orbitals
            # and compute correlation energy
            self._mean_field.mo_energy = orbital_energies
            self._mean_field.mo_coeff = orbitals

        if self.keywords["high_level"].lower() == "mp2":
            # embedded_wf = mp.MP2(self._mean_field).run()
            embedded_wf = mp.MP2(self._mean_field).set(frozen=frozen_orbitals).run()
            correlation_energy = embedded_wf.e_corr
        if (
            self.keywords["high_level"].lower() == "ccsd"
            or self.keywords["high_level"].lower() == "ccsd(t)"
        ):
            embedded_wf = cc.CCSD(self._mean_field).set(frozen=frozen_orbitals).run()
            correlation_energy = embedded_wf.e_corr
            if self.keywords["high_level"].lower() == "ccsd(t)":
                t_correction = embedded_wf.ccsd_t().T
                correlation_energy += t_correction
        # if span_orbitals provided, store correlation energy of shells
        if span_orbitals is not None:
            self.correlation_energy_shell.append(correlation_energy)
        return correlation_energy

    def effective_virtuals(self):
        """
        Slices the effective virtuals from the entire virtual space.

        Returns
        -------
        effective_orbitals : numpy.array
            Virtual orbitals without the level-shifted orbitals
            from the environment.
        """
        shift = self._n_basis_functions - self.n_env_mos
        effective_orbitals = self._mean_field.mo_coeff[:, self.n_act_mos : shift]
        return effective_orbitals

    def pseudocanonical(self, orbitals):
        """
        Returns pseudocanonical orbitals and the corresponding
        orbital energies.

        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients of orbitals to be pseudocanonicalized.

        Returns
        -------
        e_orbital_pseudo : numpy.array
            diagonal elements of the Fock matrix in the
            pseudocanonical basis.
        pseudo_orbitals : numpy.array
            pseudocanonical orbitals.
        """
        fock_matrix = self._mean_field.get_fock()
        mo_fock = orbitals.T @ fock_matrix @ orbitals
        e_orbital_pseudo, pseudo_transformation = np.linalg.eigh(mo_fock)
        pseudo_orbitals = orbitals @ pseudo_transformation
        return e_orbital_pseudo, pseudo_orbitals

    def ao_operator(self):
        """
        Returns the matrix representation of the operator chosen to
        construct the shells.

        Returns
        -------

        K : numpy.array
            Exchange.
        V : numpy.array
            Electron-nuclei potential.
        T : numpy.array
            Kinetic energy.
        H : numpy.array
            Core (one-particle) Hamiltonian.
        S : numpy.array
            Overlap matrix.
        F : numpy.array
            Fock matrix.
        K_orb : numpy.array
            K orbitals (see Feller and Davidson, JCP, 74, 3977 (1981)).
        """
        if self.keywords["operator"] == "K" or self.keywords["operator"] == "K_orb":
            self.operator = self._mean_field.get_k()
            if self.keywords["operator"] == "K_orb":
                self.operator = 0.06 * self._mean_field.get_fock() - self.operator
        elif self.keywords["operator"] == "V":
            self.operator = self._mol.intor_symmetric("int1e_nuc")
        elif self.keywords["operator"] == "T":
            self.operator = self._mol.intor_symmetric("int1e_kin")
        elif self.keywords["operator"] == "H":
            self.operator = self._mean_field.get_hcore()
        elif self.keywords["operator"] == "S":
            self.operator = self._mean_field.get_ovlp()
        elif self.keywords["operator"] == "F":
            self.operator = self._mean_field.get_fock()
        return None
