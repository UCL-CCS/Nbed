"""
Psi4 subclasses of Embed
"""
import numpy as np
import psi4

from .embed import Embed


class Psi4Embed(Embed):
    """Class with embedding methods using Psi4."""

    def run_mean_field(self, v_emb=None):
        """
        Runs Psi4 (PySCF is coming soon).
        If 'level' is not provided, it runs the a calculation at the level
        given by the 'low_level' key in self.keywords.

        Parameters
        ----------
        v_emb : numpy.array or list of numpy.array (None)
            Embedding potential.
        """
        if v_emb is None:
            self.outfile = open(self.keywords["embedding_output"], "w")
            # Preparing molecule string with C1 symmetry
            add_c1 = self.keywords["geometry"].splitlines()
            add_c1.append("symmetry c1")
            self.keywords["geometry"] = "\n".join(add_c1)

            # Running psi4 for the env (low level)
            psi4.set_memory(str(self.keywords["memory"]) + " MB")
            psi4.core.set_num_threads(self.keywords["num_threads"])
            self._mol = psi4.geometry(self.keywords["geometry"])
            self._mol.set_molecular_charge(self.keywords["charge"])
            self._mol.set_multiplicity(self.keywords["multiplicity"])

            psi4.core.be_quiet()
            psi4.core.set_output_file(self.keywords["driver_output"], True)
            psi4.set_options(
                {
                    "save_jk": "true",
                    "basis": self.keywords["basis"],
                    "reference": self.keywords["low_level_reference"],
                    "ints_tolerance": self.keywords["ints_tolerance"],
                    "e_convergence": self.keywords["e_convergence"],
                    "d_convergence": self.keywords["d_convergence"],
                    "scf_type": self.keywords["eri"],
                    "print": self.keywords["print_level"],
                    "damping_percentage": self.keywords["low_level_damping_percentage"],
                    "soscf": self.keywords["low_level_soscf"],
                }
            )

            self.e, self._wfn = psi4.energy(
                self.keywords["low_level"], molecule=self._mol, return_wfn=True
            )
            self._n_basis_functions = self._wfn.basisset().nbf()
            if self.keywords["low_level"] != "HF":
                self.e_xc_total = psi4.core.VBase.quadrature_values(
                    self._wfn.V_potential()
                )["FUNCTIONAL"]
                if self.keywords["low_level_reference"] == "rhf":
                    self.v_xc_total = self._wfn.Va().clone().np
                else:
                    self.alpha_v_xc_total = self._wfn.Va().clone().np
                    self.beta_v_xc_total = self._wfn.Vb().clone().np
            else:
                if self.keywords["low_level_reference"] == "rhf":
                    # self.v_xc_total = np.zeros([self._n_basis_functions,
                    # self._n_basis_functions])
                    self.v_xc_total = 0.0
                else:
                    # self.alpha_v_xc_total = np.zeros([self._n_basis_functions,
                    # self._n_basis_functions])
                    # self.beta_v_xc_total = np.zeros([self._n_basis_functions,
                    # self._n_basis_functions])
                    self.alpha_v_xc_total = 0.0
                    self.beta_v_xc_total = 0.0
                self.e_xc_total = 0.0
        else:
            psi4.set_options(
                {
                    "docc": [self.n_act_mos],
                    "reference": self.keywords["high_level_reference"],
                }
            )
            if self.keywords["high_level_reference"] == "rhf":
                f = open("newH.dat", "w")
                for i in range(self.h_core.shape[0]):
                    for j in range(self.h_core.shape[1]):
                        f.write("%s\n" % (self.h_core + v_emb)[i, j])
                f.close()
            else:
                psi4.set_options({"socc": [self.n_act_mos - self.beta_n_act_mos]})
                fa = open("Va_emb.dat", "w")
                fb = open("Vb_emb.dat", "w")
                for i in range(self.h_core.shape[0]):
                    for j in range(self.h_core.shape[1]):
                        fa.write("%s\n" % v_emb[0][i, j])
                        fb.write("%s\n" % v_emb[1][i, j])
                fa.close()
                fb.close()

            if (
                self.keywords["high_level"][:2] == "cc"
                and self.keywords["cc_type"] == "df"
            ):
                psi4.set_options(
                    {"cc_type": self.keywords["cc_type"], "df_ints_io": "save"}
                )
            self.e, self._wfn = psi4.energy("hf", molecule=self._mol, return_wfn=True)

        if self.keywords["low_level_reference"] == "rhf":
            self.occupied_orbitals = self._wfn.Ca_subset("AO", "OCC").np
            self.j = self._wfn.jk().J()[0].np
            self.k = self._wfn.jk().K()[0].np
        else:
            self.alpha_occupied_orbitals = self._wfn.Ca_subset("AO", "OCC").np
            self.beta_occupied_orbitals = self._wfn.Ca_subset("AO", "OCC").np
            self.alpha_j = self._wfn.jk().J()[0].np
            self.beta_j = self._wfn.jk().J()[1].np
            self.alpha_k = self._wfn.jk().K()[0].np
            self.beta_k = self._wfn.jk().K()[1].np

        self.nre = self._mol.nuclear_repulsion_energy()
        self.ao_overlap = self._wfn.S().np
        self.h_core = self._wfn.H().np
        self.alpha = self._wfn.functional().x_alpha()
        return None

    def count_active_aos(self, basis=None):
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
            basis = self._wfn.basisset()
            n_basis_functions = basis.nbf()
        else:
            projected_wfn = psi4.core.Wavefunction.build(self._mol, basis)
            basis = projected_wfn.basisset()
            n_basis_functions = basis.nbf()

        self.n_active_aos = 0
        active_atoms = list(range(self.keywords["n_active_atoms"]))
        for ao in range(n_basis_functions):
            for atom in active_atoms:
                if basis.function_to_center(ao) == atom:
                    self.n_active_aos += 1
        return self.n_active_aos

    def basis_projection(self, orbitals, projection_basis):
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
        projected_wfn = psi4.core.Wavefunction.build(self._mol, projection_basis)
        mints = psi4.core.MintsHelper(projected_wfn.basisset())
        self.projected_overlap = mints.ao_overlap().np[
            : self.n_active_aos, : self.n_active_aos
        ]
        self.overlap_two_basis = mints.ao_overlap(
            projected_wfn.basisset(), self._wfn.basisset()
        ).np[: self.n_active_aos, :]
        projected_orbitals = (
            np.linalg.inv(self.projected_overlap) @ self.overlap_two_basis @ orbitals
        )
        return projected_orbitals

    def closed_shell_subsystem(self, orbitals):
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

        density = orbitals @ orbitals.T
        psi4_orbitals = psi4.core.Matrix.from_array(orbitals)

        if hasattr(self._wfn, "get_basisset"):
            jk = psi4.core.JK.build(
                self._wfn.basisset(), self._wfn.get_basisset("DF_BASIS_SCF"), "DF"
            )
        else:
            jk = psi4.core.JK.build(self._wfn.basisset())
        jk.set_memory(int(1.25e9))
        jk.initialize()
        jk.C_left_add(psi4_orbitals)
        jk.compute()
        jk.C_clear()
        jk.finalize()

        j = jk.J()[0].np
        k = jk.K()[0].np

        if self._wfn.functional().name() != "HF":
            self._wfn.Da().copy(psi4.core.Matrix.from_array(density))
            self._wfn.form_V()
            v_xc = self._wfn.Va().clone().np
            e_xc = psi4.core.VBase.quadrature_values(self._wfn.V_potential())[
                "FUNCTIONAL"
            ]

        else:
            basis = self._wfn.basisset()
            n_basis_functions = basis.nbf()
            v_xc = 0.0
            e_xc = 0.0

        # Energy
        e = self.dot(density, 2.0 * (self.h_core + j) - self.alpha * k) + e_xc
        return e, e_xc, 2.0 * j, k, v_xc

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
        mo_fock = orbitals.T @ self._wfn.Fa().np @ orbitals
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
            jk = psi4.core.JK.build(
                self._wfn.basisset(), self._wfn.get_basisset("DF_BASIS_SCF"), "DF"
            )
            jk.set_memory(int(1.25e9))
            jk.initialize()
            jk.print_header()
            jk.C_left_add(self._wfn.Ca())
            jk.compute()
            jk.C_clear()
            jk.finalize()
            self.operator = jk.K()[0].np
            if self.keywords["operator"] == "K_orb":
                self.operator = 0.06 * self._wfn.Fa().np - self.K
        elif self.keywords["operator"] == "V":
            mints = psi4.core.MintsHelper(self._wfn.basisset())
            self.operator = mints.ao_potential().np
        elif self.keywords["operator"] == "T":
            mints = psi4.core.MintsHelper(self._wfn.basisset())
            self.operator = mints.ao_kinetic().np
        elif self.keywords["operator"] == "H":
            self.operator = self._wfn.H().np
        elif self.keywords["operator"] == "S":
            self.operator = self._wfn.S().np
        elif self.keywords["operator"] == "F":
            self.operator = self._wfn.Fa().np

    def open_shell_subsystem(self, alpha_orbitals, beta_orbitals):
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
        jk = psi4.core.JK.build(
            self._wfn.basisset(), self._wfn.get_basisset("DF_BASIS_SCF"), "DF"
        )
        jk.set_memory(int(1.25e9))
        jk.initialize()
        jk.C_left_add(psi4.core.Matrix.from_array(alpha_orbitals))
        jk.C_left_add(psi4.core.Matrix.from_array(beta_orbitals))
        jk.compute()
        jk.C_clear()
        jk.finalize()
        alpha_j = jk.J()[0].np
        beta_j = jk.J()[1].np
        alpha_k = jk.K()[0].np
        beta_k = jk.K()[1].np

        if self._wfn.functional().name() != "HF":
            self._wfn.Da().copy(psi4.core.Matrix.from_array(alpha_density))
            self._wfn.Db().copy(psi4.core.Matrix.from_array(beta_density))
            self._wfn.form_V()
            alpha_v_xc = self._wfn.Va().clone().np
            beta_v_xc = self._wfn.Vb().clone().np
            e_xc = psi4.core.VBase.quadrature_values(self._wfn.V_potential())[
                "FUNCTIONAL"
            ]
        else:
            # alpha_v_xc = np.zeros([self._n_basis_functions,
            # self._n_basis_functions])
            # beta_v_xc = np.zeros([self._n_basis_functions,
            # self._n_basis_functions])
            alpha_v_xc = 0.0
            beta_v_xc = 0.0
            e_xc = 0.0

        e = (
            self.dot(self.h_core, alpha_density + beta_density)
            + 0.5
            * (
                self.dot(alpha_j + beta_j, alpha_density + beta_density)
                - self.alpha * self.dot(alpha_k, alpha_density)
                - self.alpha * self.dot(beta_k, beta_density)
            )
            + e_xc
        )

        return e, e_xc, alpha_j, beta_j, alpha_k, beta_k, alpha_v_xc, beta_v_xc

    def orthonormalize(self, S, C, n_non_zero):
        """
        (Deprecated) Orthonormalizes a set of orbitals (vectors).

        Parameters
        ----------
        S : numpy.array
            Overlap matrix in AO basis.
        C : numpy.array
            MO coefficient matrix, vectors to be orthonormalized.
        n_non_zero : int
            Number of orbitals that have non-zero norm.

        Returns
        -------
        C_orthonormal : numpy.array
            Set of n_non_zero orthonormal orbitals.
        """

        overlap = C.T @ S @ C
        v, w = np.linalg.eigh(overlap)
        idx = v.argsort()[::-1]
        v = v[idx]
        w = w[:, idx]
        C_orthonormal = C @ w
        for i in range(n_non_zero):
            C_orthonormal[:, i] = C_orthonormal[:, i] / np.sqrt(v[i])
        return C_orthonormal[:, :n_non_zero]

    def molden(self, shell_orbitals, shell):
        """
        Creates molden file from orbitals at the shell.

        Parameters
        ----------
        span_orbitals : numpy.array
            Span orbitals.
        shell : int
            Shell index.
        """
        self._wfn.Ca().copy(psi4.core.Matrix.from_array(shell_orbitals))
        psi4.driver.molden(self._wfn, str(shell) + ".molden")
        return None

    def heatmap(self, span_orbitals, kernel_orbitals, shell):
        """
        Creates heatmap file from orbitals at the i-th shell.

        Parameters
        ----------
        span_orbitals : numpy.array
            Span orbitals.
        kernel_orbitals : numpy.array
            Kernel orbitals.
        shell : int
            Shell index.
        """
        orbitals = np.hstack((span_orbitals, kernel_orbitals))
        mo_operator = orbitals.T @ self.operator @ orbitals
        np.savetxt("heatmap_" + str(shell) + ".dat", mo_operator)
        return None

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
            nfrz = self.n_env_mos
        else:
            effective_orbitals = np.hstack((span_orbitals, kernel_orbitals))
            orbital_energies = np.concatenate(
                (span_orbital_energies, kernel_orbital_energies)
            )
            nfrz = self._n_basis_functions - self.n_act_mos - span_orbitals.shape[1]
            orbitals = np.hstack(
                (
                    self.occupied_orbitals,
                    effective_orbitals,
                    self._wfn.Ca().np[:, shift:],
                )
            )
            orbital_energies = np.concatenate(
                (
                    self._wfn.epsilon_a().np[: self.n_act_mos],
                    orbital_energies,
                    self._wfn.epsilon_a().np[shift:],
                )
            )
            self._wfn.Ca().copy(psi4.core.Matrix.from_array(orbitals))
            self._wfn.epsilon_a().np[:] = orbital_energies[:]

        # Update the number of frozen orbitals and compute energy
        frzvpi = psi4.core.Dimension.from_list([nfrz])
        self._wfn.new_frzvpi(frzvpi)
        # wf_eng, wf_wfn = psi4.energy(self.keywords['high_level'],
        # ref_wfn = self._wfn, return_wfn = True)
        psi4.energy(self.keywords["high_level"], ref_wfn=self._wfn)
        correlation_energy = psi4.core.get_variable(
            self.keywords["high_level"].upper() + " CORRELATION ENERGY"
        )
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
        effective_orbitals = self._wfn.Ca().np[:, self.n_act_mos : shift]
        return effective_orbitals
