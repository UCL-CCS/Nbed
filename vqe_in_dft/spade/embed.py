"""
Embed base class
"""
import numpy as np
import scipy.linalg

class Embed():
    """ Class with package-independent embedding methods."""

    def __init__(self, keywords):
        """
        Initialize the Embed class.

        Parameters
        ----------
        keywords (dict): dictionary with embedding options.
        """
        self.keywords = keywords
        self.correlation_energy_shell = []
        self.shell_size = 0
        self.outfile = open(keywords['embedding_output'], 'w')
        return None

    @staticmethod
    def dot(A, B):
        """
        (Deprecated) Computes the trace (dot or Hadamard product) 
        of matrices A and B.
        This has now been replaced by a lambda function in 
        embedding_module.py.

        Parameters
        ----------
        A : numpy.array
        B : numpy.array

        Returns
        -------
        The trace (dot product) of A * B

        """
        return np.einsum('ij, ij', A, B)

    def orbital_rotation(self, orbitals, n_active_aos, ao_overlap = None):
        """
        SVD orbitals projected onto active AOs to rotate orbitals.

        If ao_overlap is not provided, C is assumed to be in an
        orthogonal basis.
        
        Parameters
        ----------
        orbitals : numpy.array
            MO coefficient matrix.
        n_active_aos : int
            Number of atomic orbitals in the active atoms.
        ao_overlap : numpy.array (None)
            AO overlap matrix.

        Returns
        -------
        rotation_matrix : numpy.array
            Matrix to rotate orbitals.
        singular_values : numpy.array
            Singular values.
        """
        if ao_overlap is None:
            orthogonal_orbitals = orbitals[:n_active_aos, :]
        else:
            s_half = scipy.linalg.fractional_matrix_power(ao_overlap, 0.5)
            orthogonal_orbitals = (s_half @ orbitals)[:n_active_aos, :]

        u, s, v = np.linalg.svd(orthogonal_orbitals, full_matrices=True)
        rotation_matrix = v
        singular_values = s
        return rotation_matrix, singular_values

    def orbital_partition(self, sigma, beta_sigma = None):
        """
        Partition the orbital space by SPADE or all AOs in the
        projection basis. Beta variables are only used for open shells.

        Parameters
        ----------
        sigma : numpy.array
            Singular values.
        beta_sigma : numpy.array (None)
            Beta singular values.

        Returns
        -------
        self.n_act_mos : int
            (alpha) number of active MOs.
        self.n_env_mos : int
            (alpha) number of environment MOs.
        self.beta_n_act_mos : int
            Beta number of active MOs.
        self.beta_n_env_mos : int
            Beta number of environment MOs.
        """
        if self.keywords['partition_method'] == 'spade':
            delta_s = [-(sigma[i+1] - sigma[i]) for i in range(len(sigma) - 1)]
            self.n_act_mos = np.argpartition(delta_s, -1)[-1] + 1
            self.n_env_mos = len(sigma) - self.n_act_mos
        else:
            assert isinstance(self.keywords['occupied_projection_basis'], str),\
                '\n Define a projection basis'
            self.n_act_mos = self.n_active_aos
            self.n_env_mos = len(sigma) - self.n_act_mos

        if self.keywords['low_level_reference'] == 'rhf':
            return self.n_act_mos, self.n_env_mos
        else:
            assert beta_sigma is not None, 'Provide beta singular values'
            if self.keywords['partition_method'] == 'spade':
                beta_delta_s = [-(beta_sigma[i+1] - beta_sigma[i]) \
                    for i in range(len(beta_sigma) - 1)]
                self.beta_n_act_mos = np.argpartition(beta_delta_s, -1)[-1] + 1
                self.beta_n_env_mos = len(beta_sigma) - self.beta_n_act_mos
            else:
                assert isinstance(self.keywords['occupied_projection_basis'], str),\
                    '\n Define a projection basis'
                self.beta_n_act_mos = self.beta_n_active_aos
                self.beta_n_env_mos = len(beta_sigma) - self.beta_n_act_mos
            return (self.n_act_mos, self.n_env_mos, self.beta_n_act_mos,
                    self.beta_n_env_mos)

    def header(self):
        """Prints the header in the output file."""
        self.outfile.write('\n')
        self.outfile.write(' ' + 75*'-' + '\n')
        self.outfile.write('                               PsiEmbed\n\n')
        self.outfile.write('                        Python Stack for Improved\n')
        self.outfile.write('       and Efficient Methods and Benchmarking in'
                            + ' Embedding Development\n\n')
        self.outfile.write('                            Daniel Claudino\n')
        self.outfile.write('                            September  2019\n')
        self.outfile.write(' ' + 75*'-' + '\n')
        self.outfile.write('\n')
        self.outfile.write(' Main references: \n\n')
        self.outfile.write('     Projection-based embedding:\n')
        self.outfile.write('     F.R. Manby, M. Stella, J.D. Goodpaster,'
                            + ' T.F. Miller. III,\n')
        self.outfile.write('     J. Chem. Theory Comput. 2012, 8, 2564.\n\n')

        if self.keywords['partition_method'] == 'spade':
            self.outfile.write('     SPADE partition:\n')
            self.outfile.write('     D. Claudino, N.J. Mayhall,\n')
            self.outfile.write('     J. Chem. Theory Comput. 2019, 15, 1053.\n\n')

        if 'n_cl_shell' in self.keywords.keys():
            self.outfile.write('     Concentric localization (CL):\n')
            self.outfile.write('     D. Claudino, N.J. Mayhall,\n')
            self.outfile.write('     J. Chem. Theory Comput. 2019, 15, 6085.\n\n')

        if self.keywords['package'].lower() == 'psi4':
            self.outfile.write('     Psi4:\n')
            self.outfile.write('     R. M. Parrish, L. A. Burns, D. G. A. Smith'
                + ', A. C. Simmonett, \n')
            self.outfile.write('     A. E. DePrince III, E. G. Hohenstein'
                + ', U. Bozkaya, A. Yu. Sokolov,\n')
            self.outfile.write('     R. Di Remigio, R. M. Richard, J. F. Gonthier'
                + ', A. M. James,\n') 
            self.outfile.write('     H. R. McAlexander, A. Kumar, M. Saitow'
                + ', X. Wang, B. P. Pritchard,\n')
            self.outfile.write('     P. Verma, H. F. Schaefer III'
                + ', K. Patkowski, R. A. King, E. F. Valeev,\n')
            self.outfile.write('     F. A. Evangelista, J. M. Turney,'
                + 'T. D. Crawford, and C. D. Sherrill,\n')
            self.outfile.write('     J. Chem. Theory Comput. 2017, 13, 3185.')

        if self.keywords['package'].lower() == 'pyscf':
            self.outfile.write('     PySCF:\n')
            self.outfile.write('     Q. Sun, T. C. Berkelbach, N. S. Blunt'
                + ', G. H. Booth, S. Guo, Z. Li,\n')
            self.outfile.write('     J. Liu, J. D. McClain, E. R. Sayfutyarova'
                + ', S. Sharma, S. Wouters,\n')
            self.outfile.write('     and G. K.‐L. Chan,\n')
            self.outfile.write('     WIREs Comput. Mol. Sci. 2018, 8, e1340.')
        self.outfile.write('\n\n')
        self.outfile.write(' ' + 75*'-' + '\n')
        return None

    def print_scf(self, e_act, e_env, two_e_cross, e_act_emb, correction):
        """
        Prints mean-field info from before and after embedding.

        Parameters
        ----------
        e_act : float
            Energy of the active subsystem.
        e_env : float
            Energy of the environment subsystem.
        two_e_cross : float
            Intersystem interaction energy.
        e_act_emb : float
            Energy of the embedded active subsystem.
        correction : float
            Correction from the embedded density.
        """
        self.outfile.write('\n\n Energy values in atomic units\n')
        self.outfile.write(' Embedded calculation: '
            + self.keywords['high_level'].upper()
            + '-in-' + self.keywords['low_level'].upper() + '\n\n')
        if self.keywords['partition_method'] == 'spade':
            if 'occupied_projection_basis' not in self.keywords:
                self.outfile.write(' Orbital partition method: SPADE\n')
            else:
                self.outfile.write((' Orbital partition method: SPADE with ',
                    'occupied space projected onto '
                    + self.keywords['occupied_projection_basis'].upper() + '\n'))
        else:
            self.outfile.write(' Orbital partition method: All AOs in '
                + self.keywords['occupied_projection_basis'].upper()
                + ' from atoms in A\n')

        self.outfile.write('\n')
        if hasattr(self, 'beta_n_act_mos') == False:
            self.outfile.write(' Number of orbitals in active subsystem: %s\n'
                                % self.n_act_mos)
            self.outfile.write(' Number of orbitals in environment: %s\n'
                                % self.n_env_mos)
        else:
            self.outfile.write(' Number of alpha orbitals in active subsystem:'
                                + ' %s\n' % self.n_act_mos)
            self.outfile.write(' Number of beta orbitals in active subsystem:'
                                + ' %s\n' % self.beta_n_act_mos)
            self.outfile.write(' Number of alpha orbitals in environment:'
                                + ' %s\n' % self.n_env_mos)
            self.outfile.write(' Number of beta orbitals in environment:'
                                + ' %s\n' % self.beta_n_env_mos)
        self.outfile.write('\n')
        self.outfile.write(' --- Before embedding --- \n')
        self.outfile.write(' {:<7} {:<6} \t\t = {:>16.10f}\n'.format('('
            + self.keywords['low_level'].upper() +')', 'E[A]', e_act))
        self.outfile.write(' {:<7} {:<6} \t\t = {:>16.10f}\n'.format('('
            + self.keywords['low_level'].upper() +')', 'E[B]', e_env))
        self.outfile.write(' Intersystem interaction G \t = {:>16.10f}\n'.
            format(two_e_cross))
        self.outfile.write(' Nuclear repulsion energy \t = {:>16.10f}\n'.
            format(self.nre))
        self.outfile.write(' {:<7} {:<6} \t\t = {:>16.10f}\n'.format('('
            + self.keywords['low_level'].upper() + ')', 'E[A+B]',
            e_act + e_env + two_e_cross + self.nre))
        self.outfile.write('\n')
        self.outfile.write(' --- After embedding --- \n')
        self.outfile.write(' Embedded SCF E[A] \t\t = {:>16.10f}\n'.
            format(e_act_emb))
        self.outfile.write(' Embedded density correction \t = {:>16.10f}\n'.
            format(correction))
        self.outfile.write(' Embedded HF-in-{:<5} E[A] \t = {:>16.10f}\n'.
            format(self.keywords['low_level'].upper(),
            e_act_emb + e_env + two_e_cross + self.nre + correction))
        self.outfile.write(' <SD_before|SD_after> \t\t = {:>16.10f}\n'.format(
            abs(self.determinant_overlap)))
        self.outfile.write('\n')
        return None

    def print_summary(self, e_mf_emb):
        """
        Prints summary of CL shells.

        Parameters
        ----------
        e_mf_emb : float
            Mean-field embedded energy.
        """
        self.outfile.write('\n Summary of virtual shell energy convergence\n\n')
        self.outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format(
            'Shell #', '# active', ' Correlation', 'Total'))
        self.outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format(
            8*'', 'virtuals', 'energy', 'energy'))
        self.outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format(
            7*'-', 8*'-', 13*'-', 16*'-'))

        for ishell in range(self.n_cl_shell+1):
            self.outfile.write('{:^8d} \t {:^8} \t {:^12.10f} \t {:^12.10f}\n'\
                .format(ishell, self.shell_size*(ishell+1),
                self.correlation_energy_shell[ishell],
                e_mf_emb + self.correlation_energy_shell[ishell]))

        if (ishell == self.max_shell and
            self.keywords['n_cl_shell'] > self.max_shell):
            n_virtuals = self._n_basis_functions - self.n_act_mos
            n_effective_virtuals = (self._n_basis_functions - self.n_act_mos
                                 - self.n_env_mos)
            self.outfile.write('{:^8} \t {:^8} \t {:^12.10f} \t {:^12.10f}\n'.
                format('Eff.', n_effective_virtuals,
                self.correlation_energy_shell[-1],
                e_mf_emb + self.correlation_energy_shell[-1]))
            self.outfile.write('{:^8} \t {:^8} \t {:^12.10f} \t {:^12.10f}\n'.
                format('Full', n_virtuals, self.correlation_energy_shell[-1],
                e_mf_emb + self.correlation_energy_shell[-1]))
        self.outfile.write('\n')
        return None
    
    def print_sigma(self, sigma, ishell):
        """
        Formats the printing of singular values from the CL shells.

        Parameters
        ----------
        sigma : numpy.array or list
            Singular values.
        ishell :int
            CL shell index.
        """
        self.outfile.write('\n{:>10} {:>2d}\n'.format('Shell #', ishell))
        self.outfile.write('  ------------\n')
        self.outfile.write('{:^5} \t {:^14}\n'.format('#','Singular value'))
        for i in range(len(sigma)):
            self.outfile.write('{:^5d} \t {:>12.10f}\n'.format(i, sigma[i]))
        self.outfile.write('\n')
        return None

    def determinant_overlap(self, orbitals, beta_orbitals = None):
        """
        Compute the overlap between determinants formed from the
        provided orbitals and the embedded orbitals

        Parameters
        ----------
        orbitals : numpy.array
            Orbitals to compute the overlap with embedded orbitals.
        beta_orbitals : numpy.array (None)
            Beta orbitals, if running with references other than RHF.
        """
        if self.keywords['high_level_reference'] == 'rhf' and beta_orbitals == None:
            overlap = self.occupied_orbitals.T @ self.ao_overlap @ orbitals
            u, s, vh = np.linalg.svd(overlap)
            self.determinant_overlap = (
                np.linalg.det(u)*np.linalg.det(vh)*np.prod(s))
        else:
            assert beta_orbitals is not None, '\nProvide beta orbitals.'
            alpha_overlap = (self.alpha_occupied_orbitals.T @ self.ao_overlap
                @ beta_orbitals)
            u, s, vh = np.linalg.svd(alpha_overlap)
            self.determinant_overlap = 0.5*(
                np.linalg.det(u)*np.linalg.det(vh)*np.prod(s))
            beta_overlap = (self.beta_occupied_orbitals.T @ self.ao_overlap
                @ beta_orbitals)
            u, s, vh = np.linalg.svd(beta_overlap)
            self.determinant_overlap += 0.5*(
                np.linalg.det(u)*np.linalg.det(vh)*np.prod(s))
        return None

    def count_shells(self):
        """
        Guarantees the correct number of shells are computed.

        Returns
        -------
        max_shell : int
            Maximum number of virtual shells.
        self.n_cl_shell : int
            Number of virtual shells.
        """
        effective_dimension = (self._n_basis_functions - self.n_act_mos
                            - self.n_env_mos)
        self.max_shell = int(effective_dimension/self.shell_size)-1
        if (self.keywords['n_cl_shell']
            > int(effective_dimension/self.shell_size)):
            self.n_cl_shell = self.max_shell
        elif effective_dimension % self.shell_size == 0:
            self.n_cl_shell = self.max_shell - 1
        else:
            self.n_cl_shell = self.keywords['n_cl_shell']
        return self.max_shell, self.n_cl_shell

