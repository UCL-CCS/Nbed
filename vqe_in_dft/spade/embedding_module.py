"""
These really would be better as methods of the Embed classes.
"""

import os
import numpy as np
import inspect
from .embed import Embed
from .psi4_embed import Psi4Embed
from .pyscf_embed import PySCFEmbed
from typing import Dict

import logging

logger = logging.getLogger(__name__)

matrix_dot = lambda A, B: np.einsum('ij,ij', A, B)

def subsystem_orbitals_densities(keywords: Dict, embed: Embed):
    # Whether or not to project occupied orbitals onto another (minimal) basis
    low_level_orbitals = embed.occupied_orbitals
    
    if 'occupied_projection_basis' in keywords:
        projection_orbitals = embed.basis_projection(low_level_orbitals,
            keywords['occupied_projection_basis'])
        n_act_aos = embed.count_active_aos(
            keywords['occupied_projection_basis'])
        ao_overlap = embed.projection_orbitals
    else:
        projection_orbitals = low_level_orbitals
        n_act_aos = embed.count_active_aos()
        ao_overlap = embed.ao_overlap

    # Orbital rotation and partition into subsystems A and B
    rotation_matrix, sigma = embed.orbital_rotation(projection_orbitals,
        n_act_aos, ao_overlap)
    n_act_mos, n_env_mos = embed.orbital_partition(sigma)

    # Defining active and environment orbitals and density
    act_orbitals = low_level_orbitals @ rotation_matrix.T[:, :n_act_mos]
    env_orbitals = low_level_orbitals @ rotation_matrix.T[:, n_act_mos:]
    act_density = 2.0 * act_orbitals @ act_orbitals.T
    env_density = 2.0 * env_orbitals @ env_orbitals.T

    return (act_orbitals, act_density, env_orbitals, env_density)

def save_details(keywords, embed: Embed, v_emb, act_orbitals) -> None:
    """
    If needed, save the requested information
    """
    if keywords['save_embedding_potential']:
        np.savetxt('embedding_potential.txt', v_emb)
        embed.outfile.write(' Embedding potential saved to '
            + 'embedding_potential.txt.\n') 
    if keywords['save_embedded_h_core']:
        np.savetxt('embedded_h_core.txt', embed.h_core + v_emb)
        embed.outfile.write(' Embedded core Hamiltonian saved to '
            + 'embedded_h_core.txt.\n') 
    if keywords['save_embedded_orbitals']:
        np.savetxt('embedded_orbitals.txt', act_orbitals)
        embed.outfile.write(' Embedded orbitals saved to '
            + 'embedded_orbitals.txt.\n') 
    if keywords['run_high_level'] == False:
        embed.outfile.write(' Requested files generated. Ending PsiEmbed.\n\n') 
        raise SystemExit(0)

def run_closed_shell(keywords):
    """
    Runs embedded calculation for closed shell references.

    Parameters
    ----------
    keywords: dict
        Options to control embedded calculations.
    """

    if keywords['package'].lower() == 'psi4':
        embed = Psi4Embed(keywords)
    else:
        embed = PySCFEmbed(keywords)

    embed.run_mean_field()
    embed.header()

    act_orbitals, act_density, env_orbitals, env_density = subsystem_orbitals_densities(keywords, embed)

    # Retrieving the subsytem energy terms and potential matrices
    e_act, e_xc_act, j_act, k_act, v_xc_act = (
        embed.closed_shell_subsystem(act_orbitals))
    e_env, e_xc_env, j_env, k_env, v_xc_env = (
        embed.closed_shell_subsystem(env_orbitals))

    # Computing cross subsystem terms
    j_cross = 0.5 * (matrix_dot(act_density, j_env)
            + matrix_dot(env_density, j_act))

    if keywords['package'].lower() == 'psi4':
        k_cross = 0.5 * embed.alpha * (matrix_dot(act_density, k_env)
                + matrix_dot(env_density, k_act))
    else:
        k_cross = 0.0

    xc_cross = embed.e_xc_total - e_xc_act - e_xc_env
    two_e_cross = j_cross + k_cross + xc_cross

    # TODO: generate orbital (molden) file before embedding

    # Defining the embedded core Hamiltonian
    projector = keywords['level_shift'] * (embed.ao_overlap @ env_density 
        @ embed.ao_overlap)
    v_emb = (j_env - embed.alpha * k_env + embed.v_xc_total - v_xc_act
        + projector)
    #h_core_emb = h_core + embedding_potential + projector

    save_details(keywords, embed, v_emb, act_orbitals)

    embed.run_mean_field(v_emb)

    # Overlap between the C_A determinant and the embedded determinant
    embed.determinant_overlap(act_orbitals)
    
    # Computing the embedded SCF energy
    density_emb = 2.0 * embed.occupied_orbitals @ embed.occupied_orbitals.T
    if keywords['package'].lower() == 'psi4':
        e_act_emb = matrix_dot(density_emb, embed.h_core + embed.j - 0.5 * embed.k)
    else:
        e_act_emb = matrix_dot(density_emb, embed.h_core + 0.5 * embed.j - 0.25 * embed.k)
    
    # Compute the total energy
    correction = (matrix_dot(v_emb, density_emb - act_density))
    e_mf_emb = e_act_emb + e_env + two_e_cross + embed.nre + correction
    embed.print_scf(e_act, e_env, two_e_cross, e_act_emb, correction)

    # --- Post embedded HF calculation ---
    if 'n_cl_shell' not in keywords:
        e_correlation = embed.correlation_energy()
        e_total = e_mf_emb + e_correlation
        embed.outfile.write(' Embedded {:>5}-in-{:<5} E[A] \t = {:>16.10f}\n'.
            format(keywords['high_level'].upper(),
            keywords['low_level'].upper(), e_total))
    else:
        assert isinstance(keywords['n_cl_shell'], int)
        embed.outfile.write('\n Singular values of '
            + str(keywords['n_cl_shell'] + 1) + ' CL shells\n')
        embed.outfile.write(' Shells constructed with the ' 
            + keywords['operator']  + ' operator\n')

        # First virtual shell
        n_act_aos = embed.count_active_aos(keywords['virtual_projection_basis'])
        effective_virtuals = embed.effective_virtuals()

        # Projecting the effective virtuals onto (maybe) another basis
        projected_orbitals = embed.basis_projection(effective_virtuals,
            keywords['virtual_projection_basis'])

        # Defining the first shell
        shell_overlap = (projected_orbitals.T
            @ embed.overlap_two_basis[:n_act_aos, :] @ effective_virtuals)
        rotation_matrix, sigma = embed.orbital_rotation(shell_overlap,
            shell_overlap.shape[1])
        shell_size = (sigma[:n_act_aos] >= 1.0e-15).sum()
        embed.shell_size = shell_size
        span_orbitals = effective_virtuals @ rotation_matrix.T[:,:shell_size]
        kernel_orbitals = effective_virtuals @ rotation_matrix.T[:,shell_size:]
        sigma = sigma[:shell_size]

        # Computing energy in pseudocanonical basis
        e_orbital_span, pseudo_span = embed.pseudocanonical(span_orbitals)
        e_orbital_kernel, pseudo_kernel = (
            embed.pseudocanonical(kernel_orbitals))
        e_correlation = embed.correlation_energy(pseudo_span,
            pseudo_kernel, e_orbital_span, e_orbital_kernel)

        # Printing shell results
        embed.print_sigma(sigma, 0)
        embed.outfile.write((' {}-in-{} energy of shell # {} with {} orbitals '
            + '= {:^12.10f}\n').format(keywords['high_level'].upper(),
            keywords['low_level'].upper(), 0, shell_size,
            e_mf_emb + e_correlation))
        if keywords['molden']:
            embed.molden(virtual_span, virtual_ker, 0, n_env_mos)

        # Checking the maximum number of shells
        max_shell, n_cl_shell = embed.count_shells()

        # Choosing the operator to be used to grow the shells
        embed.ao_operator()
        operator = embed.operator
        ishell = 0

        # Iterative spanning of the virtual space via CL orbitals
        for ishell in range(1, n_cl_shell + 1):
            mo_operator = span_orbitals.T @ operator @ kernel_orbitals
            rotation_matrix, sigma = embed.orbital_rotation(mo_operator,
                shell_size)
            sigma = sigma[:shell_size]

            new_shell = kernel_orbitals @ rotation_matrix.T[:, :shell_size]
            span_orbitals = np.hstack((span_orbitals, new_shell))
            kernel_orbitals = (kernel_orbitals
                @ rotation_matrix.T[:, shell_size:])

            if keywords['molden']:
                embed.molden(virtual_span, virtual_kernel, ishell, n_env_mos)
                embed.heatmap(virtual_span, virtual_kernel, ishell, operator)

            e_orbital_span, pseudo_span = embed.pseudocanonical(span_orbitals)
            e_orbital_kernel, pseudo_kernel = (
                embed.pseudocanonical(kernel_orbitals))
            e_correlation = embed.correlation_energy(pseudo_span,
                pseudo_kernel, e_orbital_span, e_orbital_kernel)

            embed.print_sigma(sigma, ishell)
            embed.outfile.write(
                ' {}-in-{} energy of shell # {} with {} orbitals = {:^12.10f}\n'
                .format(keywords['high_level'].upper(),
                keywords['low_level'].upper(), ishell, shell_size*(ishell + 1),
                e_mf_emb + e_correlation))

        if ishell == max_shell and keywords['n_cl_shell'] > max_shell:
            virtual_span = np.hstack((span_orbitals, kernel_orbitals))
            e_orbital_span, pseudo_span = embed.pseudocanonical(virtual_span)
            e_correlation = embed.correlation_energy()
            embed.outfile.write(' Energy of all ({}) orbitals = {:^12.10f}\n'.
                format(virtual_span.shape[1], e_mf_emb + e_correlation))

        embed.print_summary(e_mf_emb)
        projected_env_correction = matrix_dot(projector,
            act_density - density_emb)
        embed.outfile.write(' Correction from the projected B\t = {:>16.2e}\n'.
            format(projected_env_correction))
        embed.outfile.close()
    if keywords['package'].lower() == 'psi4': os.system('rm newH.dat')
    
    
def run_open_shell(keywords):
    """
    Runs embedded calculation for closed shell references.

    Parameters
    ----------
    keywords: dict
        Options to control embedded calculations.
    """

    if keywords['package'].lower() == 'psi4':
        embed = Psi4Embed(keywords)
    else:
        embed = PySCFEmbed(keywords)
    embed.run_mean_field()
    embed.header()
    alpha_occupied_orbitals = embed.alpha_occupied_orbitals
    beta_occupied_orbitals = embed.beta_occupied_orbitals

    # Whether or not to project occupied orbitals onto another (minimal) basis
    if 'occupied_projection_basis' in keywords:
        alpha_projection_orbitals = embed.basis_projection(
            alpha_occupied_orbitals, keywords['occupied_projection_basis'])
        beta_projection_orbitals = embed.basis_projection(
            beta_occupied_orbitals, keywords['occupied_projection_basis'])
        ao_overlap = embed.ao_overlap
        n_act_aos = embed.count_active_aos(
            keywords['occupied_projection_basis'])
    else:
        alpha_projection_orbitals = alpha_occupied_orbitals
        beta_projection_orbitals = beta_occupied_orbitals
        ao_overlap = embed.ao_overlap
        n_act_aos = embed.count_active_aos(keywords['basis'])

    # Orbital rotation and partition into subsystems A and B
    alpha_rotation_matrix, alpha_sigma = embed.orbital_rotation(
        alpha_projection_orbitals, n_act_aos, ao_overlap)
    beta_rotation_matrix, beta_sigma = embed.orbital_rotation(
        beta_projection_orbitals, n_act_aos, ao_overlap)
    alpha_n_act_mos, alpha_n_env_mos, beta_n_act_mos, beta_n_env_mos = (
        embed.orbital_partition(alpha_sigma, beta_sigma))

    alpha_act_orbitals = (alpha_occupied_orbitals
        @ alpha_rotation_matrix.T[:, :alpha_n_act_mos])
    alpha_env_orbitals = (alpha_occupied_orbitals
        @ alpha_rotation_matrix.T[:, alpha_n_act_mos:])
    beta_act_orbitals = (beta_occupied_orbitals
        @ beta_rotation_matrix.T[:, :beta_n_act_mos])
    beta_env_orbitals = (beta_occupied_orbitals
        @ beta_rotation_matrix.T[:, beta_n_act_mos:])

    alpha_act_density = alpha_act_orbitals @ alpha_act_orbitals.T
    beta_act_density = beta_act_orbitals @ beta_act_orbitals.T
    alpha_env_density = alpha_env_orbitals @ alpha_env_orbitals.T
    beta_env_density = beta_env_orbitals @ beta_env_orbitals.T

    # Retrieving the subsytem energy terms and potential matrices
    (e_act, e_xc_act, alpha_j_act, beta_j_act, alpha_k_act,
        beta_k_act, alpha_v_xc_act, beta_v_xc_act) = (
        embed.open_shell_subsystem(alpha_act_orbitals, beta_act_orbitals))
    (e_env, e_xc_env, alpha_j_env, beta_j_env, alpha_k_env,
        beta_k_env, alpha_v_xc_env, beta_v_xc_env) = (
        embed.open_shell_subsystem(alpha_env_orbitals, beta_env_orbitals))

    # Computing cross subsystem terms
    j_cross = 0.5*(matrix_dot(alpha_j_act, alpha_env_density)
            + matrix_dot(alpha_j_act, beta_env_density)
            + matrix_dot(beta_j_act, alpha_env_density)
            + matrix_dot(beta_j_act, beta_env_density)
            + matrix_dot(alpha_j_env, alpha_act_density)
            + matrix_dot(alpha_j_env, beta_act_density)
            + matrix_dot(beta_j_env, alpha_act_density)
            + matrix_dot(beta_j_env, beta_act_density))
    k_cross = -0.5*embed.alpha*(matrix_dot(alpha_k_act, alpha_env_density)
            + matrix_dot(beta_k_act, beta_env_density)
            + matrix_dot(alpha_k_env, alpha_act_density)
            + matrix_dot(beta_k_env, beta_act_density))
    xc_cross = embed.e_xc_total - e_xc_act - e_xc_env
    two_e_cross = j_cross + k_cross + xc_cross

    # Defining the embedding potential
    ao_overlap = embed.ao_overlap
    alpha_projector = keywords['level_shift']*(ao_overlap @ alpha_env_density
        @ ao_overlap)
    beta_projector = keywords['level_shift']*(ao_overlap @ beta_env_density
        @ ao_overlap)
    alpha_v_emb = (alpha_j_env + beta_j_env - embed.alpha*alpha_k_env
                + alpha_projector + embed.alpha_v_xc_total - alpha_v_xc_act)
    beta_v_emb = (alpha_j_env + beta_j_env - embed.alpha*beta_k_env
                + beta_projector + embed.beta_v_xc_total - beta_v_xc_act)

    if keywords['save_embedded_h_core']:
        embed.outfile.write(' Cannot save embedded core Hamiltonian for '
            + 'open-shells. Saving the embedding potentials instead.\n') 
    if keywords['save_embedding_potential'] or keywords['save_embedded_h_core']:
        np.savetxt('alpha_embedding_potential.txt', v_emb)
        embed.outfile.write(' Alpha embedding potential saved to '
            + 'alpha_embedding_potential.txt.\n') 
        np.savetxt('beta_embedding_potential.txt', v_emb)
        embed.outfile.write(' Beta embedding potential saved to '
            + 'beta_embedding_potential.txt.\n') 
    if keywords['save_embedded_orbitals']:
        np.savetxt('alpha_embedded_orbitals.txt', act_orbitals)
        embed.outfile.write(' Alpha embedded orbitals saved to '
            + 'alpha_embedded_orbitals.txt.\n') 
        np.savetxt('beta_embedded_orbitals.txt', act_orbitals)
        embed.outfile.write(' Beta embedded orbitals saved to '
            + 'beta_embedded_orbitals.txt.\n') 
    if keywords['run_high_level'] == False:
        embed.outfile.write(' Requested files generated. Ending PsiEmbed.\n\n') 
        raise SystemExit(0)

    embed.run_mean_field([alpha_v_emb, beta_v_emb])

    # Overlap between the C_A determinant and the embedded determinant
    embed.determinant_overlap(alpha_act_orbitals, beta_act_orbitals)
    
    # Computing the embedded SCF energy
    alpha_density_emb = (embed.alpha_occupied_orbitals
        @ embed.alpha_occupied_orbitals.T)
    beta_density_emb = (embed.beta_occupied_orbitals
        @ embed.beta_occupied_orbitals.T)
    alpha_j_emb = embed.alpha_j
    beta_j_emb = embed.beta_j
    alpha_k_emb = embed.alpha_k
    beta_k_emb = embed.beta_k
    h_core = embed.h_core
    e_act_emb = (matrix_dot(alpha_density_emb + beta_density_emb, h_core)
                + 0.5*matrix_dot(alpha_density_emb + beta_density_emb,
                alpha_j_emb + beta_j_emb)
                - 0.5*(matrix_dot(alpha_density_emb, alpha_k_emb)
                + matrix_dot(beta_density_emb, beta_k_emb)))
    correction = (matrix_dot(alpha_v_emb, alpha_density_emb - alpha_act_density)
                + matrix_dot(beta_v_emb, beta_density_emb - beta_act_density))
    e_mf_emb = e_act_emb + e_env + two_e_cross + embed.nre + correction
    embed.print_scf(e_act, e_env, two_e_cross, e_act_emb, correction)

    # --- Post embedded HF calculation ---
    if 'n_cl_shell' not in keywords:
        e_correlation = embed.correlation_energy()
        e_total = e_mf_emb + e_correlation
        embed.outfile.write(' Embedded {:>5}-in-{:<5} E[A] \t = {:>16.10f}\n'.
            format(keywords['high_level'].upper(),
            keywords['low_level'].upper(), e_total))
    else:
        raise NotImplementedError('CL orbitals for open-shells coming soon!')
        assert isinstance(keywords['n_cl_shell'], int)
        embed.outfile.write('\n Singular values of '
            + str(keywords['n_cl_shell'] + 1) + ' virtual shells\n')
        embed.outfile.write(' Shells constructed with the %s operator\n'.format(
            keywords['operator']))

        # First virtual shell
        n_act_aos = embed.count_active_aos(keywords['virtual_projection_basis'])
        effective_virtuals = embed.effective_virtuals()

        # Projecting the effective virtuals onto (maybe) another basis
        projected_orbitals = embed.basis_projection(effective_virtuals,
            keywords['virtual_projection_basis'])

        # Defining the first shell
        shell_overlap = (projected_orbitals.T
            @ embed.overlap_two_basis[:n_act_aos, :] @ effective_virtuals)
        rotation_matrix, sigma = embed.orbital_rotation(shell_overlap,
            n_act_aos)
        shell_size = (sigma[:n_act_aos] >= 1.0e-15).sum()
        embed.shell_size = shell_size
        span_orbitals = effective_virtuals @ rotation_matrix.T[:,:shell_size]
        kernel_orbitals = effective_virtuals @ rotation_matrix.T[:,shell_size:]
        sigma = sigma[:shell_size]

        # Pseudocanonicalizing the 1st shell
        e_orbital_span, pseudo_span = embed.pseudocanonical(span_orbitals)
        e_orbital_kernel, pseudo_kernel = embed.pseudocanonical(kernel_orbitals)
        e_correlation = embed.correlation_energy(span_orbitals,
            kernel_orbitals, e_orbital_span, e_orbital_kernel)

        embed.print_sigma(sigma, 0)
        embed.outfile.write('{}-in-{} energy of shell # {} with {} orbitals '
            + '= {:^12.10f}\n'.format(keywords['high_level'].upper(), \
            keywords['low_level'].upper(), 0, shell_size,
            e_mf_emb + e_correlation))
        if keywords['molden']:
            embed.molden(vritual_span, virtual_ker, 0, n_env_mos)

        # Checking the maximum number of shells
        max_shell, n_cl_shell = embed.count_shells()

        # Choosing the operator to be used to grow the shells
        embed.ao_operator()
        operator = embed.operator
           
        # Iterative spanning of the virtual space via CL orbitals
        for ishell in range(1, n_cl_shell + 1):
            mo_operator = span_orbitals.T @ operator @ kernel_orbitals
            rotation_matrix, sigma = embed.orbital_rotation(mo_operator,
                shell_size)
            sigma = sigma[:shell_size]

            new_shell = kernel_orbitals @ rotation_matrix.T[:, :shell_size]
            span_orbitals = np.hstack((span_orbitals, new_shell))
            kernel_orbitals = kernel_orbitals @ rotation_matrix.T[:, shell_size:]

            if keywords['molden']:
                embed.molden(new_shell, ishell)
                embed.heatmap(virtual_span, virtual_kernel, ishell, operator)

            e_orbital_span, pseudo_span = embed.pseudocanonical(span_orbitals)
            e_orbital_kernel, pseudo_kernel = (
                embed.pseudocanonical(kernel_orbitals))
            e_correlation = embed.correlation_energy(span_orbitals,
                kernel_orbitals, e_orbital_span, e_orbital_kernel)

            embed.print_sigma(sigma, ishell)
            embed.outfile.write(' {}-in-{} energy of shell # {} with {} '
                + 'orbitals = {:^12.10f}\n'.format(keywords['high_level'].upper(),
                keywords['low_level'].upper(), ishell, shell_size*(ishell + 1),
                e_mf_emb + e_correlation))

        if ishell == max_shell and keywords['n_cl_shell'] > max_shell:
            virtual_span = np.hstack((span_orbitals, kernel_orbitals))
            e_orbital_span, pseudo_span = embed.pseudocanonical(virtual_span)
            e_correlation = embed.correlation_energy()
            embed.outfile.write(' Energy of all ({}) orbitals = {:^12.10f}\n'.
                format(virtual_span.shape[1], e_mf_emb + e_correlation))

        embed.print_summary(e_mf_emb)
        projected_env_correction = matrix_dot(projector,
            act_density - density_emb)
        embed.outfile.write(' Correction from the projected B\t = {:>16.2e}\n'.
            format(projected_env_correction))
        embed.outfile.close()
    if keywords['package'].lower() == 'psi4':os.system('rm Va_emb.dat Vb_emb.dat')
    
