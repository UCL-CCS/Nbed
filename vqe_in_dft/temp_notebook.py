# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # VQE in DFT with PsiEmbed and Qiskit
#
# Here we define the inputs as required by PsiEmbed. Note that we'll follow the logic of `embedding_module/run_open_shell`.
# %% [markdown]
# We can think of this procedure are requiring three steps:
#
# 1. Pre-embedding
#
#     Here we define the problem, and run a low-level calculation of the whole system. From this we obtain the pre-embedded density matrices $\gamma^A$ and $\gamma^B$
#
#     We then define the level-shift projector $P$ and embedding potential $V_{emb}$.
#
# 2. Embedding
#
#     Using $V_{emb}$ we run a high-level method simulation of the active region to get the embedded density matrix $\gamma^A_{emb}$.
#
#     We calculate the correction term $tr[V_{emb}(\gamma^A_{emb}-\gamma^A)]$
#
# 3. Post-embedding
#
#     Finally we calculate the embedded energy, by removing $V_{emb}$ from the Hamiltonian, and using density matrix $\gamma^A_{emb}$.
#
#     The total energy is then given by: $E = E[\gamma^A_{emb}] + E[\gamma^B] + g[\gamma^A, \gamma^B] + E_{nuclear} + tr[V_{emb}(\gamma^A_{emb}-\gamma^A)]$
# %% [markdown]
# # 0. Set Parameters
#
# First we'll set the parameters

# %%
from copy import copy
from typing import Dict
from spade import fill_defaults
import numpy as np
from scipy import linalg
from spade.main import driver
from spade.embedding_module import run_closed_shell

ethylene = """
H    2.933  -0.150  -9.521
H    2.837   1.682  -9.258
C    3.402   0.773  -9.252
C    4.697   0.791  -8.909
H    5.262  -0.118  -8.904
H    5.167   1.714  -8.641
    """

methanol = """
O     -0.6582     -0.0067      0.1730 
C      0.7031      0.0083     -0.1305
H     -1.1326     -0.0311     -0.6482
H      1.2001      0.0363      0.8431
H      0.9877      0.8943     -0.7114
H      1.0155     -0.8918     -0.6742
  """
# H     -1.1326     -0.0311     -0.6482 <--- this goes with the oxygen


formadehyde = """
C      0.5979      0.0151      0.0688
H      1.0686     -0.1411      1.0408
H      1.2687      0.2002     -0.7717
O     -0.5960     -0.0151     -0.0686
  """

water = """
O          0.00000        0.00000        0.1653507
H          0.00000        0.7493682     -0.4424329
H          0.00000       -0.7493682     -0.4424329
    """

# this isn't right but lests just use it to try
h_peroxide = """
O          0.00000        0.00000        0.00000
O          1.00000        0.00000        0.00000
H          0.00000        0.50000        0.00000
H          1.00000       -0.50000        0.00000
"""

fci_values = {
    "formaldehyde": -113.58371577461213,
    "water": -75.7315,
    "ethylene": -77.9892,
    "methanol": -114.75641069780156,
}

options = {}
options["geometry"] = ethylene
options[
    "n_active_atoms"
] = 3  # number of active atoms (first n atoms in the geometry string)

run_fci = False

options["basis"] = "STO-6G"  # basis set
options["low_level"] = "b3lyp"  # level of theory of the environment
options["high_level"] = "mp2"  # level of theory of the embedded system
options["low_level_reference"] = "rhf"
options["high_level_reference"] = "rhf"
options["package"] = "pyscf"

keywords = fill_defaults(options)

e_psiembed = run_closed_shell(keywords)

# %% [markdown]
# # 1. Low-level whole system calculation
#
# The first step is to run a mean field caluclation of the whole system.
#
# The Embed class and its subclasses have a method to do this which also sets the following properties:
#     Exchange correlation potentials (v_xc_total if embedding potential is not set, or alpha/beta_v_xc_total)
#

# %%
from pyscf import gto, scf, cc, ao2mo, fci

mol = gto.Mole(atom=keywords["geometry"], basis=keywords["basis"], charge=0).build()

ks = scf.RKS(mol)
ks.conv_tol = keywords["e_convergence"]
ks.xc = keywords["low_level"]
e_initial = ks.kernel()

# Store the initial value of h core as this is needed later,
# but is overwritten

initial_h_core = ks.get_hcore()

mol_copy = copy(mol)


# %%
hf = mol_copy.RHF().run()
if run_fci:
    ref_fci = fci.FCI(hf)
    ref_fci.conv_tol = keywords["e_convergence"]
    fci_result = ref_fci.kernel()

    # This DOES have nuclear energy included!
    expected_energy = fci_result[0]

else:
    ref_cc = cc.CCSD(hf)
    ref_cc.conv_tol = keywords["e_convergence"]
    cc_result = ref_cc.kernel()

    expected_energy = hf.energy_tot() + cc_result[0]

f"{expected_energy=}"

# %% [markdown]
# # 2. Orbital Localisation
# Find the orbitals of the active space and environment, using SPADE.

# %%
n_occupied_orbitals = np.count_nonzero(ks.mo_occ == 2)
occupied_orbitals = ks.mo_coeff[:, :n_occupied_orbitals]

n_act_aos = mol.aoslice_by_atom()[keywords["n_active_atoms"] - 1][-1]
ao_overlap = ks.get_ovlp()

# Orbital rotation and partition into subsystems A and B
# rotation_matrix, sigma = embed.orbital_rotation(occupied_orbitals,
#    n_act_aos, ao_overlap)

rotated_orbitals = linalg.fractional_matrix_power(ao_overlap, 0.5) @ occupied_orbitals
_, sigma, right_vectors = linalg.svd(rotated_orbitals[:n_act_aos, :])

# n_act_mos, n_env_mos = embed.orbital_partition(sigma)
value_diffs = sigma[1:] - sigma[:-1]
n_act_mos = np.argmin(value_diffs)
n_env_mos = len(sigma) - n_act_mos

# Defining active and environment orbitals and density
act_orbitals = occupied_orbitals @ right_vectors.T[:, :n_act_mos]
env_orbitals = occupied_orbitals @ right_vectors.T[:, n_act_mos:]
act_density = 2.0 * act_orbitals @ act_orbitals.T
env_density = 2.0 * env_orbitals @ env_orbitals.T


# %%
n_occupied_orbitals

# %% [markdown]
# # 3. Calculate the cross subsytem terms

# %%
# Retrieving the subsytem energy terms and potential matrices

# The function called looks like this
def closed_shell_subsystem(scf, density):
    # It seems that PySCF lumps J and K in the J array
    j = scf.get_j(dm=density)
    k = np.zeros(np.shape(j))
    two_e_term = scf.get_veff(scf.mol, density)
    e_xc = two_e_term.exc
    v_xc = two_e_term - j

    # Energy
    e = np.einsum("ij,ij", density, scf.get_hcore() + j / 2) + e_xc
    return e, e_xc, j, k, v_xc


e_act, e_xc_act, j_act, k_act, v_xc_act = closed_shell_subsystem(ks, act_density)
e_env, e_xc_env, j_env, k_env, v_xc_env = closed_shell_subsystem(ks, env_density)

# Computing cross subsystem terms
# Note that the matrix dot product is equivalent to the trace.
j_cross = 0.5 * (
    np.einsum("ij,ij", act_density, j_env) + np.einsum("ij,ij", env_density, j_act)
)

k_cross = 0.0

xc_cross = ks.get_veff().exc - e_xc_act - e_xc_env
two_e_cross = j_cross + k_cross + xc_cross
print(f"{e_act=},{e_xc_act=}")  # , {j_act=}, {k_act=}, {v_xc_act=}")
f"{two_e_cross=}, {xc_cross=}"

# %% [markdown]
# # 4. Define $V_{emb}$
#
# We can now define the projector used to orthogonalise the Molecular and Atomic orbitals. From this we calculate the embedding potential.
#
# $P_{\alpha, \beta} = S\gamma^BS$
#
# From this we can now also define the embedding potential.
#
# $V_{emb} = g[\gamma^A, \gamma^B] - g[\gamma^A] + \mu P$

# %%
# Define the mu-projector
print(f"{keywords['level_shift']=}")
projector = keywords["level_shift"] * (ks.get_ovlp() @ env_density @ ks.get_ovlp())

v_xc_total = ks.get_veff() - ks.get_j()

# Defining the embedded core Hamiltonian
v_emb = j_env + v_xc_total - v_xc_act + projector


# %%


# %% [markdown]
# # 5 Run HF of full system with $V_{emb}$ to get $\gamma^A_{emb}$
#
# Here, PsiEmbed gives us the option to stop, outputting values for calculation by other means.
#
# To continue, we run the mean field method, but with the embedding potentials as calulated.
#
# Note we don't need to run a high-level calculation here as that doesn't change the density matrix.

# %%
embedded_scf = scf.RHF(mol)
embedded_scf.conv_tol = keywords["e_convergence"]
embedded_scf.mol.nelectron = 2 * n_act_mos

h_core = embedded_scf.get_hcore()

embedded_scf.get_hcore = lambda *args, **kwargs: h_core + v_emb

embedded_scf.kernel()

embedded_occ_orbs = embedded_scf.mo_coeff[:, embedded_scf.mo_occ > 0]
embedded_density = 2 * embedded_occ_orbs @ embedded_occ_orbs.T

e_emb = embedded_scf.energy_elec(dm=embedded_density, vhf=embedded_scf.get_veff())[0]

print(f"{e_emb=}")


# %%
embedded_occ_orbs.shape

# %% [markdown]
# # 6 Calculate correction term

# %%
# Compute the correction
# - There are two versions used for different embeddings
dm_correction = np.einsum("ij,ij", v_emb, embedded_density - act_density)
wf_correction = np.einsum("ij,ij", act_density, v_emb)

print(f"{wf_correction=}, {dm_correction=}")

# %% [markdown]
# # 7 Calculate $E[\gamma^A_{emb}]$
#
# We calculate the Hartree-fock energy of the embedded region, we then add correlation later.

# %%
from scipy.linalg import LinAlgError

embedded_scf.get_hcore = lambda *args, **kwargs: h_core

# Can use either of these methods
# This needs to change if we're not using PySCFEmbed
# The j and k matrices are defined differently in PySCF and Psi4
e_act_emb_explicit = np.einsum(
    "ij,ij",
    embedded_density,
    initial_h_core + 0.5 * embedded_scf.get_j() - 0.25 * embedded_scf.get_k(),
)
e_act_emb = embedded_scf.energy_elec(dm=embedded_density, vhf=embedded_scf.get_veff())[
    0
]
print(f"E_HF = {e_act_emb}")
print(f"Difference between HF methods: {e_act_emb - e_act_emb_explicit}")

try:
    # Run CCSD as WF method
    ccsd = cc.CCSD(embedded_scf)
    ccsd.conv_tol = keywords["e_convergence"]

    # Set which orbitals are to be frozen
    # The environment orbitals energies have been increased by the projector
    # so they are now at the end of the list, as orbitals are ordered by energy
    shift = mol.nao - n_env_mos
    fos = [i for i in range(shift, mol.nao)]
    ccsd.frozen = fos

    ccsd.run()
    correlation = ccsd.e_corr
    e_act_emb += correlation

except LinAlgError:
    print("Use the HF energy")
    pass

f"{e_act_emb=}"

# %% [markdown]
# # 8 Add all the parts up.
#
# e_act_emb : $\epsilon[\gamma^A_{emb}]$
# >energy of the embedded region
#
# e_env : $E[\gamma^B]$
# >energy of the environment
#
# two_e_cross : $g[\gamma^A, \gamma^B]$
# >non-additive two electron term
#
# embed.nre
# >The Coulomb energy from nuclear repulsion.
#
# correction : $tr[(\gamma^A_{emb} - \gamma^A)(h^{A in B} - h)]$ (or $tr[\gamma^A(h^{A in B} - h)]$ )
# > Correction for embedding

# %%
e_nuc = mol.energy_nuc()

e_mf_emb = e_act_emb + e_env + two_e_cross + dm_correction + e_nuc
print("Component contributions")
print(
    f"{e_act_emb=}, {e_env=}, {two_e_cross=}, {e_nuc=}, {dm_correction=}, {wf_correction=}\n"
)

# Print out the final value.
print(f"FCI Energy:\t\t{expected_energy}")
print(f"DFT Energy:\t\t{e_initial}")
print(f"Error:\t\t\t{(expected_energy-e_initial)*100/expected_energy:.2f}%")
print(f"Embedding Energy:\t{e_mf_emb}")
print(f"Error:\t\t\t{(expected_energy-e_mf_emb)*100/expected_energy:.2f}%")
