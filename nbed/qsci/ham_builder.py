from pyscf import gto, mcscf, ao2mo
import numpy as np
from typing import Tuple, Union, Optional, Dict
from openfermion.ops import InteractionOperator
from openfermion.transforms.opconversions.jordan_wigner import _jordan_wigner_interaction_op
from openfermion.transforms import get_fermion_operator
from openfermion import QubitOperator, FermionOperator
from openfermion import count_qubits

def build_integrals(mol:gto.Mole, C_mat:np.array, active_space_MO_idxs:np.array, ncas:int, nelec_emb:Tuple[int, int],
                    emb_core:Optional[Union[np.array, None]]=None, ncore:int=0) -> Tuple[float, np.array, np.array, np.array, float]:
    """
    Get Hamiltonian integrals

    Args:
        mol (gto.Mole): PySCF molecule object -> note charge should be included if ignoring environment electrons
        C_mat (np.array): MO coefficient matrix
        active_space_MO_idxs (np.array): List of which spatial MO indices are active
        ncas (int): number of cas orbitals
        nelec_emb (Tuple[int, int]): number of alpha and beta electrons for embedded problem
        emb_core (np.array): embedding potential and projectors described in AO basis 
        ncore (int): number of core electrons
    Returns:
        energy_core (float): core energy from cas + nuclear energy
        hcore_spin_mo (np.array): 2ncas x 2ncas matrix of h_pq for molecular Hamiltonian
        eri_spin_mo  (np.array): 2ncas x 2ncas x 2ncas x 2ncas tensor of g_pqrs for molecular Hamiltonian
        h1std_mo_spin (np.array): h_pq for molecular Hamiltonian without core Hamiltonian modificiations
        energy_core_std (float):  core energy from cas without core Hamiltonian modificiations
    """
    cas_emb = mcscf.CASCI(mol, ncas, nelec_emb, ncore=ncore)
    C_emb_ordered_subspace = mcscf.addons.sort_mo(cas_emb, C_mat, active_space_MO_idxs, base=0)


    h1std_mo, energy_core_std = cas_emb.get_h1eff(mo_coeff=C_emb_ordered_subspace)
    h1std_mo_spin = np.zeros((2*ncas, 2*ncas), dtype=float)
    h1std_mo_spin[0::2, 0::2] = h1std_mo_spin[1::2, 1::2] = h1std_mo 


    ## build embedded Hamiltonian object!
    hcore_std = cas_emb.get_hcore()
    cas_emb.get_hcore = lambda *args: hcore_std + emb_core
    
    h1eff_mo, energy_core = cas_emb.get_h1eff(mo_coeff=C_emb_ordered_subspace)
    ## energy_core: includes nuclear term & h1eff has been transformed to MO basis

    eri_cas_mo_S1 = ao2mo.restore(1, 
                                  cas_emb.get_h2eff(mo_coeff=C_emb_ordered_subspace),
                                   cas_emb.ncas)

    hcore_spin_mo = np.zeros((2*ncas, 2*ncas), dtype=float)
    hcore_spin_mo[0::2, 0::2] = hcore_spin_mo[1::2, 1::2] = h1eff_mo 

    eri_spin_mo = np.zeros((2*ncas, 2*ncas, 2*ncas, 2*ncas), dtype=float)
    phys_S1 = eri_cas_mo_S1.transpose(0, 3, 2, 1)
    eri_spin_mo[ ::2, ::2, ::2, ::2] = phys_S1
    eri_spin_mo[1::2,1::2,1::2,1::2] = phys_S1

    eri_spin_mo[0::2,1::2,1::2,0::2] = phys_S1
    eri_spin_mo[1::2,0::2,0::2,1::2] = phys_S1

    return energy_core, hcore_spin_mo, eri_spin_mo, h1std_mo_spin, energy_core_std


def build_molecular_H(energy_core: float, hcore_spin_mo:np.array, eri_spin_mo:np.array, 
                      return_type:Optional[str]="jordan_wigner") -> Union[InteractionOperator, QubitOperator, FermionOperator]:
    """
    Build molecular Hamiltonian from molecular intregrals

    Args:
        energy_core (float): core energy from cas + nuclear energy
        hcore_spin_mo (np.array): 2ncas x 2ncas matrix of h_pq for molecular Hamiltonian
        eri_spin_mo  (np.array): 2ncas x 2ncas x 2ncas x 2ncas tensor of g_pqrs for molecular Hamiltonian
        return_type (str): what encoding to use... if not recognized then returns InteractionOperator
    Returns:
        H (InteractionOperator, QubitOperator, FermionOperator): Molecular Hamiltonian
    """

    H = InteractionOperator(energy_core, hcore_spin_mo, 0.5*eri_spin_mo)

    # TODO: add more encodings here
    if return_type == "jordan_wigner":
        H = _jordan_wigner_interaction_op(H)
    elif return_type == "fermion":
        H = get_fermion_operator(H)
    else:
        pass

    return H


def qubit_op_to_dict(Op:QubitOperator) -> Dict[str, float]:
    """
    Convert openfermion qubit operator into a dictionary to dump
    """
    n_qubits = count_qubits(Op)
    op_dict = dict()
    for pauli_term, coeff in Op.terms.items():
        P_str = ["I"]*n_qubits
        for q_idx, s_str in pauli_term:
            P_str[q_idx] = s_str
        op_dict["".join(P_str)] = float(coeff)
    return op_dict
