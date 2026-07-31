from pyscf import gto, mcscf, ao2mo
import numpy as np
from typing import Tuple, Union, Optional, Dict
from openfermion.ops import InteractionOperator
from openfermion.transforms.opconversions.jordan_wigner import _jordan_wigner_interaction_op
from openfermion.transforms import get_fermion_operator
from openfermion import QubitOperator, FermionOperator
from openfermion import count_qubits

def build_spin_integrals(hcore_spatial_mo:np.array, eri_spatial_mo:np.array, norb_spatial:int) -> Tuple[float, np.array, np.array, np.array, float]:
    """
    Get Hamiltonian SPIN integrals in physicist order
    """

    
    h1_mo_spin = np.zeros((2*norb_spatial, 2*norb_spatial), dtype=float)
    h1_mo_spin[0::2, 0::2] = h1_mo_spin[1::2, 1::2] = hcore_spatial_mo 

    eri_mo_S1 = ao2mo.restore(1, 
                                    eri_spatial_mo,
                                    norb_spatial)

    eri_spin_mo = np.zeros((2*norb_spatial, 2*norb_spatial, 2*norb_spatial, 2*norb_spatial), dtype=float)
    phys_S1 = eri_mo_S1.transpose(0, 3, 2, 1)
    eri_spin_mo[ ::2, ::2, ::2, ::2] = phys_S1
    eri_spin_mo[1::2,1::2,1::2,1::2] = phys_S1

    eri_spin_mo[0::2,1::2,1::2,0::2] = phys_S1
    eri_spin_mo[1::2,0::2,0::2,1::2] = phys_S1

    return h1_mo_spin, eri_spin_mo


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


def build_number_operator(n_qubits:int, type="fermion") -> Union[QubitOperator, FermionOperator]:
    """
    Build number operator for n qubits
    """
    if type == "fermion":
        Na = FermionOperator()
        Nb = FermionOperator()
        for i in range(n_qubits//2):
            Na += FermionOperator(f"{2*i}^ {2*i}", 1) 
            Nb += FermionOperator(f"{2*i+1}^ {2*i+1}", 1) 
        return Na, Nb
    elif type == "qubit_jw":
        Na = QubitOperator()
        Nb = QubitOperator()
        for i in range(n_qubits//2):
            Na += QubitOperator(f"", 0.5)  - QubitOperator(f"Z{2*i}", 0.5) 
            Nb +=  QubitOperator(f"", 0.5) - QubitOperator(f"Z{2*i+1}", 0.5) 
    else:
        raise ValueError(f"Invalid type: {type}")
    return Na,Nb
