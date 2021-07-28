from .embed import Embed

from qiskit import Aer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.operators import OperatorBase
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.aqua.operators import I, X, Z
from qiskit_nature.circuit.library import UCCSD
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit.chemistry import FermionicOperator
from qiskit_nature.drivers import PySCFDriver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem

import logging
logger = logging.getLogger(__name__)

def get_operator(embed: Embed):
    """
    Create a PySCF Driver to get the operator need for VQE
    """
    molecule = embed._mol
    basis = embed._mol.basis

    driver = PySCFDriver(molecule=molecule, basis=basis)
    
    # Get the electronic structure hamiltonian
    es_problem = ElectronicStructureProblem(driver)
    operator = es_problem.second_q_ops()

    # we need to truncate this is some way so that it fits into SPADE

    # to an operator with QubitConverter
    # can set the number of qubits and particles here but not sure 
    # what that will actually do
    converter = QubitConverter(mapper=JordanWignerMapper())
    operator = converter.convert(second_q_op=operator)

    return operator


def setup_vqe(h1=None, h2=None):
    """
    Initial qiskit setup
    """
    seed = 547
    iteration = 125
    aqua_globals.random_seed = seed

    # instance contains the backend
    qi = QuantumInstance(
        backend=Aer.get_backend("qasm_simulator"),
        seed_simulator=seed,
        seed_transpiler=seed
        )
    
    # If we can get the second quantised hamiltonain, that can be converted
    second_q_op = FermionicOperator(h1, h2)
    H2_op = (-1.052373245772859 * I ^ I) + \
        (0.39793742484318045 * I ^ Z) + \
        (-0.39793742484318045 * Z ^ I) + \
        (-0.01128010425623538 * Z ^ Z) + \
        (0.18093119978423156 * X ^ X)


    #need to get the number of qubits
    num_qubits = None

    operator = get_operator(embed=embed)

    # Create a VQE object
    # Not sure on the ansatz import
    vqe = VQE(
        operator=operator, 
        var_form=UCCSD,
        optimizer=SPSA(max_trials=1000, c=0.1, gamma=0.1),
        quantum_instance=qi)

if __name__ == "__main__":
    setup_vqe(None,None)