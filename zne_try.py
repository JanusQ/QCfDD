from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import FakeMontreal
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Estimator
from qiskit.utils import algorithm_globals
from vqe_helpers import *
from vqe_experiment import *
from circuit_manipulation import *
from qiskit_aer import AerSimulator
from mitiq import zne
from mitiq.interface.mitiq_qiskit.qiskit_utils import initialized_depolarizing_noise
import qiskit

seed = 170
algorithm_globals.random_seed = seed
seed_transpiler = seed
shots = 1000
'''psi = RealAmplitudes(num_qubits=2, reps=2)
H = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
theta = [0, 1, 1, 2, 3, 5]'''

#read the noise model
import qiskit.providers.aer.noise as noise
import pickle
with open('NoiseModel/fakekolkata.pkl', 'rb') as file:
    noise_model = pickle.load(file)
noise_model_t = noise.NoiseModel()
noise_model = noise_model_t.from_dict(noise_model)

#noise_model=NoiseModel.from_backend(FakeMontreal())



bond_length = 1.0
# H2 molecule string
atom_string = f"O 0 0 0; H 0.45 -0.1525 -0.8454"
num_orbitals = 6
coeffs, paulis, HF_bitstring = molecule(atom_string, num_orbitals,charge=1)
n_qubits = len(paulis[0])

#save_dir = "./"
#result_file = "result.txt"
budget = 500
vqe_kwargs = {
    "ansatz_reps": 2,
    "init_last": False,
    "HF_bitstring": HF_bitstring
}


from qiskit_experiments.library import LocalReadoutError, CorrelatedReadoutError
with open('NoiseModel/fakekolkata.pkl', 'rb') as file:
    noise_model = pickle.load(file)
    noise_model1 = noise.NoiseModel()
    noise_modelreal = noise_model1.from_dict(noise_model)
print(n_qubits)
backend_noise= AerSimulator(noise_model=noise_modelreal)
#temporary
'''exp = LocalReadoutError(list(range(n_qubits)))
exp.analysis.set_options(plot=True)
result = exp.run(backend_noise)
#print(result)
mitigator = result.analysis_results(0).value'''


#debug
cafqa_params=[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 3, 0, 3, 1, 0, 1, 0, 2, 0, 1, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1 ,2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 1]
# VQE with CAFQA initialization
shots = 8192

qr = QuantumRegister(n_qubits)
cr = ClassicalRegister(n_qubits)
circuit = QuantumCircuit(qr, cr)

parameters=cafqa_params
init_func=hartreefock
ansatz_func=efficientsu2_full
ansatz_reps=2

'''init_func(circuit)
add_ansatz(circuit, ansatz_func, parameters, ansatz_reps)'''

vqe_kwargs = {
        "ansatz_reps": 2,
        "init_last": False,
        "HF_bitstring": HF_bitstring,
        #ac add:
        "readout_error_mitigation": True,
    }
pauli=paulis[1]  #ac: 先算第一个电路,注意在2时会出现错误!!! 发现是因为使用了qiskit中的u门而mitiq不支持导致的,此外,还发现原来的vqe_helper中Y处添加的门有问题
imp_circuit=vqe_circuit(n_qubits, parameters, pauli, **vqe_kwargs)

#tcircs = all_transpiled_vqe_circuits(n_qubits, parameters, paulis, backend, **kwargs)

#noise_model=NoiseModel.from_backend(FakeMontreal())
noise_backend=backend_noise
sys_backend=FakeMontreal()



def qiskit_executor(circuit: qiskit.QuantumCircuit, shots: int = 5000) -> float:
    """Returns the expectation value to be mitigated.

    Args:
        circuit: Circuit to run.
        shots: Number of times to execute the circuit to compute the expectation value.
    """
    tcircuit= transpile(circuit, sys_backend, optimization_level=0, seed_transpiler=seed_transpiler)
    result = noise_backend.run(tcircuit, shots=shots).result()
    counts = result.get_counts(0) #对应pauli=paulis[1]  
    
    # Convert from raw measurement counts to the expectation value
    '''if counts.get("0") is None:
        expectation_value = 0.
    else:
        expectation_value = counts.get("0") / shots'''
    #initiate the expectation value to 0
    expectation_val = 0
    #compute the expectation
    for el in counts.keys(): #keys应当是0-2^n的所有整数
        sign = 1
        #change sign if there are an odd number of ones
        if el.count('1')%2 == 1:
            sign = -1
        expectation_val += sign*counts[el]/shots 
    return expectation_val



unmitigated=qiskit_executor(imp_circuit)
mitigated = zne.execute_with_zne(imp_circuit, qiskit_executor)
print(f"Unmitigated result {unmitigated:.3f}")
print(f"Mitigated result {mitigated:.3f}")
