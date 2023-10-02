#this code is used to completely implement the ZNE mitigation method
#pay attention that we (for now) will only do ZNE on the important terms (pauli string with more weight)
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

#global parameters
seed = 170
algorithm_globals.random_seed = seed
seed_transpiler = seed
shots = 1000
budget = 500 #budget is for optimization iterations

#read the noise model
import qiskit.providers.aer.noise as noise
import pickle
with open('NoiseModel/fakekolkata.pkl', 'rb') as file:
    noise_model = pickle.load(file)
noise_model_t = noise.NoiseModel()
noise_model = noise_model_t.from_dict(noise_model)
backend_noise= AerSimulator(noise_model=noise_model)
#noise_model=NoiseModel.from_backend(FakeMontreal()) 


#read in and process the molecule infomation
bond_length = 1.0
# H2 molecule string
atom_string = f"O 0 0 0; H 0.45 -0.1525 -0.8454"
num_orbitals = 6
coeffs, paulis, HF_bitstring = molecule(atom_string, num_orbitals,charge=1)
n_qubits = len(paulis[0])
print("number of qubits:",n_qubits)

#save_dir = "./"
#result_file = "result.txt"

from qiskit_experiments.library import LocalReadoutError, CorrelatedReadoutError
#temporary comment
'''exp = LocalReadoutError(list(range(n_qubits)))
exp.analysis.set_options(plot=True)
result = exp.run(backend_noise)
#print(result)
mitigator = result.analysis_results(0).value'''

#debug
cafqa_params=[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 3, 0, 3, 1, 0, 1, 0, 2, 0, 1, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1 ,2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 1]
# VQE with CAFQA initialization
#vqe params
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
        "readout_error_mitigation": False,
    }

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
    result = noise_backend.run(tcircuit, shots=shots).result() #ac: bug fixed, add ', shots=shots'
    #针对IIIIIIIIIIII需要特殊对待
    #if _id == len(_id)*'I':
    #    all_counts.append({len(_id)*'0':shots})
    counts = result.get_counts(0) #无论传入的有几项,传出的是列表,我们默认传入的是一项,所以只取'0'  
    
    # Convert from raw measurement counts to the expectation value
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

#debug, ac, only for test
'''pauli=paulis[3]  #先算第一个电路 注意为2时是有问题的!!! ac
imp_circuit=vqe_circuit(n_qubits, parameters, pauli, **vqe_kwargs)

unmitigated=qiskit_executor(imp_circuit)
mitigated = zne.execute_with_zne(imp_circuit, qiskit_executor)
print(f"Unmitigated result {unmitigated:.3f}")
print(f"Mitigated result {mitigated:.3f}")'''

def find_imp_terms(coeffs,threshold=1):
    #to determine which pauli strings need to be ZNE to improve accuracy,return  a list of indexes
    temp=[]
    for i,j in enumerate(coeffs):
        if abs(j)>threshold:
            temp.append(i)

    #debug
    #temp=[0, 1, 2, 26, 38, 46, 60, 61, 124, 172, 204, 216, 631]
    return temp

def cut_paulis(important_terms,coeffs,paulis):
    #将coeffs和paulis都分解为重要和不重要的两部分
    imp_coeffs=[coeffs[ind] for ind in important_terms]
    imp_paulis=[paulis[ind] for ind in important_terms]
    trivial_coeffs=list(coeffs) #modified, 注意它是np.array而非list
    trivial_paulis=list(paulis)
    for ind in important_terms[::-1] :
        #hp sk, ac, NSy L, 逆序
        trivial_coeffs.pop(ind)
        trivial_paulis.pop(ind)
    return imp_coeffs,imp_paulis,trivial_coeffs,trivial_paulis

def compute_expectations_zne(n_qubits, parameters,  noise_backend,paulis, shots, backend, mitigator=None, mode="device_execution", **vqe_kwargs):
    expectations=[]
    unmitigated_expectations=[]
    for pauli in paulis:
        if pauli == len(pauli)*'I':
            expectations.append(1.0)
        else:
            imp_circuit=vqe_circuit(n_qubits, parameters, pauli, **vqe_kwargs)
            unmitigated=qiskit_executor(imp_circuit)
            mitigated = zne.execute_with_zne(imp_circuit, qiskit_executor)
            expectations.append(mitigated)
            unmitigated_expectations.append(unmitigated)
    #unmitigated_expectations is for debugging only, ac
    return expectations,unmitigated_expectations

important_terms=find_imp_terms(coeffs,threshold=1) #format:[1,8,13,...] indexes
#important_terms.pop(0) #ac: temporary,debug
print('important_terms: ',important_terms) #debug, ac
imp_coeffs,imp_paulis,trivial_coeffs,trivial_paulis=cut_paulis(important_terms,coeffs,paulis)
print('imp_coeffs: ',imp_coeffs,'\n imp_paulis',imp_paulis)
#print('\n trivial_coeffs',trivial_coeffs)

#, mitigator=mitigator 暂时删去
imp_expectations, _=compute_expectations_zne(n_qubits, parameters, noise_backend=noise_backend,paulis=imp_paulis, shots=shots, backend=sys_backend, mode="device_execution", **vqe_kwargs)
trivial_expectations = compute_expectations(n_qubits, parameters, noise_backend=noise_backend,paulis=trivial_paulis, shots=shots, backend=sys_backend, mode="device_execution", **vqe_kwargs)

#debug: print, ac
print('imp_expectations: ',imp_expectations)
print('unmitigated_imp_expectations:', _)
print('trivial_expectations: ',trivial_expectations)
imp_energy = np.inner(imp_coeffs, imp_expectations)
trivial_energy=np.inner(trivial_coeffs,trivial_expectations)
print('important energy:',imp_energy,'trivial energy:',trivial_energy,'total energy:',imp_energy+trivial_energy)

def vqe_zne(n_qubits, parameters, coeffs, paulis, coeffs, shots , backend, mode="device_execution", loss_filename=None, params_filename=None,mitigator=None,noise_backend=None, **vqe_kwargs):
    """
    Compute the VQE loss/energy.
    n_qubits (Int): Number of qubits in circuit.
    parameters (Iterable[Float]): VQE parameters.ä
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    loss_filename (String): Path to save file for VQE loss/energy.
    params_filename (String): Path to save file for VQE parameters.
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.
    backend: pay attention that this is the system backend
    Returns:
    (Float) VQE energy. 
    """
    #print('vqe start')  #debug,ac
    start = timer()
    expectations = compute_expectations(n_qubits, parameters,mitigator=mitigator,noise_backend=noise_backend, **kwargs)
    loss = np.inner(coeffs, expectations)
    end = timer()
    print(f'Loss computed by VQE is {loss}, in {end - start} s.')
    
    if loss_filename is not None:
        with open(loss_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([loss])
    
    if params_filename is not None and parameters is not None:
        with open(params_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(parameters)
    return loss


def run_vqe(n_qubits, coeffs, paulis, param_guess, budget, shots, backend, save_dir, loss_file, params_file, mode="device_execution", mitigator=None,noise_backend=None,**vqe_kwargs):
    """
    Run VQE instance. Uses skquant for optimization.
    n_qubits (Int): Number of qubits in circuit.
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    paulis (Iterable[String]): Corresponding Pauli strings in Hamiltonian (same order as coeffs).
    param_guess (Iterable[Float]): Initial guess for VQE parameters.
    budget (Int): Max number of optimization iterations.
    shots (Int): Number of VQE circuit execution shots.
    mode (String): ["no_noisy_sim", "device_execution", "noisy_sim"].
    backend (IBM backend): Can be simulator, fake backend or real backend; irrelevant with mode = "no_noisy_sim".
    save_dir (String): Save directory.
    loss_file (String): Name of save file for VQE loss/energy.
    params_file (String): Name of save file for VQE parameters.
    vqe_kwargs (Dict): Dictionary with additional keyword arguments for vqe() call.

    Returns:
    Tuple of energy estimate and optimized parameters.
    """
    # check right number of parameters given
    _, num_params = efficientsu2_full(n_qubits, vqe_kwargs["ansatz_reps"])
    if len(param_guess) == 0:
        param_guess = [0] * num_params
    assert len(param_guess) == num_params, f"Number of parameters given ({len(param_guess)}) does not match ansatz ({num_params})." 

    bounds = np.array([[0, np.pi*2]]*num_params)  #>>> b=[[1,0]]*3  >>> b  [[1, 0], [1, 0], [1, 0]]  上下界
    initial_point = np.array(param_guess)
    #print('run_vqe') #debug,ac
    vqe_result = minimize(
        lambda c: vqe_zne(
            n_qubits=n_qubits,
            parameters=c, 
            loss_filename=save_dir + "/" + loss_file,
            params_filename=save_dir + "/" + params_file,
            paulis=paulis, 
            coeffs=coeffs,
            shots=shots, 
            backend=backend, 
            mode=mode, 
            mitigator=mitigator,
            noise_backend=noise_backend,
            **vqe_kwargs
        ),  
        initial_point, 
        bounds, 
        budget, 
        method='imfil')
    energy_vqe = vqe_result[0].optval
    params_vqe = vqe_result[0].optpar
    return energy_vqe, params_vqe