import numpy as np

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter,JordanWignerMapper

from skquant.opt import minimize
import hypermapper
import json
import sys
from numbers import Number

from vqe_helpers import *
from circuit_manipulation import *


def molecule(atom_string, new_num_orbitals=None,charge=0, **kwargs):
    """
    Compute Hamiltonian for molecule in qubit encoding using Qiskit Nature.
    atom_string (String): string to describe molecule, passed to PySCFDriver.
    new_num_orbitals (Int): Number of orbitals in active space (if None, use default result from PySCFDriver).
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.

    Returns:
    (Iterable[Float], Iterable[String], String) (Pauli coefficients, Pauli strings, Hartree-Fock bitstring)
    """
    #modified
    #converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
    converter = QubitConverter(JordanWignerMapper(), two_qubit_reduction=True)
    driver = PySCFDriver(
        atom=atom_string,
        basis="sto3g",
        charge=charge,#modified
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()
    if new_num_orbitals is not None:
        num_electrons = (problem.num_alpha, problem.num_beta)
        transformer = ActiveSpaceTransformer(num_electrons, new_num_orbitals)
        problem = transformer.transform(problem)
    ferOp = problem.hamiltonian.second_q_op()
    qubitOp = converter.convert(ferOp, problem.num_particles)
    initial_state = HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        converter
    )
    bitstring = "".join(["1" if bit else "0" for bit in initial_state._bitstr])
    # need to reverse order bc of qiskit endianness
    paulis = [x[::-1] for x in qubitOp.primitive.paulis.to_labels()]
    # add the shift as extra I pauli
    paulis.append("I"*len(paulis[0]))
    paulis = np.array(paulis)
    coeffs = list(qubitOp.primitive.coeffs)
    # add the shift (nuclear repulsion)
    coeffs.append(problem.nuclear_repulsion_energy)
    coeffs = np.array(coeffs).real
    return coeffs, paulis, bitstring

def ising_model(N, Jx, h, Jy=0., periodic=False):
    """
    Constructs qubit Hamiltonian for linear Ising model.
    H = sum_{i=0...N-2} (Jx_i X_i X_{i+1} + Jy_i Y_i Y_{i+1}) + sum_{i=0...N-1}  h_i Z_i

    N (Int): # sites/qubits.
    Jx (Float, Iterable[Float]): XX strength, either constant value or list (values for each pair of neighboring sites).
    h (Float, Iterable[Float]): Z self-energy, either constant value or list (values for each site).
    Jy (Float, Iterable[Float]): YY strength, either constant value or list (values for each pair of neighboring sites).
    periodic: If periodic boundary conditions. If True, include term X_0 X_{N-1} and Y_0 Y_{N-1}.

    Returns:
    (Iterable[Float], Iterable[String], String) (Pauli coefficients, Pauli strings, "0"*N)
    """
    if isinstance(Jx, Number):
        if periodic:
            Jx = [Jx] * N
        else:
            Jx = [Jx] * (N-1)
    if isinstance(Jy, Number):
        if periodic:
            Jy = [Jy] * N
        else:
            Jy = [Jy] * (N-1)
    if isinstance(h, Number):
        h = [h] * N        
    if N > 1:
        assert len(Jx) == N if periodic else len(Jx) == N-1, "Jx has wrong length"
        assert len(Jy) == N if periodic else len(Jy) == N-1, "Jy has wrong length"
        assert len(h) == N, "h has wrong length"
    coeffs = []
    paulis = []
    # add XX terms
    for j in range(N-1):
        if np.abs(Jx[j]) > 1e-12:
            coeffs.append(Jx[j])
            paulis.append("I"*j+"XX"+"I"*(N-j-2))
    if N > 2 and periodic and np.abs(Jx[N-1]) > 1e-12:
        coeffs.append(Jx[N-1])
        paulis.append("X"+"I"*(N-2)+"X")
    # add YY terms
    for j in range(N-1):
        if np.abs(Jy[j]) > 1e-12:
            coeffs.append(Jy[j])
            paulis.append("I"*j+"YY"+"I"*(N-j-2))
    if N > 2 and periodic and np.abs(Jy[N-1]) > 1e-12:
        coeffs.append(Jy[N-1])
        paulis.append("Y"+"I"*(N-2)+"Y")
    # add Z terms
    for j in range(N):
        if np.abs(h[j]) > 1e-12:
            coeffs.append(h[j])
            paulis.append("I"*j+"Z"+"I"*(N-j-1))
    return coeffs, paulis, "0"*N

def run_vqe(n_qubits, coeffs, paulis, param_guess, budget, shots, mode, backend, save_dir, loss_file, params_file, vqe_kwargs):
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

    bounds = np.array([[0, np.pi*2]]*num_params)  #>>> b=[[1,0]]*3  >>> b  [[1, 0], [1, 0], [1, 0]]  限制取值范围? 不,当参数type为float时,这表示的是上下界
    initial_point = np.array(param_guess)
    vqe_result = minimize(
            lambda c: vqe(
                n_qubits=n_qubits,
                parameters=c, 
                loss_filename=save_dir + "/" + loss_file,
                params_filename=save_dir + "/" + params_file,
                paulis=paulis, 
                coeffs=coeffs,
                shots=shots, 
                backend=backend, 
                mode=mode, 
                **vqe_kwargs
            ),  #lambda函数,也就是匿名函数,这里定义了一个cost function, c是这个函数传入的参数,返回值=vqe(...)
            initial_point, 
            bounds, 
            budget, 
            method='imfil')
    energy_vqe = vqe_result[0].optval
    params_vqe = vqe_result[0].optpar
    return energy_vqe, params_vqe

#下面这个函数是本程序中最重要的
def run_cafqa(n_qubits, coeffs, paulis, param_guess, budget, shots, mode, backend, save_dir, loss_file, params_file, vqe_kwargs):
    """
    Run CAFQA VQE instance. Uses stim for fast Clifford circuit simulation and hypermapper for discrete optimization.
    n_qubits (Int): Number of qubits in circuit.
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    paulis (Iterable[String]): Corresponding Pauli strings in Hamiltonian (same order as coeffs).
    param_guess (Iterable[0...3]): Initial guess for CAFQA VQE parameters, which are factors for pi/2. E.g. param_guess = [1,0,0,2,3,1] for 6-parameter VQE with real parameters [pi/2,0,0,pi,3pi/2,pi/2].
    budget (Int): Max number of optimization iterations.
    shots (Int): --Not relevant here--.
    mode (String): --Not relevant here--.
    backend (IBM backend): --Not relevant here--.
    save_dir (String): Save directory.
    loss_file (String): Name of save file for VQE loss/energy.
    params_file (String): Name of save file for VQE parameters.
    vqe_kwargs (Dict): Dictionary with additional keyword arguments for vqe_cafqa_stim() call.

    Returns:
    Tuple of energy estimate and optimized CAFQA parameters.
    """
    # check right number of parameters given
    _, num_params = efficientsu2_full(n_qubits, vqe_kwargs["ansatz_reps"])
    if len(param_guess) == 0:
        param_guess = [0] * num_params
    assert len(param_guess) == num_params, f"Number of parameters given ({len(param_guess)}) does not match ansatz ({num_params})." 

    hypermapper_config_path = save_dir + "/hypermapper_config.json"
    config = {}
    config["application_name"] = "cafqa_optimization"
    config["optimization_objectives"] = ["value"]
    number_of_RS = budget//1 #在 Python 中，你使用双斜线 // 运算符来执行向下取整除
    config["design_of_experiment"] = {}
    config["design_of_experiment"]["number_of_samples"] = number_of_RS  #NMS
    config["optimization_iterations"] = budget
    config["models"] = {}
    config["models"]["model"] = "random_forest"
    config["input_parameters"] = {}
    config["print_best"] = True
    config["print_posterior_best"] = True
    for i in range(num_params):
        x = {}
        x["parameter_type"] = "ordinal"  #序数,与integer和float区分
        x["values"] = [0, 1, 2, 3]
        x["parameter_default"] = param_guess[i]
        config["input_parameters"]["x" + str(i)] = x
    config["log_file"] = save_dir + '/hypermapper_log.log'
    config["output_data_file"] = save_dir + "/hypermapper_output.csv"
    with open(hypermapper_config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)   #以上所做的,全都是定义一个config字典量并存入文件hypermapper_config_path中

    stdout = sys.stdout  #临时保存当前的输出方式到stdout变量, 这一行的意义是什么??
    
    #该函数运行之后,会将结果存到config中定义的log_file和output_data_file中
    #parameters_file

    hypermapper.optimizer.optimize(       
        hypermapper_config_path,   
        lambda x: vqe_cafqa_stim(
            inputs=x,
            n_qubits=n_qubits,
            loss_filename=save_dir + "/" + loss_file,
            params_filename=save_dir + "/" + params_file,
            paulis=paulis, 
            coeffs=coeffs,
            shots=shots, 
            backend=backend, 
            mode=mode, 
            **vqe_kwargs
        ))  #black_box_function
    sys.stdout = stdout

    energy_cafqa = np.inf  #infinite 无穷大
    x_cafqa = None
    with open(config["log_file"]) as f:  #读取minimize函数输出的内容
        lines = f.readlines()
        counter = 0
        for idx, line in enumerate(lines[::-1]):  #[::-1]： 代表从全列表倒序取
            #enumerate is useful for obtaining an indexed list:    (0, seq[0]), (1, seq[1]), (2, seq[2]), ...
            if line[:16] == "Best point found" or line[:29] == "Minimum of the posterior mean": #line[:16]表示这一行前16个字符(组成的字符串或者列表)
                '''格式大致如下:

                Best point found:
                x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,value
                0,0,0,0,0,0,0,0,0,0,0,0,-1.0661086493179344
                '''
                counter += 1
                parts = lines[-1-idx+2].split(",") #.split: Return a list of the substrings in the string, using sep as the separator string.
                #parts相当于将lines[-1-idx+2]这一行的内容,以原文中的','作为分割的标志,分割成一个个字符串在一个列表中
                energy = float(parts[-1])
                if energy < energy_cafqa:
                    energy_cafqa = energy
                    x_cafqa = [int(y) for y in parts[:-1]] #记录各个xi的坐标
            if counter == 2:
                break
    return energy_cafqa, x_cafqa
