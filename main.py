import numpy as np
from qiskit.providers.fake_provider import *
import sys
from vqe_experiment import *



def main():
    bond_length = 1.0
    # H2 molecule string
    atom_string = f"O 0 0 0; H 0.45 -0.1525 -0.8454"
    num_orbitals = 6
    coeffs, paulis, HF_bitstring = molecule(atom_string, num_orbitals,charge=1)
    n_qubits = len(paulis[0])
    
    print(coeffs) #to see the order
    
    save_dir = "./"
    result_file = "result.txt"
    budget = 500
    vqe_kwargs = {
        "ansatz_reps": 2,
        "init_last": False,
        "HF_bitstring": HF_bitstring,
        #ac add:
        "readout_error_mitigation": True,
    }
    cafqa_params=None
    #assign the optimal parameters
    cafqa_params=[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 3, 0, 3, 1, 0, 1, 0, 2, 0, 1, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1 ,2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 1]
    
    if cafqa_params==None: #if the optimal parameters are not given, run CAFQA to find them.
        # run CAFQA
        cafqa_guess = [] # will start from all 0 parameters
        loss_file = "cafqa_loss.txt"
        params_file = "cafqa_params.txt"
        cafqa_energy, cafqa_params = run_cafqa(
            n_qubits=n_qubits,
            coeffs=coeffs,
            paulis=paulis,
            param_guess=cafqa_guess,
            budget=budget,
            shots=None,
            mode=None,
            backend=None,
            save_dir=save_dir,
            loss_file=loss_file,
            params_file=params_file,
            vqe_kwargs=vqe_kwargs
        )
        with open(save_dir + result_file, "w") as res_file:
            res_file.write(f"CAFQA energy:\n{cafqa_energy}\n")
            res_file.write(f"CAFQA params (x pi/2):\n{np.array(cafqa_params)}\n\n")
    else:
        pass


    from qiskit_experiments.library import LocalReadoutError, CorrelatedReadoutError
    with open('NoiseModel/fakekolkata.pkl', 'rb') as file:
        noise_model = pickle.load(file)
        noise_model1 = noise.NoiseModel()
        noise_modelreal = noise_model1.from_dict(noise_model)
    print(n_qubits)
    backend_noise= AerSimulator(noise_model=noise_modelreal)
    exp = LocalReadoutError(list(range(n_qubits)))
    exp.analysis.set_options(plot=True)
    result = exp.run(backend_noise)
    #print(result)
    mitigator = result.analysis_results(0).value
    
    
    
    # VQE with CAFQA initialization
    shots = 8192
    loss_file = "vqe_loss.txt"
    params_file = "vqe_params.txt"
    vqe_energy, vqe_params = run_vqe(
        n_qubits=n_qubits,
        coeffs=coeffs,
        paulis=paulis,
        param_guess=np.array(cafqa_params)*np.pi/2, #注意,cafqa_params是由CAFQA部分函数计算所得的在Clifford空间中最佳的参数,本段程序是想要在此基础上求得在vqe空间中的最佳参数,并优化cafqa部分所算得的energy的结果
        budget=budget,
        shots=shots,
        mode="device_execution",
        backend=FakeMontreal(),
        save_dir=save_dir,
        loss_file=loss_file,
        params_file=params_file,
        mitigator=mitigator,
        noise_backend=backend_noise,
        **vqe_kwargs #MODIFIED,ac
    )
    with open(save_dir + result_file, "a") as res_file:
        res_file.write(f"VQE energy:\n{vqe_energy}\n")
        res_file.write(f"VQE params:\n{np.array(vqe_params)}\n\n")


if __name__ == "__main__":
    main()
