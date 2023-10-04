# this code is used to completely implement the ZNE mitigation method
# pay attention that we (for now) will only do ZNE on the important terms
# (pauli string with more weight)
from argparse import ArgumentParser
import csv
import numpy as np
from typing import Iterable, Union, List, Tuple
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import FakeMontreal
from timeit import default_timer as timer
from qiskit_experiments.library import (
    LocalReadoutError,
    CorrelatedReadoutError,
)
from skquant.opt import minimize
from qiskit.utils import algorithm_globals
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import pickle
from mitiq import zne
from global_settings import (
    BUDGET,
    SAVE_DIR,
    SEED,
    PREPARED_CAFQA_PARAMS,
    SEED_TRANSPILER,
)
from vqe_experiment import init_cafqa
from vqe_utils import efficientsu2_full, hartreefock, vqe_circuit

algorithm_globals.random_seed = SEED

# Load noise model and create simulator.
with open("NoiseModel/fakekolkata.pkl", "rb") as file:
    noise_model_dict = pickle.load(file)
noisy_simulator = AerSimulator(
    noise_model=NoiseModel.from_dict(noise_model_dict)
)
sys_backend = FakeMontreal()


# Read in and process the molecule infomation
bond_length = 1.0
# OH molecule string
atom_string = "O 0 0 0; H 0.45 -0.1525 -0.8454"
num_orbitals = 6
coefs, paulis, HF_bitstring = init_cafqa(atom_string, num_orbitals, charge=1)
n_qubits = len(paulis[0])
print(f"Require {n_qubits} qubits.")


exp = LocalReadoutError(list(range(n_qubits)))
exp.analysis.set_options(plot=True)
result = exp.run(noisy_simulator)
mitigator = result.analysis_results(0).value


# VQE with CAFQA initialization
# vqe params
parameters = PREPARED_CAFQA_PARAMS
init_func = hartreefock
ansatz_func = efficientsu2_full
ansatz_reps = 2


def run_circuit(circuit: QuantumCircuit, shots: int = 5000) -> float:
    """Returns the expectation value to be mitigated.

    Args:
        circuit: Circuit to run.
        shots: Times to execute the circuit to compute the expectation value.
    """
    transpiled_circuit = transpile(
        circuit,
        sys_backend,
        optimization_level=0,
        seed_transpiler=SEED_TRANSPILER,
    )
    counts = (
        noisy_simulator.run(transpiled_circuit, shots=shots)
        .result()
        .get_counts(0)
    )
    expectation_val, _ = mitigator.expectation_value(counts)
    return expectation_val


def find_important_terms(
    coefs: np.ndarray, threshold: float = 1.0
) -> List[int]:
    """Find important pauli strings which need
    to be applied ZNE to improve accuracy.

    Parameters
    ----------
    coefs : np.ndarry
        Coefficients for each Pauli strings.
    threshold: float, optional
        The threshold that can be considered as a large weight, by default 1.0.

    Returns
    -------
    List[int]
        A list of indexes representing important Pauli strings.
    """
    res = []
    for i, j in enumerate(coefs):
        if abs(j) > threshold:
            res.append(i)
    return res


def cut_paulis(
    important_terms: List[int], coefs: np.ndarray, paulis: np.ndarray
) -> Tuple[List, List, List, List]:
    """Divide Pauli strings into important and trivial parts.

    Parameters
    ----------
    important_terms : List[int]
        Indexes of important Pauli strings.
    coefs : np.ndarray
        Coefficients for each Pauli string.
    paulis : np.ndarray
        Pauli strings decomposed from Hamiltonian.

    Returns
    -------
    Tuple[List, List, List, List]
        Important Pauli strings with coefficients and
        trivial Pauli strings with coefficients.
    """
    important_coefs = [coefs[i] for i in important_terms]
    important_paulis = [paulis[i] for i in important_terms]
    trivial_coefs = list(coefs)
    trivial_paulis = list(paulis)
    for ind in important_terms[::-1]:
        trivial_coefs.pop(ind)
        trivial_paulis.pop(ind)
    return important_coefs, important_paulis, trivial_coefs, trivial_paulis


def get_pauli_expectations(
    n_qubits,
    params,
    paulis,
    **kwargs,
) -> List:
    pauli_expectations = []
    for pauli in paulis:
        if pauli == len(pauli) * "I":
            pauli_expectations.append(1.0)
        else:
            circuit = vqe_circuit(n_qubits, params, pauli, **kwargs)
            circuit = transpile(
                circuit,
                sys_backend,
                optimization_level=2,
                seed_transpiler=SEED_TRANSPILER,
            )
            mitigated = zne.execute_with_zne(circuit, run_circuit)
            pauli_expectations.append(mitigated)
    return pauli_expectations


important_terms = find_important_terms(coefs, threshold=1)
print("important_terms: ", important_terms)
imp_coefs, imp_paulis, trivial_coefs, trivial_paulis = cut_paulis(
    important_terms, coefs, paulis
)
print("imp_coefs: ", imp_coefs, "\n imp_paulis", imp_paulis)


def run_vqe_iter(
    n_qubits,
    parameters,
    loss_filename=None,
    params_filename=None,
    **kwargs,
):
    """
    Compute the VQE loss/energy.
    n_qubits (Int): Number of qubits in circuit.
    parameters (Iterable[Float]): VQE parameters.ä
    coefs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    loss_filename (String): Path to save file for VQE loss/energy.
    params_filename (String): Path to save file for VQE parameters.
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.
    Returns:
    (Float) VQE energy.
    """
    start = timer()
    pauli_expectations = get_pauli_expectations(
        n_qubits,
        parameters,
        paulis=imp_paulis,
        **kwargs,
    )
    important_energy = np.inner(imp_coefs, pauli_expectations)

    print("imp_energy: ", important_energy)

    # trivial_energy=np.inner(trivial_coefs,trivial_expectations)
    trivial_energy = 0
    loss = important_energy + trivial_energy
    end = timer()
    print(f"Loss computed by VQE_ZNE is {loss}, in {end - start} s.")

    if loss_filename is not None:
        with open(loss_filename, "a") as file:
            writer = csv.writer(file)
            writer.writerow([loss])

    if params_filename is not None and parameters is not None:
        with open(params_filename, "a") as file:
            writer = csv.writer(file)
            writer.writerow(parameters)
    return loss


def run_vqe(
    n_qubits: int,
    param_guess: Union[Iterable[float], np.ndarray],
    budget: int,
    save_dir: str,
    loss_file: str,
    params_file: str,
    **kwargs,
):
    """
    Run VQE instance. Uses skquant for optimization.
    n_qubits (Int): Number of qubits in circuit.
    param_guess (Iterable[Float]): Initial guess for VQE parameters.
    budget (Int): Max number of optimization iterations.
    save_dir (String): Save directory.
    loss_file (String): Name of save file for VQE loss/energy.
    params_file (String): Name of save file for VQE parameters.
    kwargs (Dict): Dictionary with additional keyword arguments for vqe() call.

    Returns:
    Tuple of energy estimate and optimized parameters.
    """
    # check right number of parameters given
    _, num_params = efficientsu2_full(n_qubits, kwargs.get("ansatz_reps", 2))
    if len(param_guess) == 0:
        param_guess = [0] * num_params
    assert (
        len(param_guess) == num_params
    ), f"""The number of vales given doesn't match the number of parameters.
    {len(param_guess)} are given but there are {num_params} parameters."""

    vqe_result = minimize(
        lambda x: run_vqe_iter(
            n_qubits=n_qubits,
            parameters=x,
            loss_filename=save_dir + "/" + loss_file,
            params_filename=save_dir + "/" + params_file,
            **vqe_kwargs,
        ),
        x0=np.array(param_guess),
        bounds=np.array([[0, np.pi * 2]] * num_params),
        budget=budget,
        method="imfil",
    )
    energy_vqe = vqe_result[0].optval
    params_vqe = vqe_result[0].optpar
    return energy_vqe, params_vqe


print("without trivial terms. zne and Local rem")
loss_file = "vqe_zne_rem_loss2.txt"
params_file = "vqe_zne_rem_params2.txt"
vqe_energy, vqe_params = run_vqe(
    n_qubits=n_qubits,
    param_guess=np.array(PREPARED_CAFQA_PARAMS) * np.pi / 2,
    budget=BUDGET,
    save_dir=SAVE_DIR,
    loss_file=loss_file,
    params_file=params_file,
    **vqe_kwargs,  # MODIFIED,ac
)
print("vqe_energy, vqe_params", vqe_energy, vqe_params)
with open(SAVE_DIR + RESULT_FILE, "a") as res_file:
    res_file.write(f"VQE energy:\n{vqe_energy}\n")
    res_file.write(f"VQE params:\n{np.array(vqe_params)}\n\n")
