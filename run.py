import csv
from timeit import default_timer as timer
from typing import Callable, List, Tuple

import numpy as np
from mitiq import zne
from qiskit import QuantumCircuit, transpile
from skquant.opt import minimize

from global_settings import Context
from vqe_utils import get_param_num, get_vqe_circuit


def get_circuit_executor(ctx: Context) -> Callable:
    """Construct the circuit execution function.

    Parameters
    ----------
    ctx : Context
        Context with configuration and meta information of the application.

    Returns
    -------
    Callable
        The circuit execution function.
    """

    def run_circuit(circuit: QuantumCircuit) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
        """
        # HACK: Waste transpiling.
        transpiled_circuit = transpile(
            circuit,
            ctx.system_model,
            optimization_level=0,
            seed_transpiler=ctx.seed,
        )
        counts = (
            ctx.noisy_simulator.run(transpiled_circuit, shots=ctx.shots)
            .result()
            .get_counts(0)
        )
        expectation_val, _ = ctx.readout_mitigator.expectation_value(counts)
        return expectation_val

    return run_circuit


def get_pauli_expectations(
    ctx: Context,
    paulis: List[str],
    params: np.ndarray,
) -> List[float]:
    # TODO: Pauli grouping.
    """Compute the expectations of each Pauli string.

    Parameters
    ----------
    ctx : Context
        Context with configuration and meta information of the application.
    paulis : List[str]
        Pauli strings constituting the Hamiltonian.
    params : np.ndarray
        Parameters in quantum circuit of VQE.

    Returns
    -------
    List
        Returns a list, where each element represents the expected value
        of the measurement with the corresponding Pauli string.
    """
    pauli_expectations = []
    for pauli in paulis:
        if pauli == len(pauli) * "I":
            # NOTE: Result of identity is always 1 as the unit vector.
            pauli_expectations.append(1.0)
        else:
            # TODO: Use a single instance and assign parameters.
            circuit = get_vqe_circuit(
                ctx.num_qubits, params, pauli, **ctx.vqe_kwargs
            )
            circuit = transpile(
                circuit,
                ctx.system_model,
                optimization_level=2,
                seed_transpiler=ctx.seed,
            )
            mitigated = zne.execute_with_zne(
                circuit, get_circuit_executor(ctx)
            )
            pauli_expectations.append(mitigated)
    return pauli_expectations


def run_vqe_iter(
    ctx: Context,
    params: np.ndarray,
) -> float:
    """Compute the ground state energy in each iteration.

    Parameters
    ----------
    ctx : Context
        Context with configuration and meta information of the application.
    params : np.ndarray
        Parameters in quantum circuit of VQE.

    Returns
    -------
    float
        The computed energy in each iteration.
    """
    start = timer()
    pauli_expectations = get_pauli_expectations(
        ctx=ctx, paulis=ctx.hamiltonian.paulis, params=params
    )
    # NOTE: We only compute the important part when execute locally.
    energy = np.inner(
        ctx.hamiltonian.coefs, pauli_expectations
    )
    end = timer()
    print(f"Energy computed by VQE is {energy}, in {end - start}s.")

    with open(f"{ctx.save_dir}/energy_log.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow([energy])

    with open(f"{ctx.save_dir}/params_log.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow(params)
    return energy


def run_vqe(
    ctx: Context,
) -> Tuple[float, np.ndarray]:
    """Run VQE instance to calculate the ground
    state energy of the hydroxyl cation.

    Parameters
    ----------
    ctx : Context
        Context with configuration and meta information of the application.

    Returns
    -------
    Tuple[float, np.ndarray]
        Return the caculated ground state energy of ·OH and
        the best parameters during VQE iterations.
    """

    # HINT: Check number of parameters given.
    num_params = get_param_num(
        ctx.num_qubits, ctx.vqe_kwargs.get("ansatz_reps", 2)
    )
    param_guess = ctx.prepared_cafqa_params
    if len(param_guess) == 0:
        param_guess = [0] * num_params
    assert (
        len(param_guess) == num_params
    ), f"""The number of vales given doesn't match the number of parameters.
    {len(param_guess)} are given but there are {num_params} parameters."""

    vqe_result = minimize(
        lambda x: run_vqe_iter(ctx, params=x),
        x0=np.array(param_guess) * np.pi / 2.0,
        bounds=np.array([[0, np.pi * 2]] * num_params),
        budget=ctx.budget,
        method=ctx.optimization_method,
    )
    energy_vqe = vqe_result[0].optval
    params_vqe = vqe_result[0].optpar
    return energy_vqe, params_vqe


def solve(ctx: Context) -> None:
    energy, params = run_vqe(ctx)
    with open(f"{ctx.save_dir}/ans.txt", "a") as file:
        file.write(f"energy:\n{energy}\n")
        file.write(f"params:\n{np.array(params)}\n")
