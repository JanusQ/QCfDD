import csv
from timeit import default_timer as timer
from typing import Callable, List, Tuple

import numpy as np
from mitiq import zne
from qiskit import QuantumCircuit, transpile
from skquant.opt import minimize

from global_settings import Context
from vqe_utils import Estimator, get_ansatz, get_vqe_circuit
from qiskit.quantum_info import SparsePauliOp


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
    # NOTE: 传进来一个带参数的ansatz，然后利用ansatz计算每个Pauli的期望，
    # NOTE: 即 `estimator.run(ansatz,pauils)`，
    # NOTE: paulis可用SparsePauliOp这个类（包含字符串和系数）。
    # NOTE: 这就是Estimator的用处，利用Estimator可实现Pauli分组。
    # NOTE: 在Estimator中添加读取噪声抑制方法。

    # HINT: 本函数传入需要计算的Paulis，返回一个用于ZNE的函数：
    # HINT: compute_expectation(ansatz（确定参数）:QuantumCircuit)->float
    # HINT:   return estimator.run([ansatz]*num_paulis, pauilis,params)
    # HINT: ZNE的结果直接作为能量值，用经典迭代器优化参数params。

    # TODO: 重点是要在Estimator中添加读取噪声抑制方法。

    def compute_expectation(parameterized_ansatz: QuantumCircuit) -> float:
        """Returns the expectation value to be mitigated.

        Args:
            circuit: Circuit to run.
        """
        return (
            Estimator(
                run_options={"seed": ctx.seed, "shots": ctx.shots},
                transpile_options={
                    "optimization_level": 2,
                    "backend": ctx.system_model,
                    "seed_transpiler": ctx.seed,
                },
                backend=ctx.noisy_simulator,
                mitigator=ctx.readout_mitigator,
            )
            .run(
                parameterized_ansatz,
                SparsePauliOp.from_list(
                    [
                        (pauli, coef)
                        for coef, pauli in zip(
                            ctx.hamiltonian.coefs, ctx.hamiltonian.paulis
                        )
                    ]
                ),
            )
            .result()
            .values[0]
        )

    return compute_expectation


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

    # NOTE: We only compute the important part when execute locally.
    energy = zne.execute_with_zne(
        ctx.ansatz.bind_parameters(params),
        lambda circuit: np.mean(
            [
                Estimator(
                    run_options={"seed": ctx.seed, "shots": ctx.shots},
                    transpile_options={
                        "optimization_level": 2,
                        "backend": ctx.system_model,
                        "seed_transpiler": ctx.seed,
                    },
                    mitigator=ctx.readout_mitigator,
                    backend=ctx.noisy_simulator,
                )
                .run(
                    circuit,
                    SparsePauliOp.from_list(
                        [
                            (pauli, coef)
                            for coef, pauli in zip(
                                ctx.hamiltonian.coefs, ctx.hamiltonian.paulis
                            )
                        ]
                    ),
                )
                .result()
                .values[0]
                for _ in range(4)
            ]
        ),
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

    vqe_result = minimize(
        lambda x: run_vqe_iter(ctx, params=x),
        x0=np.array(ctx.prepared_cafqa_params) * np.pi / 2.0,
        bounds=np.array([[0, np.pi * 2]] * len(ctx.prepared_cafqa_params)),
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


def debug(ctx: Context) -> None:
    print(
        get_ansatz(
            ctx.num_qubits, ctx.system_model, **ctx.vqe_kwargs
        ).num_parameters
    )
