import csv
from timeit import default_timer as timer
from typing import Callable, List, Tuple

import numpy as np
from mitiq import zne
from qiskit import transpile
from skquant.opt import minimize as skquant_minimize
from scipy.optimize import minimize as scipy_minimize

from global_settings import Context
from vqe_utils import get_vqe_circuit, get_optimization_lib


def _get_group_exp(ctx: Context, count_general, pauli):
    pauli = pauli[::-1]
    expectation_val = 0
    num = pauli.count("I")
    num = ctx.num_qubits - num
    I_pos = []
    M_pos = []
    # 记录测量的位置
    for ind in range(len(pauli)):
        if pauli[ind] != "I":
            M_pos.append(ind)
        else:
            I_pos.append(ind)
    count = {}
    for i in range(2**num):
        ind = 0
        p_str = list("000000000000")
        bi_i = bin(i)[2:]
        bi_i = bi_i.rjust(num, "0")
        bi_i = list(bi_i)
        for j in M_pos:
            p_str[j] = bi_i[ind]
            ind += 1
        p_str = "".join(p_str)
        count[p_str] = 0
    for pau in count_general.keys():
        p_str_g = list(pau)
        for i in I_pos:
            p_str_g[i] = "0"
        p_str_g = "".join(p_str_g)
        count[p_str_g] += count_general[pau]
    expectation_val, _ = ctx.readout_mitigator.expectation_value(count)
    return expectation_val


def _count_to_exps(ctx, count, grouped_paulis):
    ans = []
    for pauli in grouped_paulis:
        if pauli == len(pauli) * "I":
            ans.append(1.0)
        else:
            mitigated = _get_group_exp(ctx, count, pauli)
            ans.append(mitigated)
    return ans


def get_pauli_expectations(
    ctx: Context,
    params: np.ndarray,
    grouped_paulis,
    fold: Callable,
) -> List[float]:
    # TODO: Pauli grouping with Qiskit.
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
    circuit = get_vqe_circuit(
        ctx.num_qubits, params, "ZZZZZZZZZZZZ", **ctx.vqe_kwargs
    )
    circuit = transpile(
        circuit,
        ctx.system_model,
        optimization_level=3,
        seed_transpiler=ctx.seed,
    )
    scaled_circuits = [fold(circuit, scale) for scale in ctx.zne_scale]
    counts = [
        ctx.noisy_simulator.run(circuit, shots=ctx.shots).result().get_counts()
        for circuit in scaled_circuits
    ]
    scaled_expectations = []

    for count in counts:
        exp = _count_to_exps(ctx, count, grouped_paulis)
        scaled_expectations.append(exp)
    zne_exps = [
        zne.RichardsonFactory.extrapolate(
            ctx.zne_scale, [exp[i] for exp in scaled_expectations]
        )
        for i in range(len(scaled_expectations[0]))
    ]

    return zne_exps


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
    # NOTE: We only compute the important part.
    expectations = get_pauli_expectations(
        ctx,
        params,
        grouped_paulis=ctx.hamiltonian.paulis,
        fold=ctx.zne_fold,
    )
    end = timer()
    energy = np.inner(ctx.hamiltonian.coefs, expectations)
    print(f"======Condition: {ctx.modification}======")
    print(f"Energy computed by VQE is {energy}, in {end - start}s.")

    with open(
        f"{ctx.save_dir}/energy_log.{ctx.modification}.csv",
        "a",
    ) as file:
        writer = csv.writer(file)
        writer.writerow([energy])

    with open(
        f"{ctx.save_dir}/params_log.{ctx.modification}.csv",
        "a",
    ) as file:
        writer = csv.writer(file)
        writer.writerow([energy])
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
    params_guess = np.array(ctx.prepared_cafqa_params) * np.pi / 2.0
    bounds = np.array(
        [
            [
                max(param - np.pi * ctx.bounds_shift[0], 0),
                min(param + np.pi * ctx.bounds_shift[1], np.pi * 1.5),
            ]
            for param in params_guess
        ]
    )
    if get_optimization_lib(ctx.optimization_method) == "scipy":
        vqe_result = scipy_minimize(
            lambda x: run_vqe_iter(ctx, x),
            x0=params_guess,
            bounds=bounds,
            method=ctx.optimization_method,
            options={"maxiter": ctx.budget},
        )
        energy_vqe = vqe_result.fun
        params_vqe = vqe_result.x

    else:
        vqe_result = skquant_minimize(
            lambda x: run_vqe_iter(ctx, params=x),
            x0=params_guess,
            bounds=bounds,
            budget=ctx.budget,
            method=ctx.optimization_method,
        )
        energy_vqe = vqe_result[0].optval
        params_vqe = vqe_result[0].optpar
    return energy_vqe, params_vqe


def solve(ctx: Context) -> None:
    energy, params = run_vqe(ctx)
    with open(
        f"{ctx.save_dir}/ans.{ctx.modification}.txt",
        "a",
    ) as file:
        file.write(f"energy:\n{energy}\n")
        file.write(f"params:\n{np.array(params)}\n")


def debug(ctx: Context) -> None:
    pass
