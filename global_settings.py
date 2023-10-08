import pickle
import random
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Union, NamedTuple

import numpy as np
from mitiq.zne.scaling import fold_gates_at_random, fold_global
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import FakeCairo, FakeKolkata, FakeMontreal
from qiskit.result import CorrelatedReadoutMitigator, LocalReadoutMitigator
from qiskit.utils import algorithm_globals
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_experiments.library import (
    CorrelatedReadoutError,
    LocalReadoutError,
)

from vqe_utils import init_molecule

Hamiltonian = NamedTuple(
    "Hamiltonian",
    [
        ("coefs", list[float]),
        ("paulis", list[str]),
    ],
)
Context = NamedTuple(
    "Context",
    [
        (
            "readout_mitigator",
            Union[LocalReadoutMitigator, CorrelatedReadoutMitigator],
        ),
        ("num_qubits", int),
        ("noisy_simulator", AerSimulator),
        ("optimization_method", str),
        ("system_model", Union[FakeCairo, FakeKolkata, FakeMontreal]),
        ("budget", int),
        ("save_dir", Path),
        ("seed", int),
        ("shots", int),
        ("hamiltonian", Hamiltonian),
        ("vqe_kwargs", dict[str, Any]),
        ("prepared_cafqa_params", list[int]),
        ("zne_scale", list[float]),
        ("zne_fold", Callable[[QuantumCircuit, list[float]]]),
        ("bounds_shift", tuple[float, float]),
    ],
)


def _load_noise(noise_name: str) -> AerSimulator:
    """Load noise model from .pkl files.

    Parameters
    ----------
    noise_name : str
        The backend name of noise.

    Returns
    -------
    AerSimulator
        Noisy simulator.
    """
    with open(f"NoiseModel/fake{noise_name}.pkl", "rb") as file:
        noise_dict = pickle.load(file)
    return AerSimulator(noise_model=NoiseModel.from_dict(noise_dict))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    algorithm_globals.random_seed = seed


def _get_quantum_info():
    coefs, paulis, hf_bitstring = init_molecule(
        "O 0 0 0; H 0.45 -0.1525 -0.8454", 6, 1
    )
    num_qubits = len(paulis[0])
    return coefs, paulis, hf_bitstring, num_qubits


def _get_readout_mitigator(
    mitigator_type: str, num_qubits: int, noisy_simulator: AerSimulator
) -> Union[LocalReadoutMitigator, CorrelatedReadoutMitigator]:
    assert mitigator_type == "local" or mitigator_type == "correlated"
    experiment = (
        LocalReadoutError(list(range(num_qubits)), noisy_simulator)
        if mitigator_type == "local"
        else CorrelatedReadoutError(list(range(num_qubits)), noisy_simulator)
    )
    return experiment.run(noisy_simulator).analysis_results(0).value


def _find_important_terms(
    coefs: np.ndarray, threshold: float = 1.0
) -> list[int]:
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


def _simplify_paulis(
    important_terms: list[int], coefs: np.ndarray, paulis: np.ndarray
) -> tuple[list[float], list[str]]:
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
        Important Pauli strings with coefficients.
    """
    important_coefs = [coefs[i] for i in important_terms]
    important_paulis = [paulis[i] for i in important_terms]
    return important_coefs, important_paulis


def get_context(args: Namespace) -> Context:
    noisy_simulator = _load_noise(args.noise)
    (
        coefs,
        paulis,
        HF_bitstring,
        num_qubits,
    ) = _get_quantum_info()
    coefs, paulis = _simplify_paulis(
        _find_important_terms(coefs, args.threshold), coefs, paulis
    )
    _seed_everything(args.seed)
    save_dir = Path(args.save)
    save_dir.mkdir(exist_ok=True)
    vqe_kwargs = {
        "ansatz_reps": 2,
        "init_last": False,
        "HF_bitstring": HF_bitstring,
    }
    system_model = FakeMontreal()
    return Context(
        readout_mitigator=_get_readout_mitigator(
            args.readout_mitigator, num_qubits, noisy_simulator
        ),
        num_qubits=num_qubits,
        noisy_simulator=noisy_simulator,
        optimization_method=args.optimizer,
        system_model=system_model,
        budget=args.budget,
        save_dir=save_dir,
        seed=args.seed,
        shots=args.shots,
        hamiltonian=Hamiltonian(coefs=coefs, paulis=paulis),
        vqe_kwargs=vqe_kwargs,
        zne_scale=args.zne_scale,
        zne_fold=fold_global
        if args.zne_fold == "global"
        else fold_gates_at_random,
        bounds_shift=tuple(args.bounds_shift),
        prepared_cafqa_params=[
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            2,
            0,
            3,
            0,
            3,
            1,
            0,
            1,
            0,
            2,
            0,
            1,
            0,
            0,
            2,
            2,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            2,
            0,
            0,
            0,
            0,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            2,
            0,
            0,
            0,
            0,
            0,
            3,
            2,
            0,
            0,
            0,
            1,
        ],
    )
