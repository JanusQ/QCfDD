from argparse import Namespace
import pickle
from typing import Any, Dict, List, NamedTuple, Union
import numpy as np
from qiskit.utils import algorithm_globals
import random
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_experiments.library import (
    LocalReadoutError,
    CorrelatedReadoutError,
)
from qiskit.result import LocalReadoutMitigator, CorrelatedReadoutMitigator
from qiskit.providers.fake_provider import FakeKolkata, FakeCairo, FakeMontreal
from vqe_experiment import init_cafqa

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
        ("save_dir", str),
        ("seed", int),
        ("shots", int),
        (
            "hamiltonian",
            NamedTuple(
                "Hamiltonian", [("coefs", np.ndarray), ("paulis", np.ndarray)]
            ),
        ),
        ("vqe_kwargs", Dict[str, Any]),
        ("prepared_cafqa_params", List[int]),
    ],
)


def _load_noise(noise_name: str) -> AerSimulator:
    """Load noise model from .pkl.

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
    coefs, paulis, hf_bitstring = init_cafqa(
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


def get_context(args: Namespace) -> Context:
    noisy_simulator = _load_noise(args.noise)
    (
        coefs,
        paulis,
        hf_bitstring,
        num_qubits,
    ) = _get_quantum_info()
    _seed_everything(args.seed)
    return Context(
        readout_mitigator=_get_readout_mitigator(
            args.readout_mitigator, num_qubits, noisy_simulator
        ),
        num_qubits=num_qubits,
        noisy_simulator=noisy_simulator,
        optimization_method=args.optimizer,
        system_model=FakeMontreal(),
        budget=args.budget,
        save_dir=args.save,
        seed=args.seed,
        shots=args.shots,
        hamiltonian=(coefs, paulis),
        vqe_kwargs={
            "ansatz_reps": 2,
            "init_last": False,
            "HF_bitstring": hf_bitstring,
            "readout_error_mitigation": True,
        },
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
