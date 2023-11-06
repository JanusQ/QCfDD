from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from copy import copy
from timeit import default_timer as timer
from typing import Tuple, Union
from warnings import warn

import hypermapper
import numpy as np
import stim
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterExpression
from qiskit.circuit.library import EfficientSU2
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator, EstimatorResult
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.primitives.utils import (
    _circuit_key,
    _observable_key,
    init_observable,
)
from qiskit.providers import Options
from qiskit.quantum_info import Pauli, PauliList
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.result import CorrelatedReadoutMitigator, LocalReadoutMitigator
from qiskit.result.models import ExperimentResult
from qiskit_aer import AerError, AerSimulator
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.units import DistanceUnit


def get_optimization_lib(method: str) -> str:
    if method in [
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "SLSQP",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    ]:
        return "scipy"
    elif method in ["imfil", "snobfit", "nomad", "bobyqa"]:
        return "skquant"


def get_param_num(num_qubits: int, ansatz_reps: int) -> int:
    """Get the number of parameters in ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits needed by VQE.
    ansatz_reps : int
        Repetition of ansatz layer.

    Returns
    -------
    int
        The number of parameters in ansatz.
    """
    _, num_params = _efficientsu2_full(num_qubits, ansatz_reps)
    return num_params


def init_molecule(
    atom_string: str, new_num_orbitals: int = None, charge: int = 0
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Compute Hamiltonian for molecule in qubit encoding using Qiskit Nature.

    Parameters
    ----------
    atom_string : str
        String to describe molecule, passed to PySCFDriver.
    new_num_orbitals : int, optional
        Number of orbitals in active space, by default None.
        If None, use default result from PySCFDriver.
    charge : int, optional
        The charge of the molecule, by default 0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, str]
        Coefficients for each Pauli string, Pauli strings
        and Hartree-Fock bitstring.
    """

    converter = QubitConverter(JordanWignerMapper(), two_qubit_reduction=True)
    driver = PySCFDriver(
        atom=atom_string,
        basis="sto3g",
        charge=charge,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()
    if new_num_orbitals is not None:
        num_electrons = (problem.num_alpha, problem.num_beta)
        transformer = ActiveSpaceTransformer(num_electrons, new_num_orbitals)
        problem = transformer.transform(problem)
    ferOp = problem.hamiltonian.second_q_op()
    qubitOp = converter.convert(ferOp, problem.num_particles)
    initial_state = HartreeFock(
        problem.num_spatial_orbitals, problem.num_particles, converter
    )
    bitstring = "".join(["1" if bit else "0" for bit in initial_state._bitstr])
    # HINT: Need to reverse order because of qiskit endianness.
    paulis = [x[::-1] for x in qubitOp.primitive.paulis.to_labels()]
    # HINT: Add the shift as extra "I" pauli.
    paulis.append("I" * len(paulis[0]))
    paulis = np.array(paulis)
    coeffs = list(qubitOp.primitive.coeffs)
    # HINT: Add the shift (nuclear repulsion).
    coeffs.append(problem.nuclear_repulsion_energy)
    coeffs = np.array(coeffs).real
    return coeffs, paulis, bitstring


def _append_by_hartreefock(
    circuit: QuantumCircuit, HF_bitstring: str = None
) -> None:
    """Append the EfficientSU2 ansatz to input circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The initial circuit to be appended.
    HF_bitstring : str, optional
        Bitstring to initialize to, e.g. "01101" -> |01101>, by default None.
    """
    if HF_bitstring is None:
        return
    for i in range(len(HF_bitstring)):
        if HF_bitstring[i] == "1":
            circuit.x(i)


def _efficientsu2_full(num_qubits, **kwargs):
    ansatz = EfficientSU2(
        num_qubits=num_qubits,
        entanglement=kwargs.get("entanglement", "full"),
        reps=kwargs.get("ansatz_reps", 2),
    )
    num_params_ansatz = len(ansatz.parameters)
    ansatz = ansatz.decompose()
    return ansatz, num_params_ansatz


def _add_ansatz(circuit, parameters, **kwargs):
    num_qubits = circuit.num_qubits
    ansatz, _ = _efficientsu2_full(num_qubits, **kwargs)
    if parameters is not None:
        ansatz.assign_parameters(parameters=parameters, inplace=True)
    circuit.compose(ansatz, inplace=True)


def get_ansatz(num_qubits, backend, **kwargs) -> QuantumCircuit:
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    circuit = QuantumCircuit(qr, cr)
    init_last = kwargs.get("init_last", False)
    HF_bitstring = kwargs.get("HF_bitstring", None)

    if not init_last:
        _append_by_hartreefock(circuit, HF_bitstring)
    # HINT: Append the circuit with the state preparation ansatz.
    circuit.compose(
        EfficientSU2(
            num_qubits=num_qubits,
            reps=kwargs.get("ansatz_reps", 1),
            entanglement="full",
        ),
        inplace=True,
    )

    if init_last:
        _append_by_hartreefock(circuit, HF_bitstring)

    return circuit.decompose()


def get_vqe_circuit(
    num_qubits: int,
    params: np.ndarray,
    pauli: str,
    **kwargs,
) -> QuantumCircuit:
    """Construct a single VQE circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits of the VQE circuit.
    params : np.ndarry
        Parameters in VQE circuit.
    pauli : str
        Pauli string used to determine the gate for measurement.

    Returns
    -------
    QuantumCircuit
        VQE circuit.
    """
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(num_qubits)
    circuit = QuantumCircuit(qr, cr)
    init_last = kwargs.get("init_last", False)
    HF_bitstring = kwargs.get("HF_bitstring", None)

    if not init_last:
        _append_by_hartreefock(circuit, HF_bitstring)
    # HINT: Append the circuit with the state preparation ansatz.
    _add_ansatz(circuit, params, **kwargs)
    if init_last:
        _append_by_hartreefock(circuit, HF_bitstring)

    # HINT: Add the measurement operations.
    for i, el in enumerate(pauli):
        if el == "I":
            # HINT: No measurement for identity.
            continue
        elif el == "Z":
            circuit.measure(qr[i], cr[i])
        elif el == "X":
            circuit.h(qr[i])
            circuit.measure(qr[i], cr[i])
        elif el == "Y":
            circuit.sx(qr[i])
            circuit.s(qr[i])
            circuit.measure(qr[i], cr[i])

    return circuit


def _transform_to_allowed_gates(circuit):
    """
    circuit (QuantumCircuit): Circuit with only Clifford gates
    (1q rotations Ry, Rz must be k*pi/2).
    kwargs (Dict): All the arguments that need to be passed
    on to the next function calls.

    Returns:
    (QuantumCircuit) Logically equivalent circuit but with
    gates in required format (no Ry, Rz gates; only S, Sdg, H, X, Z).
    """
    dag = circuit_to_dag(circuit)
    threshold = 1e-3
    # we will substitute nodes inplace
    for node in dag.op_nodes():
        if node.name == "ry":
            angle = float(node.op.params[0])
            # substitute gates
            if abs(angle - 0) < threshold:
                dag.remove_op_node(node)
            elif abs(angle - np.pi / 2) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.sdg(0)
                qc_loc.sx(0)
                qc_loc.s(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
            elif abs(angle - np.pi) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.y(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
            elif abs(angle - 1.5 * np.pi) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.sdg(0)
                qc_loc.sxdg(0)
                qc_loc.s(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
        elif node.name == "rz":
            angle = float(node.op.params[0])
            # substitute gates
            if abs(angle - 0) < threshold:
                dag.remove_op_node(node)
            elif abs(angle - np.pi / 2) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.s(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
            elif abs(angle - np.pi) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.z(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
            elif abs(angle - 1.5 * np.pi) < threshold:
                qc_loc = QuantumCircuit(1)
                qc_loc.sdg(0)
                qc_loc_instr = qc_loc.to_instruction()
                dag.substitute_node(node, qc_loc_instr, inplace=True)
        elif node.name == "x":
            qc_loc = QuantumCircuit(1)
            qc_loc.x(0)
            qc_loc_instr = qc_loc.to_instruction()
            dag.substitute_node(node, qc_loc_instr, inplace=True)
    return dag_to_circuit(dag).decompose()


def _qiskit_to_stim(circuit):
    """
    Transform Qiskit QuantumCircuit into stim circuit.
    circuit (QuantumCircuit): Clifford-only circuit.

    Returns:
    (stim._stim_sse2.Circuit) stim circuit.
    """
    assert isinstance(
        circuit, QuantumCircuit
    ), "Circuit is not a Qiskit QuantumCircuit."
    allowed_gates = [
        "X",
        "Y",
        "Z",
        "H",
        "CX",
        "S",
        "S_DAG",
        "SQRT_X",
        "SQRT_X_DAG",
    ]
    stim_circ = stim.Circuit()
    # make sure right number of qubits in stim circ
    for i in range(circuit.num_qubits):
        stim_circ.append("I", [i])
    for instruction in circuit:
        gate_lbl = instruction.operation.name.upper()
        if gate_lbl == "BARRIER":
            continue
        elif gate_lbl == "SDG":
            gate_lbl = "S_DAG"
        elif gate_lbl == "SX":
            gate_lbl = "SQRT_X"
        elif gate_lbl == "SXDG":
            gate_lbl = "SQRT_X_DAG"
        assert gate_lbl in allowed_gates, f"Invalid gate {gate_lbl}."
        qubit_idc = [qb.index for qb in instruction.qubits]
        stim_circ.append(gate_lbl, qubit_idc)
    return stim_circ


def _vqe_cafqa_stim(
    inputs,
    num_qubits,
    coeffs,
    paulis,
    init_func=_append_by_hartreefock,
    ansatz_func=_efficientsu2_full,
    ansatz_reps=1,
    init_last=False,
    loss_filename=None,
    params_filename=None,
    **kwargs,
):
    """
    Compute the CAFQA VQE loss/energy using stim.
    inputs (Dict): CAFQA VQE parameters (values in 0...3)
    as passed by hypermapper, e.g.: {"x0": 1, "x1": 0, "x2": 0, "x3": 2}
    num_qubits (Int): Number of qubits in circuit.
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    paulis (Iterable[String]): Corresponding Pauli
    strings in Hamiltonian (same order as coeffs).
    initialization (Function): Takes QuantumCircuit
    and applies state initialization inplace.
    parametrization (Function): Takes QuantumCircuit
    and applies ansatz inplace.
    init_last (Bool): Whether initialization should come
    after (True) or before (False) ansatz.
    loss_filename (String): Path to save file for VQE loss/energy.
    params_filename (String): Path to save file for VQE parameters.
    kwargs (Dict): All the arguments that need to
    be passed on to the next function calls.

    Returns:
    (Float) CAFQA VQE energy.
    """
    start = timer()
    parameters = []
    # take the hypermapper parameters and convert them to vqe parameters
    for key in inputs:
        parameters.append(inputs[key] * (np.pi / 2))

    vqe_qc = QuantumCircuit(num_qubits)
    if not init_last:
        init_func(vqe_qc, **kwargs)
    _add_ansatz(vqe_qc, ansatz_func, parameters, ansatz_reps, **kwargs)
    if init_last:
        init_func(vqe_qc, **kwargs)
    vqe_qc_trans = _transform_to_allowed_gates(vqe_qc)
    stim_qc = _qiskit_to_stim(vqe_qc_trans)
    sim = stim.TableauSimulator()
    sim.do_circuit(stim_qc)
    pauli_expect = [
        sim.peek_observable_expectation(stim.PauliString(p)) for p in paulis
    ]
    loss = np.dot(coeffs, pauli_expect)
    end = timer()
    print(f"Loss computed by CAFQA VQE is {loss}, in {end - start} s.")

    if loss_filename is not None:
        with open(loss_filename, "a") as file:
            writer = csv.writer(file)
            writer.writerow([loss])

    if params_filename is not None and parameters is not None:
        with open(params_filename, "a") as file:
            writer = csv.writer(file)
            writer.writerow(parameters)
    return loss


def run_cafqa(
    num_qubits,
    coeffs,
    paulis,
    param_guess,
    budget,
    shots,
    mode,
    backend,
    save_dir,
    loss_file,
    params_file,
    vqe_kwargs,
):
    """
    Run CAFQA VQE instance. Uses stim for fast Clifford circuit
    simulation and hypermapper for discrete optimization.
    num_qubits (Int): Number of qubits in circuit.
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    paulis (Iterable[String]): Corresponding Pauli strings
    in Hamiltonian (same order as coeffs).
    param_guess (Iterable[0...3]): Initial guess for
    CAFQA VQE parameters, which are factors for pi/2.
    E.g. param_guess = [1,0,0,2,3,1] for 6-parameter VQE
    with real parameters [pi/2,0,0,pi,3pi/2,pi/2].
    budget (Int): Max number of optimization iterations.
    shots (Int): --Not relevant here--.
    mode (String): --Not relevant here--.
    backend (IBM backend): --Not relevant here--.
    save_dir (String): Save directory.
    loss_file (String): Name of save file for VQE loss/energy.
    params_file (String): Name of save file for VQE parameters.
    vqe_kwargs (Dict): Dictionary with additional keyword
    arguments for vqe_cafqa_stim() call.

    Returns:
    Tuple of energy estimate and optimized CAFQA parameters.
    """
    # check right number of parameters given
    _, num_params = _efficientsu2_full(num_qubits, vqe_kwargs["ansatz_reps"])
    if len(param_guess) == 0:
        param_guess = [0] * num_params
    assert (
        len(param_guess) == num_params
    ), f"""Number of parameters given ({len(param_guess)})
    does not match ansatz ({num_params})."""

    hypermapper_config_path = save_dir + "/hypermapper_config.json"
    config = {}
    config["application_name"] = "cafqa_optimization"
    config["optimization_objectives"] = ["value"]
    number_of_RS = budget // 1  # 向下取整除
    config["design_of_experiment"] = {}
    config["design_of_experiment"]["number_of_samples"] = number_of_RS  # NMS
    config["optimization_iterations"] = budget
    config["models"] = {}
    config["models"]["model"] = "random_forest"
    config["input_parameters"] = {}
    config["print_best"] = True
    config["print_posterior_best"] = True
    for i in range(num_params):
        x = {}
        x["parameter_type"] = "ordinal"  # 序数
        x["values"] = [0, 1, 2, 3]
        x["parameter_default"] = param_guess[i]
        config["input_parameters"]["x" + str(i)] = x
    config["log_file"] = save_dir + "/hypermapper_log.log"
    config["output_data_file"] = save_dir + "/hypermapper_output.csv"
    with open(hypermapper_config_path, "w") as config_file:
        json.dump(
            config, config_file, indent=4
        )  # save the dict config into hypermapper_config_path

    hypermapper.optimizer.optimize(
        hypermapper_config_path,
        lambda x: _vqe_cafqa_stim(
            inputs=x,
            num_qubits=num_qubits,
            loss_filename=save_dir + "/" + loss_file,
            params_filename=save_dir + "/" + params_file,
            paulis=paulis,
            coeffs=coeffs,
            shots=shots,
            backend=backend,
            mode=mode,
            **vqe_kwargs,
        ),
    )  # black_box_function
    # sys.stdout = stdout #modified,ac,10.2

    energy_cafqa = np.inf  # infinite
    x_cafqa = None
    with open(
        config["log_file"]
    ) as f:  # read the output of function 'minimize'
        lines = f.readlines()
        counter = 0
        for idx, line in enumerate(
            lines[::-1]
        ):  # [::-1]： it stands for reverse order
            # enumerate is useful for obtaining an indexed list:
            # (0, seq[0]), (1, seq[1]), (2, seq[2]), ...
            if (
                line[:16] == "Best point found"
                or line[:29] == "Minimum of the posterior mean"
            ):  # line[:16] the first 16 items,ac
                """格式大致如下:

                Best point found:
                x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,value
                0,0,0,0,0,0,0,0,0,0,0,0,-1.0661086493179344
                """
                counter += 1
                parts = lines[-1 - idx + 2].split(",")
                # .split: Return a list of the substrings in the string,
                # using sep as the separator string.
                # partition the contents of lines[-1-idx+2] into list (parts),
                # with the sign of partition as ','
                energy = float(parts[-1])
                if energy < energy_cafqa:
                    energy_cafqa = energy
                    x_cafqa = [
                        int(y) for y in parts[:-1]
                    ]  # record the params of x_i
            if counter == 2:
                break
    return energy_cafqa, x_cafqa


class Estimator(BaseEstimator):
    """
    Aer implmentation of Estimator.

    :Run Options:
        - **shots** (None or int) --
          The number of shots. If None and approximation is True, it calculates the exact
          expectation values. Otherwise, it calculates expectation values with sampling.

        - **seed** (int) --
          Set a fixed seed for the sampling.

    .. note::
        Precedence of seeding for ``seed_simulator`` is as follows:

        1. ``seed_simulator`` in runtime (i.e. in :meth:`run`)
        2. ``seed`` in runtime (i.e. in :meth:`run`)
        3. ``seed_simulator`` of ``backend_options``.
        4. default.

        ``seed`` is also used for sampling from a normal distribution when approximation is True.

        When combined with the approximation option, we get the expectation values as follows:

        * shots is None and approximation=False: Return an expectation value with sampling-noise w/
          warning.
        * shots is int and approximation=False: Return an expectation value with sampling-noise.
        * shots is None and approximation=True: Return an exact expectation value.
        * shots is int and approximation=True: Return expectation value with sampling-noise using a
          normal distribution approximation.
    """

    def __init__(
        self,
        *,
        backend: AerSimulator,
        mitigator: Union[CorrelatedReadoutMitigator, LocalReadoutMitigator],
        transpile_options: dict | None = None,
        run_options: dict | None = None,
        approximation: bool = False,
        skip_transpilation: bool = False,
        abelian_grouping: bool = True,
    ):
        """
        Args:
            backend_options: Options passed to AerSimulator.
            transpile_options: Options passed to transpile.
            run_options: Options passed to run.
            approximation: If True, it calculates expectation values with normal distribution
                approximation.
            skip_transpilation: If True, transpilation is skipped.
            abelian_grouping: Whether the observable should be grouped into commuting.
                If approximation is True, this parameter is ignored and assumed to be False.
        """
        super().__init__(options=run_options)

        self._backend = backend
        self._transpile_options = Options()
        if transpile_options is not None:
            self._transpile_options.update_options(**transpile_options)
        self.approximation = approximation
        self._skip_transpilation = skip_transpilation
        self._cache: dict[
            tuple[tuple[int], tuple[int], bool], tuple[dict, dict]
        ] = {}
        self._transpiled_circuits: dict[int, QuantumCircuit] = {}
        self._layouts: dict[int, list[int]] = {}
        self._circuit_ids: dict[tuple, int] = {}
        self._observable_ids: dict[tuple, int] = {}
        self._abelian_grouping = abelian_grouping
        self._mitigator = mitigator

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        seed = run_options.pop("seed", None)
        if seed is not None:
            run_options.setdefault("seed_simulator", seed)

        if self.approximation:
            return self._compute_with_approximation(
                circuits, observables, parameter_values, run_options, seed
            )
        else:
            return self._compute(
                circuits, observables, parameter_values, run_options
            )

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> PrimitiveJob:
        circuit_indices: list = []
        for circuit in circuits:
            index = self._circuit_ids.get(_circuit_key(circuit))
            if index is not None:
                circuit_indices.append(index)
            else:
                circuit_indices.append(len(self._circuits))
                self._circuit_ids[_circuit_key(circuit)] = len(self._circuits)
                self._circuits.append(circuit)
                self._parameters.append(circuit.parameters)
        observable_indices: list = []
        for observable in observables:
            observable = init_observable(observable)
            index = self._observable_ids.get(_observable_key(observable))
            if index is not None:
                observable_indices.append(index)
            else:
                observable_indices.append(len(self._observables))
                self._observable_ids[_observable_key(observable)] = len(
                    self._observables
                )
                self._observables.append(observable)
        job = PrimitiveJob(
            self._call,
            circuit_indices,
            observable_indices,
            parameter_values,
            **run_options,
        )
        job.submit()
        return job

    def _compute(self, circuits, observables, parameter_values, run_options):
        if "shots" in run_options and run_options["shots"] is None:
            warn(
                "If `shots` is None and `approximation` is False, "
                "the number of shots is automatically set to backend options' "
                f"shots={self._backend.options.shots}.",
                RuntimeWarning,
            )

        # Key for cache
        key = (tuple(circuits), tuple(observables), self.approximation)

        # Create expectation value experiments.
        if key in self._cache:  # Use a cache
            experiments_dict, obs_maps = self._cache[key]
            exp_map = self._pre_process_params(
                circuits, observables, parameter_values, obs_maps
            )
            experiments, parameter_binds = self._flatten(
                experiments_dict, exp_map
            )
            post_processings = self._create_post_processing(
                circuits, observables, parameter_values, obs_maps, exp_map
            )
        else:
            self._transpile_circuits(circuits)
            circ_obs_map = defaultdict(list)
            # Aggregate observables
            for circ_ind, obs_ind in zip(circuits, observables):
                circ_obs_map[circ_ind].append(obs_ind)
            experiments_dict = {}
            obs_maps = (
                {}
            )  # circ_ind => obs_ind => term_ind (Original Pauli) => basis_ind
            # Group and create measurement circuit
            for circ_ind, obs_indices in circ_obs_map.items():
                pauli_list = sum(
                    [
                        self._observables[obs_ind].paulis
                        for obs_ind in obs_indices
                    ]
                ).unique()
                if self._abelian_grouping:
                    pauli_lists = pauli_list.group_commuting(qubit_wise=True)
                else:
                    pauli_lists = [PauliList(pauli) for pauli in pauli_list]
                obs_map = defaultdict(list)
                for obs_ind in obs_indices:
                    for pauli in self._observables[obs_ind].paulis:
                        for basis_ind, pauli_list in enumerate(pauli_lists):
                            if pauli in pauli_list:
                                obs_map[obs_ind].append(basis_ind)
                                break
                obs_maps[circ_ind] = obs_map
                bases = [
                    _paulis2basis(pauli_list) for pauli_list in pauli_lists
                ]
                if (
                    len(bases) == 1
                    and not bases[0].x.any()
                    and not bases[0].z.any()
                ):  # identity
                    break
                meas_circuits = [
                    self._create_meas_circuit(basis, circ_ind)
                    for basis in bases
                ]
                circuit = (
                    self._circuits[circ_ind]
                    if self._skip_transpilation
                    else self._transpiled_circuits[circ_ind]
                )
                experiments_dict[circ_ind] = self._combine_circs(
                    circuit, meas_circuits
                )
            self._cache[key] = experiments_dict, obs_maps

            exp_map = self._pre_process_params(
                circuits, observables, parameter_values, obs_maps
            )

            # Flatten
            experiments, parameter_binds = self._flatten(
                experiments_dict, exp_map
            )

            # Create PostProcessing
            post_processings = self._create_post_processing(
                circuits, observables, parameter_values, obs_maps, exp_map
            )

        # Run experiments
        if experiments:
            results = (
                self._backend.run(
                    circuits=experiments,
                    parameter_binds=parameter_binds
                    if any(parameter_binds)
                    else None,
                    **run_options,
                )
                .result()
                .results
            )
        else:
            results = []

        # Post processing (calculate expectation values)
        expectation_values, metadata = zip(
            *(
                post_processing.run(results, self._mitigator)
                for post_processing in post_processings
            )
        )
        return EstimatorResult(
            np.real_if_close(expectation_values), list(metadata)
        )

    def _pre_process_params(
        self, circuits, observables, parameter_values, obs_maps
    ):
        exp_map = defaultdict(
            dict
        )  # circ_ind => basis_ind => (parameter, parameter_values)
        for circ_ind, obs_ind, param_val in zip(
            circuits, observables, parameter_values
        ):
            self._validate_parameter_length(param_val, circ_ind)
            parameter = self._parameters[circ_ind]
            for basis_ind in obs_maps[circ_ind][obs_ind]:
                if (
                    circ_ind in exp_map
                    and basis_ind in exp_map[circ_ind]
                    and len(self._parameters[circ_ind]) > 0
                ):
                    param_vals = exp_map[circ_ind][basis_ind][1]
                    if param_val not in param_vals:
                        param_vals.append(param_val)
                else:
                    exp_map[circ_ind][basis_ind] = (parameter, [param_val])

        return exp_map

    @staticmethod
    def _flatten(experiments_dict, exp_map):
        experiments_list = []
        parameter_binds = []
        for circ_ind in experiments_dict:
            experiments_list.extend(experiments_dict[circ_ind])
            for _, (parameter, param_vals) in exp_map[circ_ind].items():
                parameter_binds.extend(
                    [
                        {
                            param: [param_val[i] for param_val in param_vals]
                            for i, param in enumerate(parameter)
                        }
                    ]
                )
        return experiments_list, parameter_binds

    def _create_meas_circuit(self, basis: Pauli, circuit_index: int):
        qargs = np.arange(basis.num_qubits)[basis.z | basis.x]
        meas_circuit = QuantumCircuit(basis.num_qubits, len(qargs))
        for clbit, qarg in enumerate(qargs):
            if basis.x[qarg]:
                if basis.z[qarg]:
                    meas_circuit.sdg(qarg)
                meas_circuit.h(qarg)
            meas_circuit.measure(qarg, clbit)
        meas_circuit.metadata = {"basis": basis}
        if self._skip_transpilation:
            return meas_circuit
        transpile_opts = copy(self._transpile_options)
        transpile_opts.update_options(
            initial_layout=self._layouts[circuit_index]
        )
        return transpile(meas_circuit, **transpile_opts.__dict__)

    @staticmethod
    def _combine_circs(
        circuit: QuantumCircuit, meas_circuits: list[QuantumCircuit]
    ):
        circs = []
        for meas_circuit in meas_circuits:
            new_circ = circuit.copy()
            for creg in meas_circuit.cregs:
                new_circ.add_register(creg)
            new_circ.compose(meas_circuit, inplace=True)
            _update_metadata(new_circ, meas_circuit.metadata)
            circs.append(new_circ)
        return circs

    @staticmethod
    def _calculate_result_index(
        circ_ind, obs_ind, term_ind, param_val, obs_maps, exp_map
    ) -> int:
        basis_ind = obs_maps[circ_ind][obs_ind][term_ind]

        result_index = 0
        for _circ_ind, basis_map in exp_map.items():
            for _basis_ind, (_, (_, param_vals)) in enumerate(
                basis_map.items()
            ):
                if circ_ind == _circ_ind and basis_ind == _basis_ind:
                    result_index += param_vals.index(param_val)
                    return result_index
                result_index += len(param_vals)
        raise AerError(
            "Bug. Please report from issue: https://github.com/Qiskit/qiskit-aer/issues"
        )

    def _create_post_processing(
        self, circuits, observables, parameter_values, obs_maps, exp_map
    ) -> list[_PostProcessing]:
        post_processings = []
        for circ_ind, obs_ind, param_val in zip(
            circuits, observables, parameter_values
        ):
            result_indices: list[int | None] = []
            paulis = []
            coeffs = []
            observable = self._observables[obs_ind]
            for term_ind, (pauli, coeff) in enumerate(
                zip(observable.paulis, observable.coeffs)
            ):
                # Identity
                if not pauli.x.any() and not pauli.z.any():
                    result_indices.append(None)
                    paulis.append(PauliList(pauli))
                    coeffs.append([coeff])
                    continue

                result_index = self._calculate_result_index(
                    circ_ind, obs_ind, term_ind, param_val, obs_maps, exp_map
                )
                if result_index in result_indices:
                    i = result_indices.index(result_index)
                    paulis[i] += pauli
                    coeffs[i].append(coeff)
                else:
                    result_indices.append(result_index)
                    paulis.append(PauliList(pauli))
                    coeffs.append([coeff])
            post_processings.append(
                _PostProcessing(result_indices, paulis, coeffs)
            )
        return post_processings

    def _compute_with_approximation(
        self, circuits, observables, parameter_values, run_options, seed
    ):
        # Key for cache
        key = (tuple(circuits), tuple(observables), self.approximation)
        shots = run_options.pop("shots", None)

        # Create expectation value experiments.
        if key in self._cache:  # Use a cache
            parameter_binds = defaultdict(dict)
            for i, j, value in zip(circuits, observables, parameter_values):
                self._validate_parameter_length(value, i)
                for k, v in zip(self._parameters[i], value):
                    if k in parameter_binds[(i, j)]:
                        parameter_binds[(i, j)][k].append(v)
                    else:
                        parameter_binds[(i, j)][k] = [v]
            experiment_manager = self._cache[key]
            experiment_manager.parameter_binds = list(parameter_binds.values())
        else:
            self._transpile_circuits(circuits)
            experiment_manager = _ExperimentManager()
            for i, j, value in zip(circuits, observables, parameter_values):
                if (i, j) in experiment_manager.keys:
                    self._validate_parameter_length(value, i)
                    experiment_manager.append(
                        key=(i, j),
                        parameter_bind=dict(zip(self._parameters[i], value)),
                    )
                else:
                    self._validate_parameter_length(value, i)
                    circuit = (
                        self._circuits[i].copy()
                        if self._skip_transpilation
                        else self._transpiled_circuits[i].copy()
                    )

                    observable = self._observables[j]
                    if shots is None:
                        circuit.save_expectation_value(
                            observable, self._layouts[i]
                        )
                    else:
                        for term_ind, pauli in enumerate(observable.paulis):
                            circuit.save_expectation_value(
                                pauli, self._layouts[i], label=str(term_ind)
                            )
                    experiment_manager.append(
                        key=(i, j),
                        parameter_bind=dict(zip(self._parameters[i], value)),
                        experiment_circuit=circuit,
                    )

            self._cache[key] = experiment_manager
        result = self._backend.run(
            experiment_manager.experiment_circuits,
            parameter_binds=experiment_manager.parameter_binds,
            **run_options,
        ).result()

        # Post processing (calculate expectation values)
        if shots is None:
            expectation_values = [
                result.data(i)["expectation_value"]
                for i in experiment_manager.experiment_indices
            ]
            metadata = [
                {"simulator_metadata": result.results[i].metadata}
                for i in experiment_manager.experiment_indices
            ]
        else:
            expectation_values = []
            rng = np.random.default_rng(seed)
            metadata = []
            experiment_indices = experiment_manager.experiment_indices
            for i in range(len(experiment_manager)):
                combined_expval = 0.0
                combined_var = 0.0
                result_index = experiment_indices[i]
                observable_key = experiment_manager.get_observable_key(i)
                coeffs = np.real_if_close(
                    self._observables[observable_key].coeffs
                )
                for term_ind, expval in result.data(result_index).items():
                    var = 1 - expval**2
                    coeff = coeffs[int(term_ind)]
                    combined_expval += expval * coeff
                    combined_var += var * coeff**2
                # Sampling from normal distribution
                standard_error = np.sqrt(combined_var / shots)
                expectation_values.append(
                    rng.normal(combined_expval, standard_error)
                )
                metadata.append(
                    {
                        "variance": np.real_if_close(combined_var).item(),
                        "shots": shots,
                        "simulator_metadata": result.results[
                            result_index
                        ].metadata,
                    }
                )

        return EstimatorResult(np.real_if_close(expectation_values), metadata)

    def _validate_parameter_length(self, parameter, circuit_index):
        if len(parameter) != len(self._parameters[circuit_index]):
            raise ValueError(
                f"The number of values ({len(parameter)}) does not match "
                f"the number of parameters ({len(self._parameters[circuit_index])})."
            )

    def _transpile(self, circuits):
        if self._skip_transpilation:
            return circuits
        return transpile(circuits, **self._transpile_options.__dict__)

    def _transpile_circuits(self, circuits):
        if self._skip_transpilation:
            for i in set(circuits):
                num_qubits = self._circuits[i].num_qubits
                self._layouts[i] = list(range(num_qubits))
            return
        for i in set(circuits):
            if i not in self._transpiled_circuits:
                circuit = self._circuits[i].copy()
                circuit.measure_all()
                num_qubits = circuit.num_qubits
                circuit = self._transpile(circuit)
                bit_map = {
                    bit: index for index, bit in enumerate(circuit.qubits)
                }
                layout = [bit_map[qr[0]] for _, qr, _ in circuit[-num_qubits:]]
                circuit.remove_final_measurements()
                self._transpiled_circuits[i] = circuit
                self._layouts[i] = layout


class _PostProcessing:
    def __init__(
        self,
        result_indices: list[int],
        paulis: list[PauliList],
        coeffs: list[list[float]],
    ):
        self._result_indices = result_indices
        self._paulis = paulis
        self._coeffs = [np.array(c) for c in coeffs]

    def run(
        self,
        results: list[ExperimentResult],
        mitigator: Union[LocalReadoutMitigator, CorrelatedReadoutMitigator],
    ) -> tuple[float, dict]:
        """Coumpute expectation value.

        Args:
            results: list of ExperimentResult.

        Returns:
            tuple of an expectation value and metadata.
        """
        combined_expval = 0.0
        combined_var = 0.0
        simulator_metadata = []
        for c_i, paulis, coeffs in zip(
            self._result_indices, self._paulis, self._coeffs
        ):
            if c_i is None:
                # Observable is identity
                expvals, variances = np.array([1], dtype=complex), np.array(
                    [0], dtype=complex
                )
                shots = 0
            else:
                result = results[c_i]
                counts = result.data.counts
                shots = sum(counts.values())
                if mitigator is not None:
                    for hex_key in list(counts.keys()):
                        counts[
                            f"{{:0{len(mitigator.qubits)}b}}".format(
                                int(hex_key, 16)
                            )
                        ] = counts.pop(hex_key)
                    counts = mitigator.quasi_probabilities(counts)
                    for val, pro in counts.items():
                        counts[val] = pro * shots
                basis = result.header.metadata["basis"]
                indices = np.where(basis.z | basis.x)[0]
                measured_paulis = PauliList.from_symplectic(
                    paulis.z[:, indices], paulis.x[:, indices], 0
                )
                expvals, variances = _pauli_expval_with_variance(
                    counts, measured_paulis, hex=(mitigator is None)
                )
                simulator_metadata.append(result.metadata)
            combined_expval += np.dot(expvals, coeffs)
            combined_var += np.dot(variances, coeffs**2)
        metadata = {
            "shots": shots,
            "variance": np.real_if_close(combined_var).item(),
            "simulator_metadata": simulator_metadata,
        }
        return combined_expval, metadata


def _update_metadata(
    circuit: QuantumCircuit, metadata: dict
) -> QuantumCircuit:
    if circuit.metadata:
        circuit.metadata.update(metadata)
    else:
        circuit.metadata = metadata
    return circuit


def _pauli_expval_with_variance(
    counts: dict, paulis: PauliList, hex: bool
) -> tuple[np.ndarray, np.ndarray]:
    # Diag indices
    size = len(paulis)
    diag_inds = _paulis2inds(paulis)

    expvals = np.zeros(size, dtype=float)
    denom = 0  # Total shots for counts dict
    for outcome, freq in counts.items():
        if hex:
            outcome = int(outcome, 16)
        denom += freq
        for k in range(size):
            coeff = (-1) ** _parity(diag_inds[k] & outcome)
            expvals[k] += freq * coeff

    # Divide by total shots
    expvals /= denom

    variances = 1 - expvals**2
    return expvals, variances


def _paulis2inds(paulis: PauliList) -> list[int]:
    nonid = paulis.z | paulis.x
    packed_vals = np.packbits(
        nonid, axis=1, bitorder="little"
    ).astype(  # pylint:disable=no-member
        object
    )
    power_uint8 = 1 << (8 * np.arange(packed_vals.shape[1], dtype=object))
    inds = packed_vals @ power_uint8
    return inds.tolist()


def _parity(integer: int) -> int:
    """Return the parity of an integer"""
    return bin(integer).count("1") % 2


def _paulis2basis(paulis: PauliList) -> Pauli:
    return Pauli(
        (
            np.logical_or.reduce(paulis.z),  # pylint:disable=no-member
            np.logical_or.reduce(paulis.x),  # pylint:disable=no-member
        )
    )


class _ExperimentManager:
    def __init__(self):
        self.keys: list[tuple[int, int]] = []
        self.experiment_circuits: list[QuantumCircuit] = []
        self.parameter_binds: list[dict[ParameterExpression, list[float]]] = []
        self._input_indices: list[list[int]] = []
        self._num_experiment: int = 0

    def __len__(self):
        return self._num_experiment

    @property
    def experiment_indices(self):
        """indices of experiments"""
        return sum(self._input_indices, [])

    def append(
        self,
        key: tuple[int, int],
        parameter_bind: dict[ParameterExpression, float],
        experiment_circuit: QuantumCircuit | None = None,
    ):
        """append experiments"""
        if experiment_circuit is not None:
            self.experiment_circuits.append(experiment_circuit)

        if key in self.keys:
            key_index = self.keys.index(key)
            for k, vs in self.parameter_binds[key_index].items():
                vs.append(parameter_bind[k])
            self._input_indices[key_index].append(self._num_experiment)
        else:
            self.keys.append(key)
            self.parameter_binds.append(
                {k: [v] for k, v in parameter_bind.items()}
            )
            self._input_indices.append([self._num_experiment])

        self._num_experiment += 1

    def get_observable_key(self, index):
        """return key of observables"""
        for i, inputs in enumerate(self._input_indices):
            if index in inputs:
                return self.keys[i][1]
        raise AerError("Unexpected behavior.")
