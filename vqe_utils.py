import csv
import json
from timeit import default_timer as timer
from typing import Tuple

import hypermapper
import numpy as np
import stim
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import EfficientSU2
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.units import DistanceUnit


def get_param_num(n_qubits: int, ansatz_reps: int) -> int:
    """Get the number of parameters in ansatz.

    Parameters
    ----------
    n_qubits : int
        Number of qubits needed by VQE.
    ansatz_reps : int
        Repetition of ansatz layer.

    Returns
    -------
    int
        The number of parameters in ansatz.
    """
    _, num_params = _efficientsu2_full(n_qubits, ansatz_reps)
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


def _efficientsu2_full(n_qubits, repetitions):
    ansatz = EfficientSU2(
        num_qubits=n_qubits,
        entanglement="full",
        reps=repetitions,
        insert_barriers=True,
    )
    num_params_ansatz = len(ansatz.parameters)
    ansatz = ansatz.decompose()
    return ansatz, num_params_ansatz


def _add_ansatz(circuit, parameters, ansatz_reps=1):
    n_qubits = circuit.num_qubits
    ansatz, _ = _efficientsu2_full(n_qubits, ansatz_reps)
    ansatz.assign_parameters(parameters=parameters, inplace=True)
    circuit.compose(ansatz, inplace=True)


def get_vqe_circuit(
    n_qubits: int,
    params: np.ndarray,
    pauli: str,
    **kwargs,
) -> QuantumCircuit:
    """Construct a single VQE circuit.

    Parameters
    ----------
    n_qubits : int
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
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    circuit = QuantumCircuit(qr, cr)
    init_last = kwargs.get("init_last", False)
    HF_bitstring = kwargs.get("HF_bitstring", None)

    if not init_last:
        _append_by_hartreefock(circuit, HF_bitstring)
    # HINT: Append the circuit with the state preparation ansatz.
    if params is not None:
        _add_ansatz(circuit, params, kwargs.get("ansatz_reps", 1))
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
    n_qubits,
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
    n_qubits (Int): Number of qubits in circuit.
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

    vqe_qc = QuantumCircuit(n_qubits)
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
    n_qubits,
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
    n_qubits (Int): Number of qubits in circuit.
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
    _, num_params = _efficientsu2_full(n_qubits, vqe_kwargs["ansatz_reps"])
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
            n_qubits=n_qubits,
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
