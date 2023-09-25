# %% [markdown]
# # Example Code for Estimating the Ground State Energy of Hydroxyl (·OH)

# %% [markdown]
# ## Basic Installation

# %% [markdown]
# Install required package, we highly recommend participant to use qiskit platform, or at least participants can finish preprocessing at other platform and transfer the circuit to qiskit format, since our noise model is from IBM real machine backend and we restricted some algorithmic seeds which could be varied from different platform.

# %%
#!pip install qiskit
#!pip install qiskit-nature[pyscf] -U

# %%
#!pip install qiskit_aer

# %%
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper,ParityMapper,QubitConverter
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import SLSQP,SPSA
from qiskit_aer.primitives import Estimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
import numpy as np
import pylab
import qiskit.providers
from qiskit import Aer,pulse, QuantumCircuit, transpile
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.quantum_info import Kraus, SuperOp
from qiskit_aer import AerSimulator
from qiskit.tools.visualization import plot_histogram
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from qiskit.tools.jupyter import *
from qiskit import *
from qiskit.providers.aer.noise import NoiseModel
import qiskit.providers.aer.noise as noise
from qiskit.utils import QuantumInstance, algorithm_globals
import time

from qiskit.providers.fake_provider import *
import pickle



with open('NoiseModel/fakekolkata.pkl', 'rb') as file:
    noise_model = pickle.load(file)
noise_model1 = noise.NoiseModel()
noise_modelreal = noise_model1.from_dict(noise_model)
noise_modelreal
# %% [markdown]
# Here we require paticipants to fix the algorithm seed in qiskit. *MUST* translate other format circuit to qiskit before any place need algorithm seed. And we give 20, 21, 30, 33, 36, 42, 43, 55, 67, 170 as seeds that requires to run, and the result will be calculated as the average of results from each seed. And please use shots as 4000.

# %%
seeds = 170
algorithm_globals.random_seed = seeds
seed_transpiler = seeds
iterations = 125
shot = 6000

# %% [markdown]
# ## Generate Hamiltonian and Pauli String

# %% [markdown]
# At this step, the example code uses PySCF to generate the hamiltonian of hydroxyl with basis function as 'sto3g' to fit the spin orbital, then uses JordanWignerMapper to map the fermionic terms to pauli strings. To be noticed, other chemistry tool also allowed to be used at this step, but keep in mind to use 'sto-3g' and Jordan Wigner Mapper which should gives 12 qubits and 631 paulil terms.

# %%
ultra_simplified_ala_string = """
O 0.0 0.0 0.0
H 0.45 -0.1525 -0.8454
"""

driver = PySCFDriver(
    atom=ultra_simplified_ala_string.strip(),
    basis='sto3g',
    charge=1,
    spin=0,
    unit=DistanceUnit.ANGSTROM
)
qmolecule = driver.run()

# %%
hamiltonian = qmolecule.hamiltonian
coefficients = hamiltonian.electronic_integrals
#print(coefficients.alpha)
second_q_op = hamiltonian.second_q_op()

# %%
mapper = JordanWignerMapper()
converter = QubitConverter(mapper=mapper, two_qubit_reduction=False)
qubit_op = converter.convert(second_q_op)

# %% [markdown]
# We recommend to use classical minimum eigensolver to obtain a reference energy at this step. In case some of the classical minimum eigensolver donot directly gives nuclear repulsion energy, we give reference energies below: *Comupted Energy*: -78.75252123, *Nuclear Repulsion_energy*: 4.36537496654537. *Obtained Reference Ground State Energy*: -74.38714627.

# %%
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

solver = GroundStateEigensolver(
    JordanWignerMapper(),
    NumPyMinimumEigensolver(),
)

# %%
result = solver.solve(qmolecule)
print(result.computed_energies)

# %%
print(result.nuclear_repulsion_energy)

# %%
ref_value = result.computed_energies + result.nuclear_repulsion_energy
print(ref_value)

# %% [markdown]
# ## Construct Ansatz

# %% [markdown]
# At this stage, you can implement various techniques to search good ansatz architecture which is important for variational quantum algorihms. Moreover, how to obtain a good initial state is a good topic to do research, we require participant to self-reflection there techniques (include the techniques for preprocessing ansatz or initial states) with maximum 10 points, and submit a short description for used techniques, we will have three graders to evaluate the techniques.

# %%
ansatz = UCCSD(
    qmolecule.num_spatial_orbitals,
    qmolecule.num_particles,
    mapper,
    initial_state=HartreeFock(
        qmolecule.num_spatial_orbitals,
        qmolecule.num_particles,
        mapper,
    ),
)


# %%
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
#from qiskit.providers.aer.noise import NoiseModel

# %%
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeVigo

'''# Build noise model from backend properties
backend = FakeVigo()
noise_model = NoiseModel.from_backend(backend)'''

# Get coupling map from backend
#coupling_map = backend.configuration().coupling_map

# %%
estimator = Estimator(
    abelian_grouping=True,
    approximation=True,
    backend_options = {
        'method': 'statevector',
        'device': 'CPU',
        'noise_model': noise_model1,
        #'coupling_map': coupling_map,
    },
    run_options = {
        'shots': shot,
        'seed': seeds,
    },
    transpile_options = {
        'seed_transpiler':seed_transpiler,
        #'backend': backend,
    }
)

# %%
vqe_solver = VQE(estimator, ansatz, SPSA())
vqe_solver.initial_point = [0.0] * ansatz.num_parameters

import datetime
print(datetime.datetime.now())

# %%
calc = GroundStateEigensolver(mapper, vqe_solver)
res = calc.solve(qmolecule)
print('noisy')
print(res)
print(datetime.datetime.now())

# %% [markdown]
# ## Calculate the Accuracy (Most Important Metric)

# %%
'''
result = res.computed_energies + res.nuclear_repulsion_energy
error_rate = abs(abs(ref_value - result) / ref_value * 100)
print("Error rate: %f%%" % (error_rate))
'''

# %% [markdown]
# ## Obtain the Duration of Quantum Circuit

# %%
from qiskit.providers.fake_provider import *
backend = FakeMontreal()

# %%
with pulse.build(backend) as my_program1:
  pulse.call(ansatz)

# %%
my_program1.duration


