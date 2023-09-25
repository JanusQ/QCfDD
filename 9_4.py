# %%
#!pip install qiskit-nature[pyscf] -U
#!pip install qiskit-nature
#!pip install cmake
#!pip install --prefer-binary pyscf
#!pip install git
#!pip install git+https://github.com/pyscf/pyscf
#!pip install --prefer-binary pyscf
#!conda install -c conda-forge pyscf

#!conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
#!conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
#!conda config --add channels https://anaconda.org
#!conda install pyscf
#!pip show pyscf
#!pip uninstall qiskit_nature --yes
#!pip uninstall pyscf --yes


#!pip install qiskit-nature[pyscf] -U
#!pip show qiskit_nature 
#!pip show pyscf

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
from qiskit import Aer,pulse, QuantumCircuit
from qiskit.utils import QuantumInstance, algorithm_globals
import time

# %%
seeds = 170
algorithm_globals.random_seed = seeds
seed_transpiler = seeds
iterations = 125
shot = 4000


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

# %% [markdown]
# 出错是因为Windows不支持pyscf,要放到linux上去运行

# %%
hamiltonian = qmolecule.hamiltonian
coefficients = hamiltonian.electronic_integrals
print(coefficients.alpha)
second_q_op = hamiltonian.second_q_op()

# %% [markdown]
# 以上来自samplecode

# %%
mapper = JordanWignerMapper()

# %%
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

numpy_solver = NumPyMinimumEigensolver()

# %%
estimator = Estimator(
    abelian_grouping=True,
    approximation=True,
    backend_options = {
        'method': 'statevector',
        'device': 'CPU'
        # 'noise_model': noise_model
    },
    run_options = {
        'shots': shot,
        'seed': seeds,
    },
    transpile_options = {
        'seed_transpiler':seed_transpiler
    }
)

# %% [markdown]
# 不理解上面的这个代码块,它是全程序的关键所在. 遇到了矛盾,帮助手册查不出... :(

# %% [markdown]
# 上面这个estimator是差别产生的根源

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

vqe_solver = VQE(estimator, ansatz, SPSA())
vqe_solver.initial_point = [0.0] * ansatz.num_parameters

# %%
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

calc = GroundStateEigensolver(mapper, vqe_solver)

# %%
import datetime
print(datetime.datetime.now())
res = calc.solve(qmolecule)
print(res)
print(datetime.datetime.now())

# %%
calc = GroundStateEigensolver(mapper, numpy_solver)
res = calc.solve(qmolecule)
print(res)


