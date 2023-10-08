# QCfDD

## Introduction

## Overview of Requirements

We strongly recommend running code on Linux x86_64.

We use Anaconda or Miniconda to create a virtual environment, and we recommend using the latest version of `conda`.

```shell
conda create -n qcfdd python=3.9
conda activate qcfdd
```

Continue to use pip to install the required environment.

```shell
pip install -r requirements.txt
```

The `pip install` process on Ubuntu 22.04 is very smooth. The installation on MacOS (especially on Macs with M chips) may encounter some difficulties when building GPy. You can try the following commands.

```shell
pip install scipy
pip install requirements.txt
```

We do not recommend execute this application on Windows, so we did not test the installation proces on Windows.

## Steps to execute the code

### Single Execution

We provide the optimal circuit for each noise model rendered in `QASM` format and the corresponding optimal parameters. The notebook describes the process of loading circuits and their parameters and running VQE. You can easily modify the values of various hyperparameters, such as `seed`, `shots`, etc.

### Full Execution

If you want to view the complete iteration process of VQE, `main.py` is the entry point of the program.

`main.py` provides a wealth of optional parameters for various configurations during program execution. Please understand the meaning of each parameter through `python main.py -h` before running.

For convenience, you can directly run the scripts in `./shell`, we have already set excellent hyperparameters for you under different noise models.

## Technical Reflection & Description

## Experiment Results