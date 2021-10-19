[![CICD Status](https://github.com/AlexisRalli/VQE_in_DFT/actions/workflows/actions.yaml/badge.svg?branch=master)](https://github.com/AlexisRalli/VQE_in_DFT/actions/workflows/actions.yaml)

# Nbed

This package contains a method for embedding quantum simulation algorithms within DFT.

Note: PySCF is not supported on Windows, so until alternative chemistry backends are implemented, this package will work only for Linux and MacOS.
## Installation
### Pip

The package is installable from the top level directory using

```
pip install .
```
### Poetry

Poetry is a packaging and dependency manager, to install it from the command line run::

    pip install poetry

with this installed, you can start working on the package by running:

    poetry install

which will create a virtual environment with the required dependencies.

This virtual environment subsequently can be activated with:

    poetry shell
## Use

The package has three main interfaces, each to the same function `embed/embedding_terms`. This function is accessable by importing the package into a python file

```
from nbed import nbed
...

nbed(...)
```

Installing this package also exposes a command line tool `nbed`, which can be used in two ways. Firstly, you can provide a YAML config file.

```
nbed --config <path to .yaml>
```

Your yaml config file should look something like this:

```
---
nbed:
  geometry: tests/molecules/water.xyz
  active_atoms: 2
  convergence: !!float 1e-6
  qubits: 8
  basis: STO-3G
  xc_functional: b3lyp
  output: openfermion
  localisation: spade
  savefile: data/savefile.json
```

Alternatively you can provide each of the components to the command line.

```
nbed --geometry tests/molecules/water.xyz --active_atoms 2 --convergence 1e-6 --qubits 8 --basis STO-3G--xc b3lyp --output openfermion --localisation spade --savefile data/savefile.json
```

The options for `output` and `localisation` can be seen in the command help.

```
nbed --help
```

### Reference Values

Additionally, to output a CCSD reference value for the whole system energy, add a line to the yaml file when using `--config`

```
---
nbed:
  ...
  ccsd: true

```

or use the the `--ccsd` flag when inputing values manually.

```
nbed --config <path to config file> -
```

### Save a Hamiltonian for later

By including the `--savefile` flag or `savefile` item in your config file, you can specify the path to a location where you'd like to save a JSON file containing a description of the qubit Hamiltonian.

Once you have a saved Hamiltonian you can use the `load_hamiltonian` function to create a python object of the desired type.

```
from nbed import load_hamiltonian
...

qham = load_hamiltonian(<path to hamiltonian JSON>, <output type>)
```

## Structure

```
VQE_IN_DFT
    notebooks
    tests
    nbed
```

### Notebooks

This folder contains jupyter notebooks which explain the embedding procedure in detail, including relevant theory.

### Tests

Contains all tests of the package

### nbed

Main functionality of the package.

- embed - main functionality
- ham_converter - class to convert between Hamiltonian formats as well as save to and read from JSON.
- localisation - methods of orbital localisation
- mol_plot - functions to plot the systems localised molecular orbitals.
- utils - log settings and cli parsing.