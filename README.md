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
  # optionally add
  # run_ccsd: True
```

Alternatively you can provide each of the components to the command line.

```
nbed --geometry tests/molecules/water.xyz --active_atoms 2 --convergence 1e-6 --qubits 8 --basis STO-3G--xc b3lyp --output openfermion --localisation spade
```

The options for `output` and `localisation` can be seen in the command help.

```
nbed --help
```



## Structure

### Notebooks

This folder contains examples of implementations using Jupyter notebooks

### tests

Contains all tests of the package

### vqe-in-dft

The main content of the package are included here.
