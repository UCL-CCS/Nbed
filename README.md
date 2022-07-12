[![CICD Status](https://github.com/UCL-CCS/Nbed/actions/workflows/release.yaml/badge.svg?branch=master)](https://github.com/UCL-CCS/Nbed/actions/workflows/release.yaml) [![Documentation Status](https://readthedocs.org/projects/nbed/badge/?version=latest)](https://nbed.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/341631818.svg)](https://zenodo.org/badge/latestdoi/341631818)



# Nbed

This package implements projection-based embedding methods to reduce the size of a molecular Hamiltonain via embedding in DFT. Output qubit hamiltonains can be solved by a suitable quantum algorithm.

Nbed uses PySCF as a backend for chemistry caluculations, which is not supported on Windows. Alternative chemistry backends are planned, however in the mean time this package will work only for Linux and MacOS.

## Documentation

Full documentation is available at [https://nbed.readthedocs.io](https://nbed.readthedocs.io).

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

The package has three main interfaces, each to the same function `embed/nbed`. 

### Importing the package
This function is accessable by importing the package into a python file.

```
from nbed import nbed
...

nbed(...)
```

This function will output a qubit Hamiltonian suitable for the backend specified by the `output` argument.

### Command Line Interface
Installing this package also exposes a command line tool `nbed`, which can be used in two ways. Firstly, you can provide a YAML config file.

```
nbed --config <path to .yaml>
```

Your YAML config file should look something like this (which is taken from the `tests` folder):

```
---
nbed:
  geometry: tests/molecules/water.xyz
  n_active_atoms: 3
  basis: STO-3G
  xc_functional: b3lyp
  output: openfermion
  projector: huzinaga
  localization: spade
  convergence: !!float 1e-9
  savefile: data/savefile.json
  transform: jordan_wigner
  run_ccsd_emb: True
  run_fci_emb: True
  unit: angstrom
```

Alternatively you can provide each of the components to the command line.

```
nbed --geometry tests/molecules/water.xyz --active_atoms 2 --convergence 1e-6 --qubits 8 --basis STO-3G--xc b3lyp --output openfermion --localization spade --savefile data/savefile.json
```

The options for `output` and `localization` can be seen in the command help.

```
nbed --help
```

#### Reference Values

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

## Save a Hamiltonian for later

By including the `--savefile` flag or `savefile` item in your config file or giving a `savefile` argument to the function, you can specify the path to a location where you'd like to save a JSON file containing a description of the qubit Hamiltonian.

Once you have a saved Hamiltonian you can use the `nbed.load_hamiltonian` function to create a python object of the desired type.

```
from nbed import load_hamiltonian
...

qham = load_hamiltonian(<path to hamiltonian JSON>, <output type>)
```

## Structure

```
Nbed
    docs_source
    nbed
    notebooks
    logs
    tests
```
### nbed

Main functionality of the package.

- `embed.py` - main functionality
- `driver.py` - Class which carries out the algorithm. Main point of access for functionality.
- `ham_converter.py` - class to convert between Hamiltonian formats as well as save to and read from JSON.
- `ham_builder.py` - class to build Hamiltonians from quantum chemistry calculations.
- `localizers/` - Classes which perform localization.
- `mol_plot.py` - functions to plot the systems localised molecular orbitals.
- `utils.py` - log settings and cli parsing.

### Notebooks
This [folder](https://github.com/UCL-CCS/Nbed/tree/master/docs/source/notebooks) contains jupyter notebooks which explain the embedding procedure in detail, including relevant theory. Notebooks to replicate results presented in publications can also be found here.


### Tests

Contains all tests of the package. These can be run from the command line using `pytest`.

### Logs
Each time the package is initialised a new log will be started in the top level director at `Nbed/.nbed.log`.

## Development
If you would like to contribute to this code base please first create an issue and a fork of the repo from which to make your pull request.