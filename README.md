[![Master CICD](https://github.com/UCL-CCS/Nbed/actions/workflows/push_to_master.yaml/badge.svg)](https://github.com/UCL-CCS/Nbed/actions/workflows/push_to_master.yaml) [![Documentation Status](https://readthedocs.org/projects/nbed/badge/?version=latest)](https://nbed.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/341631818.svg)](https://zenodo.org/badge/latestdoi/341631818)



# Nbed
This package implements projection-based embedding methods to reduce the size of a molecular Hamiltonain via embedding in DFT.

Nbed uses PySCF as a backend for chemistry caluculations, which is not supported on Windows. Alternative chemistry backends are planned, however in the mean time this package will work only for Linux and MacOS.

## Documentation
Full documentation is available at [https://nbed.readthedocs.io](https://nbed.readthedocs.io).

## Installation
### Pip

The package is available on [PyPI](https://pypi.org/project/nbed/) and can be installed with pip:

```
pip install nbed
```

### Dependencies

Development of Nbed uses the packaging and dependency manager uv, to install it from the command line run::
```shell
    pip install uv
```
with this installed, you can start working on the package by running:
```shell
    uv venv .venv/
    uv pip install .
```

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
Installing this package also exposes a command line tool `nbed`, which can be used in two ways. Firstly, you can provide a JSON config file.

```
nbed --config <path to .json>
```

Your JSON config file should look something like this (which is taken from the `tests` folder):

```JSON
{
  "geometry":"3\n\nO   0.0000  0.000  0.115\nH   0.0000  0.754  -0.459\nH   0.0000  -0.754  -0.459",
  "n_active_atoms":2,
  "basis":"STO-3G",
  "xc_functional":"b3lyp",
  "projector":"mu",
  "localization":"spade",
  "convergence":1e-6,
  "charge":0,
  "spin":0,
  "unit":"angstrom",
  "symmetry":false,
  "mu_level_shift":1000000.0,
  "run_ccsd_emb":false,
  "run_fci_emb":false,
  "run_dft_in_dft":false,
  "run_virtual_localization":true,
  "n_mo_overwrite":[null,null],
  "max_ram_memory":4000,
  "occupied_threshold":0.95,
  "virtual_threshold":0.95,
  "max_shells":4,
  "init_huzinaga_rhf_with_mu":false,
  "max_hf_cycles":50,
  "max_dft_cycles":50,
  "force_unrestricted":false,
  "mm_coords":null,
  "mm_charges":null,
  "mm_radii":null,
}
```

#### Reference Values



## Save Output

By including the `savefile` item in your config file or giving a `savefile` argument to the function, you can specify the path to a location where you'd like to save a JSON file containing the output of Nbed.


## Overview

- `embed.py` - main functionality
- `config.py` - Data validation model of configuration needed to run Nbed.
- `driver.py` - Class which carries out the algorithm. Main point of access for functionality.
- `ham_builder.py` - class to build Hamiltonians from quantum chemistry calculations.
- `localizers/` - Classes which perform localization.
- `utils.py` - log settings and cli parsing.

## Examples and Explainers
This [folder](https://github.com/UCL-CCS/Nbed/tree/master/docs/source/notebooks) contains jupyter notebooks which explain the embedding procedure in detail, including relevant theory. Notebooks to replicate results presented in publications can also be found here.

## Development
If you would like to contribute to this codebase please first create an issue describing your feature request or bug. We'll be happy to help.

If you have made changes yourself, make sure to fork the repo and open your PR from there.
