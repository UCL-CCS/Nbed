[![Master CICD](https://github.com/UCL-CCS/Nbed/actions/workflows/push_to_master.yaml/badge.svg)](https://github.com/UCL-CCS/Nbed/actions/workflows/push_to_master.yaml) [![Documentation Status](https://readthedocs.org/projects/nbed/badge/?version=latest)](https://nbed.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/341631818.svg)](https://zenodo.org/badge/latestdoi/341631818)



# Nbed
This package implements projection-based embedding methods to reduce the size of a molecular Hamiltonain via embedding in DFT.

Nbed uses PySCF as a backend for chemistry caluculations, which is not supported on Windows. Alternative chemistry backends could be added, however in the mean time this package will work only for Linux and MacOS.

Note the active space code is **experimental**. The full space version should be stable. 


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

## Development
If you would like to contribute to this codebase please first create an issue describing your feature request or bug. We'll be happy to help.

If you have made changes yourself, make sure to fork the repo and open your PR from there.
