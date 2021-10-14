# VQE-in-DFT

This package provides methods to fragment molecular Hamiltonians and to simulate these using Variational Qauntum Eigensolver algorithms embedded into Density Functional Theory.

## Use

Note: PySCF is not supported on Windows, so until alternative chemistry backends are implemented, this package will work only for Linux and MacOS.
### Poetry

Poetry is a packaging and dependency manager, to install it from the command line run::

    pip install poetry

with this installed, you can start working on the package by running:

    poetry install

which will create a virtual environment with the required dependencies.

This virtual environment subsequently can be activated with:

    poetry shell

## Structure

### Notebooks

This folder contains examples of implementations using Jupyter notebooks

### tests

Contains all tests of the package

### vqe-in-dft

The main content of the package are included here.
