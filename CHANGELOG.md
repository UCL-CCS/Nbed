# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## Added
- `.pre-commit-config.yaml` added

## Changed
- python version requirement changed to `>=3.9, <4`

## [0.0.7]
### Changed
- `HamiltonianBuilder` now checks type of indices of mo_occ to determine resteriction.
- Black version `^24.0.0`
- Summer version now `symmer=~0.0.7`.

## [0.0.6]
### Changed
- `symmer dependence was pinned to 0.0.6, now ^0.0.6

## [0.0.5]
### Fixed

### Changed
- `HamBuilder` uses `Symmer` functionality to calculate frozen core integrals.
- black version update to `>22` from `^22`
- `HamConverter` now uses `SparsePauliOp` rather than `PauliSumOp`.
- Symmer version dependency updates to `0.0.6`

### Added
- `get_hartree_fock_state` function in `HamBuilder`
- `run_qmmm`, `mm_coords`, `mm_charges`, `mm_radii` and `symmetry` attributes for NbedDriver.

### Deprecated
- `HamBuilder.build` has `taper` bool input, but the main way to do this is now the `Symmer.QubitReductionDriver`

## [0.0.4]
### Fixed
- Typo on `xc_functional` arg for driver
- HamiltonianBuilder sets occupancy correctly for both restricted and unrestricted
- issue with concentric localization which left `c_ispan` unchanged over iterations
- error in `test_localizers.py` which had incorrect shell sizes in assert

### Changed
- Driver defaults to `run_virtual_localization=True`
- `frozen_orb_list` of embedded PySCF functions renamed `frozen` in line with PySCF
- readthedocs config updated to python3.9
- readthedocs build controlled directly with commands

### Added
- `frozen` option for FCI calls CASSCF
- `driver.cl_shells` attribute assigned when concentric localization is run
- `SpadeLocalizer.singular_values` and `.shells` properties

## [0.0.3]
### Fixed
- Correct error in embedded CCSD energy.
- Correct error in driver for calculating e_act

### Changed
- tests now use `b3lyp5` functional to match those in v1 before PySCF update.
- Dependencies updated to enable pypi install on Apple Silicon devices.

### Added
- Function to convert Symmer PualiwordOp to openfermion faster (to be removed when updated symmer is released)
- `notebook/publications` containing jupyter notebook for replicating results of PBE paper.

## [0.0.2]
### Fixed
- Typo with the license file, which was called 'LISENCE' instead of 'LICENSE'.
### Added
- Support for unrestricted SCF methods.
    - `spin` argument for Driver
    - `force_unrestricted` argument for Driver
    - Builder calculates electron integrals correctly
    - Active space reduction in builder
- Tests for unrestricted SCF methods.
- Virtual orbital localization in SPADE localizer, using concentric localization.
- `HamBuilder.build()` now allows boolean input for qubit tapering and contextual subspace projection.
### Changed
- pyscf version updated to `2.3.0` to correct installation issues with `1.7.6.post1`
- Location of explainer notebooks moved up a level to `docs/notebooks`
- Update of dependencies
- Python dependency updated to `>=3.8, <3.11` to pull in symmer.
- `HamBuilder.build()` now accepts MO indices for active space reduction. This overwrites `n_qubits` if used.
### Removed
- `savefile` argument removed from driver, as it is not used.


## [0.0.1] - 2022-03-07
### Fixed
- `NbedDriver._init_local_rhf` reversion which used the old `local_basis_transform`
### Added
- Basic functionality for initial release
- Sphinx documentation on readthedocs
- Changelog!
- DOI button from zenodo in readme
- DOI added to citation file
### Changed
- LISCENCE.txt changed to LICENSE.md
