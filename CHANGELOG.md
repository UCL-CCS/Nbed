# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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