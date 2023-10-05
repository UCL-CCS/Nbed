# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Fixed
- Typo with the license file, which was called 'LISENCE' instead of 'LICENSE'.
### Added
- Support for unrestricted SCF methods
- Tests for unrestricted SCF methods
### Changed
- pyscf version updated to `2.3.0` to correct installation issues with `1.7.6.post1`
- Location of explainer notebooks moved up a level to `docs/notebooks`


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