"""Shared fixtures: small molecules, cached mean fields and a numeric snapshot store."""

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, settings
from pyscf import dft, gto, scf

from nbed.act_env_space import select_act_env_space
from nbed.emb_scf import EmbedSCF

settings.register_profile(
    "default",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

XC = "b3lyp"
MU = 1e6
# huzinaga level shift used throughout; without it formamide HF-in-DFT oscillates
HUZ_SHIFT = 1.0

# name -> (geometry in angstrom, fragment atoms, spin)
MOLECULES = {
    "h2": ("H 0 0 0; H 0 0 0.74", [0], 0),
    "water": ("O 0 0 0.115; H 0 0.754 -0.459; H 0 -0.754 -0.459", [0], 0),
    "methanol": (
        "O -0.6582 -0.0067 0.1730; H -1.1326 -0.0311 -0.6482; "
        "C 0.7031 0.0083 -0.1305; H 0.9877 0.8943 -0.7114; "
        "H 1.0155 -0.8918 -0.6742; H 1.2001 0.0363 0.8431",
        [0, 1],
        0,
    ),
    "formamide": (
        "C 0.0000 0.4165 0.0000; O 1.1942 0.2217 0.0000; N -0.9373 -0.5551 0.0000; "
        "H -0.4299 1.4182 0.0000; H -0.6608 -1.5171 0.0000; H -1.9020 -0.2881 0.0000",
        [1],
        0,
    ),
    "acetonitrile": (
        "C 0 0 -1.1949; C 0 0 0.2914; N 0 0 1.4457; H 0 1.0230 -1.5662; "
        "H 0.8860 -0.5115 -1.5662; H -0.8860 -0.5115 -1.5662",
        [1, 2],
        0,
    ),
    "methyl": ("C 0 0 0; H 1.079 0 0; H -0.5395 0.9344 0; H -0.5395 -0.9344 0", [0], 1),
}
CLOSED_SHELL = [name for name, (_, _, spin) in MOLECULES.items() if spin == 0]
# molecules with an environment left over after a two-orbital fragment
EMBEDDABLE = [name for name in MOLECULES if name != "h2"]


def build_mol(name, basis="sto-3g", geometry=None):
    """Build a molecule from the table, optionally overriding its geometry."""
    atom, _, spin = MOLECULES[name]
    return gto.M(atom=geometry or atom, basis=basis, spin=spin, verbose=0)


def run_ks(mol):
    """Converged restricted (open-shell) B3LYP."""
    mf = (dft.ROKS if mol.spin else dft.RKS)(mol, xc=XC)
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    return mf


@functools.lru_cache(maxsize=None)
def global_ks(name, basis="sto-3g"):
    """Converged global KS for a table molecule, cached across tests."""
    return run_ks(build_mol(name, basis))


@functools.lru_cache(maxsize=None)
def global_hf(name, basis="sto-3g"):
    """Converged global HF for a table molecule, cached across tests."""
    mol = build_mol(name, basis)
    mf = (scf.ROHF if mol.spin else scf.RHF)(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf


def make_embedding(mf, atoms, n_occ=2, n_vir=2, mu=MU):
    """Partition a mean field and wrap it in EmbedSCF."""
    act, env, _, _ = select_act_env_space(mf, atoms, n_occ, n_vir)
    emb = EmbedSCF(
        mf, act, env, mf.mo_coeff, mf.mo_occ, mf.get_ovlp(), 4000, mu_val=mu
    )
    return emb, act, env


@functools.lru_cache(maxsize=None)
def embedding(name, basis="sto-3g", n_occ=2, n_vir=2):
    """Cached EmbedSCF for a table molecule and its fragment atoms."""
    return make_embedding(global_ks(name, basis), MOLECULES[name][1], n_occ, n_vir)[0]


@dataclass
class EmbeddedRun:
    """Outputs of one embedded mean-field run."""

    e_tot: float
    mf: object
    emb_corr: float
    env_cols: np.ndarray
    env_plus_corrections: float


@functools.lru_cache(maxsize=None)
def embedded_run(name, method, proj, basis="sto-3g"):
    """Cached result of build_emb_dft / build_emb_hf."""
    emb = embedding(name, basis)
    if method == "dft":
        out = emb.build_emb_dft(XC, proj_type=proj, huz_level_shift=HUZ_SHIFT, warn=False)
    else:
        out = emb.build_emb_hf(proj_type=proj, huz_level_shift=HUZ_SHIFT, warn=False)
    return EmbeddedRun(*out)


##################################################################################
############################## numeric snapshots #################################
##################################################################################

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def pytest_addoption(parser):
    """Register --snapshot-update."""
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        help="rewrite tests/snapshots/*.json instead of comparing against them",
    )


def _to_jsonable(value):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_to_jsonable(v) for v in np.asarray(value).tolist()]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _compare(actual, expected, path, rtol, atol):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), f"{path}: keys differ"
        for key in expected:
            _compare(actual[key], expected[key], f"{path}.{key}", rtol, atol)
    elif isinstance(expected, list):
        assert np.shape(actual) == np.shape(expected), f"{path}: shape differs"
        assert np.allclose(actual, expected, rtol=rtol, atol=atol), (
            f"{path}: {actual} != {expected}"
        )
    elif isinstance(expected, float):
        assert np.isclose(actual, expected, rtol=rtol, atol=atol), (
            f"{path}: {actual!r} != {expected!r}"
        )
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


@pytest.fixture
def snapshot(request):
    """Compare a dict of numbers against tests/snapshots/<test id>.json.

    Floats are compared with a tolerance rather than as strings, so the snapshots
    survive changes of BLAS or thread count. Run pytest with --snapshot-update to
    (re)write them.
    """
    update = request.config.getoption("--snapshot-update")
    stem = request.node.name.replace("[", "__").replace("]", "").replace("/", "_")
    path = SNAPSHOT_DIR / f"{stem}.json"

    def check(data, rtol=1e-6, atol=1e-7):
        data = _to_jsonable(data)
        if update or not path.exists():
            if not update:
                pytest.fail(f"missing snapshot {path.name}; run with --snapshot-update")
            SNAPSHOT_DIR.mkdir(exist_ok=True)
            path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            return
        _compare(data, json.loads(path.read_text()), stem, rtol, atol)

    return check
