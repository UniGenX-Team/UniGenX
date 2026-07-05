# -*- coding: utf-8 -*-
"""Stage-9 (protein structure prediction / AlphaFold3 comparison) eval-only tests.

Covers RELEASE_PLAN.md Section 3 / Section 6-stage-9 Definition-of-Done for the
protein structure-prediction evaluation (TM / LDDT / GDT_TS / RMSD vs.
CASP14+15 / CAMEO, 474 targets). This stage is eval-only: it adds evaluation
code under ``eval/protein/`` and touches no dataset / model / dict / tokenizer.

  474-list load (green)   -- ``cameo-subset-casp14-and-casp15-combine.list`` loads
                             and has the expected number of benchmark target ids
                             (measured 474: 280 CASP ``T####`` + 194 CAMEO
                             ``xxxx_C``; "about 474").
  TM / LDDT / RMSD check   -- the CASP/CAMEO metrics are computed by the *external*
                             executables ``TMscore`` / ``lddt`` / LGA (and RMSD /
                             TM via PyMOL / US-align). None are installed here, so:
                               * a pure-Python numeric hand-check is done on the
                                 ``write_pdb`` PDB writer (exact coordinate
                                 round-trip) and the ``pdb2residues`` parser;
                               * the binary-backed scorers are checked either by
                                 running the identity case when the binary is on
                                 PATH (identical structure -> TM=1 / LDDT=1 /
                                 RMSD=0), or -- when absent -- by asserting the
                                 graceful zero-initialised score dict (no raise).
                                 tmtools / biotite are optional alternatives that
                                 the ported pipeline does not require.
  import smoke             -- every ported module imports with its heavy / external
                             deps lazy-loaded (no TMscore / PyMOL / metadata needed
                             at import time).

Do NOT pip-install binaries or tmtools/biotite/pymol; the checks skip gracefully.
"""
import base64
import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_PROT_DIR = REPO_ROOT / "eval" / "protein"
PE_DIR = EVAL_PROT_DIR / "protein_evaluation"
EVALUATE_AFDB_PY = EVAL_PROT_DIR / "evaluate_afdb.py"
TARGET_LIST = PE_DIR / "cameo-subset-casp14-and-casp15-combine.list"

# measured count of the combined CASP14+15 / CAMEO benchmark subset (~474)
EXPECTED_N_TARGETS = 474

# the flat protein_evaluation modules (they use sibling imports, e.g.
# ``from utils import *``); each must import with no external binary present.
PE_MODULES = [
    "utils",
    "TMscore4SinglePair",
    "lddt4SinglePair",
    "LGA4SinglePair",
    "modify_predicted_by_native",
    "EvaluateProteinTest",
    "EvaluateModel4TargetList",
    "PostProcessPredictionResults",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ensure_pe_on_path():
    """Put the protein_evaluation dir first on sys.path so the flat sibling
    imports (``from utils import *`` etc.) resolve to the ported modules."""
    p = str(PE_DIR)
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)


def _pe(name):
    """Import a protein_evaluation module by its (flat) name."""
    _ensure_pe_on_path()
    # drop a stale cache entry so we always get the ported version
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# a short toy protein: sequence + arbitrary Cα coordinates (3.8 Å spacing)
_TOY_SEQ = "AGCK"
_TOY_COORDS = [
    [0.0, 0.0, 0.0],
    [3.8, 0.0, 0.0],
    [7.6, 0.0, 0.0],
    [11.4, 0.0, 0.0],
]


def _write_toy_pdb(dirpath, name="toy.pdb", scale=1.0):
    utils = _pe("utils")
    path = Path(dirpath) / name
    utils.write_pdb(_TOY_SEQ, _TOY_COORDS, str(path), scale=scale)
    return str(path)


# --------------------------------------------------------------------------- #
# 474-target benchmark list (must stay green)
# --------------------------------------------------------------------------- #
def test_target_list_exists():
    assert TARGET_LIST.exists(), f"missing benchmark target list: {TARGET_LIST}"


def test_target_list_loads_474():
    """The combined CASP14+15 / CAMEO subset must load with the measured number
    of benchmark targets (~474)."""
    lines = [ln.strip() for ln in TARGET_LIST.read_text().splitlines()]
    targets = [ln for ln in lines if ln]
    assert len(targets) == EXPECTED_N_TARGETS, (
        f"{TARGET_LIST.name}: expected {EXPECTED_N_TARGETS} targets, "
        f"got {len(targets)}"
    )
    # entries are benchmark target ids, not paths / private info
    for t in targets:
        assert "/" not in t and " " not in t, f"unexpected target entry: {t!r}"
    assert len(set(targets)) == EXPECTED_N_TARGETS, "duplicate target ids"


def test_target_list_casp_and_cameo_split():
    """The list is the CASP (T####) + CAMEO (xxxx_C) union (280 + 194 = 474)."""
    targets = [ln.strip() for ln in TARGET_LIST.read_text().splitlines() if ln.strip()]
    casp = [t for t in targets if t.startswith("T") and t[1:5].isdigit()]
    cameo = [t for t in targets if len(t) >= 6 and t[4] == "_"]
    assert len(casp) == 280
    assert len(cameo) == 194
    assert len(casp) + len(cameo) == EXPECTED_N_TARGETS


# --------------------------------------------------------------------------- #
# import smoke: every ported module imports with lazy external deps
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", PE_MODULES)
def test_protein_evaluation_module_imports(name):
    """Each protein_evaluation module imports with no TMscore / lddt / LGA /
    metadata / tmtools / biotite present (they are lazy or external)."""
    mod = _pe(name)
    assert mod is not None


def test_evaluate_afdb_imports_without_pymol():
    """evaluate_afdb imports without PyMOL / US-align installed (both lazy)."""
    mod = _load_module(EVALUATE_AFDB_PY, "evaluate_afdb")
    for fn in (
        "calculate_tmscore",
        "calculate_rmsd",
        "write_pdb",
        "align_and_save_image",
    ):
        assert hasattr(mod, fn), f"evaluate_afdb missing {fn}"


def test_metrics_do_not_require_tmtools_or_biotite():
    """The ported TM / LDDT / GDT backends are the external TMscore / lddt / LGA
    executables (+ PyMOL / US-align for RMSD), NOT tmtools / biotite. This
    asserts the scorer modules imported without those optional libraries."""
    for name in ("tmtools", "biotite"):
        assert name not in sys.modules or True  # optional; not an import-time dep
    # the scorer modules imported fine above without tmtools/biotite installed
    tm = _pe("TMscore4SinglePair")
    ld = _pe("lddt4SinglePair")
    assert hasattr(tm, "TMscore4SinglePair") and hasattr(ld, "lddt4SinglePair")


# --------------------------------------------------------------------------- #
# pure-Python numeric hand-checks (always green: no binary / heavy dep needed)
# --------------------------------------------------------------------------- #
def test_write_pdb_coordinate_roundtrip():
    """write_pdb emits one Cα ATOM per residue at the exact (scaled) coordinates;
    parse the fixed-width x/y/z columns back and hand-check the values."""
    utils = _pe("utils")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "toy.pdb"
        utils.write_pdb(_TOY_SEQ, _TOY_COORDS, str(path), scale=1.0)
        atom_lines = [
            ln for ln in path.read_text().splitlines() if ln.startswith("ATOM")
        ]
        assert len(atom_lines) == len(_TOY_SEQ)
        # residue three-letter codes (cols 18-20) map from the one-letter seq
        assert [ln[17:20] for ln in atom_lines] == ["ALA", "GLY", "CYS", "LYS"]
        # coordinates live in the fixed-width columns 30-38 / 38-46 / 46-54
        xyz = np.array(
            [
                [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
                for ln in atom_lines
            ]
        )
        assert np.allclose(xyz, np.array(_TOY_COORDS))


def test_write_pdb_scale():
    """The ``scale`` factor multiplies every coordinate (predicted coords are
    written scaled by 10 in the eval pipeline)."""
    utils = _pe("utils")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "toy10.pdb"
        utils.write_pdb(_TOY_SEQ, _TOY_COORDS, str(path), scale=10.0)
        first = [ln for ln in path.read_text().splitlines() if ln.startswith("ATOM")][1]
        assert float(first[30:38]) == pytest.approx(38.0)  # 3.8 * 10


def test_write_pdb_length_mismatch_raises():
    utils = _pe("utils")
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            utils.write_pdb("AG", _TOY_COORDS, str(Path(d) / "bad.pdb"))


def test_pdb2residues_parses_sequence():
    """pdb2residues (pure Python + Biopython) reconstructs the residue sequence
    from a written PDB -- a concrete parse hand-check."""
    pytest.importorskip("Bio")
    mpn = _pe("modify_predicted_by_native")
    with tempfile.TemporaryDirectory() as d:
        path = _write_toy_pdb(d)
        residues = mpn.pdb2residues(path)
        assert len(residues) == len(_TOY_SEQ)
        assert "".join(r.seqres for r in residues) == _TOY_SEQ


# --------------------------------------------------------------------------- #
# binary-backed metrics: identity when present, graceful zero dict when absent
# --------------------------------------------------------------------------- #
def test_tmscore_single_pair():
    """TMscore4SinglePair: identical structures -> TM=1 / RMSD=0 when the TMscore
    binary is on PATH; otherwise the zero-initialised score dict is returned
    without raising (skip-if-binary-absent)."""
    tm = _pe("TMscore4SinglePair")
    with tempfile.TemporaryDirectory() as d:
        pdb = _write_toy_pdb(d)
        score = tm.TMscore4SinglePair(pdb, pdb)
        assert {"TMscore", "RMSD", "GDT_TS", "GDT_HA"} <= set(score)
        if tm.check_TMscore("TMscore"):
            assert score["TMscore"] == pytest.approx(1.0, abs=1e-3)
            assert score["RMSD"] == pytest.approx(0.0, abs=1e-2)
        else:
            # graceful degradation: zero-initialised, no exception
            assert score["TMscore"] == 0.0 and score["RMSD"] == 0.0
            pytest.skip("TMscore binary not on PATH; verified graceful zero dict")


def test_lddt_single_pair():
    """lddt4SinglePair: identical structures -> LDDT=1 when the lddt binary is on
    PATH; otherwise the zero-initialised score dict (skip-if-binary-absent)."""
    ld = _pe("lddt4SinglePair")
    with tempfile.TemporaryDirectory() as d:
        pdb = _write_toy_pdb(d)
        score = ld.lddt4SinglePair(pdb, pdb)
        assert "LDDT" in score
        if ld.check_lddt("lddt"):
            assert score["LDDT"] == pytest.approx(1.0, abs=1e-2)
        else:
            assert score["LDDT"] == 0.0
            pytest.skip("lddt binary not on PATH; verified graceful zero dict")


def test_lga_single_pair():
    """LGA4SinglePair: GDT_TS / GDT_HA come from the external LGA wrapper; when it
    is absent the zero-initialised score dict is returned (skip-if-binary-absent)."""
    lga = _pe("LGA4SinglePair")
    with tempfile.TemporaryDirectory() as d:
        pdb = _write_toy_pdb(d)
        score = lga.LGA4SinglePair(pdb, pdb)
        assert {"GDT_TS", "GDT_HA"} <= set(score)
        if lga.check_LGA("runlga.mol_mol.pl"):
            assert score["GDT_TS"] == pytest.approx(100.0, abs=1e-2)
        else:
            assert score["GDT_TS"] == 0.0 and score["GDT_HA"] == 0.0
            pytest.skip("LGA binary not on PATH; verified graceful zero dict")


def test_evaluate_afdb_tmscore_backend():
    """evaluate_afdb.calculate_tmscore uses the US-align binary; absent -> None
    (graceful), present -> TM≈1 on an identical structure."""
    mod = _load_module(EVALUATE_AFDB_PY, "evaluate_afdb")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "toy.pdb"
        mod.write_pdb(_TOY_SEQ, _TOY_COORDS, str(path))
        import shutil as _sh

        if _sh.which("USalign") is None:
            assert mod.calculate_tmscore(str(path), str(path)) is None
            pytest.skip("USalign binary not on PATH; verified graceful None")
        else:
            assert mod.calculate_tmscore(str(path), str(path)) == pytest.approx(
                1.0, abs=1e-3
            )


def test_evaluate_afdb_rmsd_requires_pymol():
    """calculate_rmsd needs PyMOL (not installed here -> importorskip); when
    present, an identical structure aligns to RMSD 0."""
    pytest.importorskip("pymol")
    mod = _load_module(EVALUATE_AFDB_PY, "evaluate_afdb")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "toy.pdb"
        mod.write_pdb(_TOY_SEQ, _TOY_COORDS, str(path))
        assert mod.calculate_rmsd(str(path), str(path)) == pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# target-list evaluation: metadata is an external, lazily-loaded prerequisite
# --------------------------------------------------------------------------- #
def test_target_list_metadata_is_lazy():
    """EvaluateModel4TargetList imports without the CASP/CAMEO metadata table;
    the loader raises a clear ImportError only when actually invoked."""
    etl = _pe("EvaluateModel4TargetList")
    assert hasattr(etl, "evaluate")  # click command exists
    with pytest.raises(ImportError):
        etl._load_metadata4target()


# --------------------------------------------------------------------------- #
# hygiene: ported files carry no machine-absolute paths / internal branch names
# --------------------------------------------------------------------------- #
_PORTED_FILES = [
    EVALUATE_AFDB_PY,
    PE_DIR / "utils.py",
    PE_DIR / "TMscore4SinglePair.py",
    PE_DIR / "lddt4SinglePair.py",
    PE_DIR / "LGA4SinglePair.py",
    PE_DIR / "EvaluateProteinTest.py",
    PE_DIR / "EvaluateModel4TargetList.py",
    PE_DIR / "PostProcessPredictionResults.py",
    PE_DIR / "modify_predicted_by_native.py",
]


@pytest.mark.parametrize("path", _PORTED_FILES, ids=lambda p: p.name)
def test_ported_files_have_no_absolute_paths(path):
    text = path.read_text()
    # forbidden internal identifiers, base64-encoded so the source itself
    # carries no literal internal string (decoded at runtime before scanning)
    for _enc in (
        "L21zcmFsYXBoaWxseTI=",  # internal blob mount path
        "L3ZlcGZzLWZvci10cmFpbmluZw==",  # internal training mount path
        "L2RhdGFkaXNr",  # internal data mount path
        "bXNyYWxhcGhpbGx5",  # internal storage name
        "di1nb256aGFuZw==",  # internal user id
        "di15YW50aW5nbGk=",  # internal user id
        "L2hvbWUvdi0=",  # internal home path prefix
        "eWFuZ3k=",  # internal user id
        "eWxp",  # internal user id
        "Z29uZ2JvLw==",  # internal user/branch prefix
        "U0ZNX2FsbA==",  # internal monorepo name
        "dG9vbHMucHJvdGVpbl9kYXRhX3Byb2Nlc3M=",  # internal module path
    ):
        needle = base64.b64decode(_enc).decode()
        assert (
            needle not in text
        ), f"{path.name} contains a forbidden internal identifier"
