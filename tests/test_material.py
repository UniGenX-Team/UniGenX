# -*- coding: utf-8 -*-
"""Stage-2 (material CSP) alignment smoke tests.

Covers RELEASE_PLAN.md Section 3 Definition-of-Done for the crystal-structure
prediction (CSP) path:

  3. dict vocab assertion -- ``dict_mat.txt`` -> vocab 355 (non-empty lines + 7
                             special tokens), using only the committed dict (no
                             checkpoint needed). Plus a skip-if-present check that
                             a ``csp_*`` checkpoint's ``embed_tokens.weight`` has
                             dim0 == 355.
  4. collation            -- 3 self-made MP-20-style records fed through the
                             ``material`` target's ``get_train_item`` /
                             ``get_infer_item`` assert the crystal-domain
                             invariant: the lattice vectors occupy the FIRST 3
                             coordinate slots, the ``coordinates_mask`` layout is
                             ``[0]*(n+2) + [1]*(n+3) + [0]``, and the token
                             sequence has the expected shape.
  6. eval                 -- the CSP ``StructureMatcher`` thresholds
                             (LTOL/STOL/ANGLE_TOL) and the smact / structure
                             validity criteria on a hand-verifiable toy
                             (prediction == ground truth must match with rms ~ 0).
                             ``smact`` is not installed here, so that assertion
                             skips via ``importorskip``; the thresholds are also
                             pinned by a dependency-free source check.

The dict-vocab, collation and threshold-source tests stay green with no external
dependencies (fixtures are generated at runtime into a temp .jsonl, so nothing is
committed). The checkpoint test skips when the (multi-GB) ``csp_*`` checkpoints
are absent, and the smact eval test skips when ``smact`` / ``pymatgen`` are
unavailable.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "unigenx" / "data"
EVAL_CSP = REPO_ROOT / "eval" / "material" / "evaluate_csp.py"

# vocab = non-empty dict lines + 7 special tokens
# (<pad> <bos> <eos> <unk> prepended, <mask> <coord> <sg> appended).
DICT_MAT_VOCAB = 355

# Optional local checkpoints live one level above the repo (the internal release
# workspace: a checkpoints/ sibling of the package repo). A public clone won't
# have them, so the dependent test skips. Env vars override for other layouts. No
# machine-absolute paths are committed.
_WORKSPACE = REPO_ROOT.parent
_CKPT_DIR = Path(os.environ.get("UNIGENX_CHECKPOINTS", str(_WORKSPACE / "checkpoints")))
_CKPT_ARGS_PKG = os.environ.get("UNIGENX_CKPT_ARGS_PKG")
CSP_CHECKPOINTS = ["csp_carbon24.pt", "csp_mp20.pt", "csp_mpts52.pt"]

# Self-made MP-20-style crystals: cubic NaCl (2 sites), Li2O (3 sites), TiO2 (3).
# Each record carries the fields the material dataset path reads: id, formula,
# a 3x3 lattice, and sites with element + fractional coordinates.
_TOY_RECORDS = [
    {
        "id": 0,
        "formula": "Na1Cl1",
        "lattice": [[5.64, 0.0, 0.0], [0.0, 5.64, 0.0], [0.0, 0.0, 5.64]],
        "sites": [
            {"element": "Na", "fractional_coordinates": [0.0, 0.0, 0.0]},
            {"element": "Cl", "fractional_coordinates": [0.5, 0.5, 0.5]},
        ],
    },
    {
        "id": 1,
        "formula": "Li2O1",
        "lattice": [[4.61, 0.0, 0.0], [0.0, 4.61, 0.0], [0.0, 0.0, 4.61]],
        "sites": [
            {"element": "Li", "fractional_coordinates": [0.25, 0.25, 0.25]},
            {"element": "Li", "fractional_coordinates": [0.75, 0.75, 0.75]},
            {"element": "O", "fractional_coordinates": [0.0, 0.0, 0.0]},
        ],
    },
    {
        "id": 2,
        "formula": "Ti1O2",
        "lattice": [[4.59, 0.0, 0.0], [0.0, 4.59, 0.0], [0.0, 0.0, 2.96]],
        "sites": [
            {"element": "Ti", "fractional_coordinates": [0.0, 0.0, 0.0]},
            {"element": "O", "fractional_coordinates": [0.3, 0.3, 0.0]},
            {"element": "O", "fractional_coordinates": [0.7, 0.7, 0.0]},
        ],
    },
]


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def toy_jsonl(tmp_path_factory):
    """Write the self-made records to a temp .jsonl (not committed; *.jsonl is
    intentionally gitignored as an output format)."""
    path = tmp_path_factory.mktemp("material") / "mp20_toy.jsonl"
    with open(path, "w") as f:
        for rec in _TOY_RECORDS:
            f.write(json.dumps(rec) + "\n")
    return str(path)


def _material_config(target="material"):
    """Build a minimal UniGenXConfig for the material/uni_mat collation path.

    Fields are set as attributes so this stays robust regardless of which are
    declared dataclass fields; ``space_group`` in particular is injected at
    runtime by ``unigenx_infer.py`` via ``saved_config.update(...)``, and CSP
    runs with ``--no_space_group`` (see scripts/gen_mat.sh).
    """
    from unigenx.model.config import UniGenXConfig

    cfg = UniGenXConfig()
    cfg.target = target
    cfg.space_group = False
    cfg.reorder = False
    cfg.rotation_augmentation = False
    cfg.translation_augmentation = False
    cfg.scale_coords = None
    cfg.max_sites = None
    cfg.tokenizer = "num"
    return cfg


def _load_dataset(path, mode, target="material"):
    from unigenx.data.dataset import MODE, UniGenXDataset
    from unigenx.data.tokenizer import UniGenXTokenizer

    cfg = _material_config(target)
    tok = UniGenXTokenizer.from_file(str(DATA_DIR / "dict_mat.txt"), cfg)
    m = MODE.TRAIN if mode == "train" else MODE.INFER
    ds = UniGenXDataset(tok, path, args=cfg, shuffle=False, mode=m)
    return ds, tok


# --------------------------------------------------------------------------- #
# DoD 3: dict vocab assertion (committed dict, no checkpoint needed)
# --------------------------------------------------------------------------- #
def test_dict_mat_vocab():
    from unigenx.data.tokenizer import UniGenXTokenizer

    path = DATA_DIR / "dict_mat.txt"
    assert path.exists(), f"missing committed dict: {path}"
    tok = UniGenXTokenizer.from_file(str(path))
    assert len(tok) == DICT_MAT_VOCAB, (
        f"dict_mat.txt: expected vocab {DICT_MAT_VOCAB}, got {len(tok)} "
        "(vocab must equal non-empty dict lines + 7 special tokens)"
    )


# --------------------------------------------------------------------------- #
# DoD 3 (additional): csp_* checkpoint embedding vocab, skip-if-absent
# --------------------------------------------------------------------------- #
def test_csp_checkpoint_embedding_vocab():
    present = [_CKPT_DIR / n for n in CSP_CHECKPOINTS if (_CKPT_DIR / n).exists()]
    if not present:
        pytest.skip(f"no csp_* checkpoints under {_CKPT_DIR}")
    import torch

    def _load(p):
        return torch.load(str(p), map_location="cpu", weights_only=False, mmap=True)

    for ckpt in present:
        try:
            try:
                state = _load(ckpt)
            except ModuleNotFoundError:
                # Unpickling the saved args needs the internal training package
                # on the path; add it and retry, else skip rather than fail.
                if _CKPT_ARGS_PKG and _CKPT_ARGS_PKG not in sys.path:
                    sys.path.insert(0, _CKPT_ARGS_PKG)
                state = _load(ckpt)
        except Exception as e:  # pragma: no cover - environment dependent
            pytest.skip(f"cannot load {ckpt.name} ({type(e).__name__}: {e})")

        container = state
        if isinstance(state, dict):
            for key in ("model", "module", "state_dict"):
                if key in state and isinstance(state[key], dict):
                    container = state[key]
                    break

        # Search by suffix (the key carries net./model. prefixes) instead of
        # assuming a fixed key name.
        matches = [
            k
            for k in container
            if isinstance(k, str) and k.endswith("embed_tokens.weight")
        ]
        assert matches, f"{ckpt.name}: no *embed_tokens.weight in state dict"
        for k in matches:
            assert container[k].shape[0] == DICT_MAT_VOCAB, (
                f"{ckpt.name}:{k} embedding dim0 {container[k].shape[0]} "
                f"!= {DICT_MAT_VOCAB} (csp_* must map to dict_mat, vocab 355)"
            )


# --------------------------------------------------------------------------- #
# DoD 4: collation -- lattice occupies the first 3 coordinate slots
# --------------------------------------------------------------------------- #
def test_material_train_collation(toy_jsonl):
    """TRAIN mode: coordinates array carries the lattice in its first 3 rows."""
    ds, tok = _load_dataset(toy_jsonl, "train")
    assert len(ds.data) == len(_TOY_RECORDS), "all records pass the length filter"

    for idx in range(len(ds.data)):
        n = len(ds.data[idx]["sites"])
        item = ds.get_train_item(idx)
        assert {"tokens", "coordinates", "coordinates_mask"} <= set(item)

        toks = item["tokens"]
        coords = item["coordinates"]
        mask = item["coordinates_mask"]

        # token sequence: <bos> [n elems] <coord> [3 mask] [n mask] <eos>
        assert len(toks) == 2 * n + 6
        assert len(mask) == 2 * n + 6
        assert toks[0] == tok.bos_idx
        assert toks[n + 1] == tok.coord_idx
        assert toks[-1] == tok.eos_idx
        # the 3 lattice slots and the n atom slots are <mask> in the input ids
        assert all(t == tok.mask_idx for t in toks[n + 2 : 2 * n + 5])

        # ---- the crystal-domain invariant ----
        # coordinates = concatenate([lattice(3x3), fractional atom coords(n x 3)])
        assert coords.shape == (n + 3, 3)
        lattice = np.array(ds.data[idx]["lattice"], dtype=np.float32)
        assert np.allclose(
            coords[:3], lattice
        ), "lattice vectors must occupy the first 3 coordinate slots"

        # coordinates_mask marks exactly the 3 lattice + n atom slots
        assert int(mask.sum()) == n + 3
        expected_mask = np.array([0] * (n + 2) + [1] * (n + 3) + [0])
        assert np.array_equal(mask, expected_mask)


def test_material_infer_collation(toy_jsonl):
    """INFER mode: prompt stops at <coord>; the mask spans the slots to fill."""
    ds, tok = _load_dataset(toy_jsonl, "infer")
    assert len(ds.data) == len(_TOY_RECORDS)

    for idx in range(len(ds.data)):
        n = len(ds.data[idx]["sites"])
        item = ds.get_infer_item(idx)
        assert "tokens" in item and "coordinates_mask" in item
        # coordinates are generated, not provided, at inference time
        assert "coordinates" not in item

        toks = item["tokens"]
        mask = item["coordinates_mask"]

        # inference prompt: <bos> [n elems] <coord>  (length n + 2)
        assert len(toks) == n + 2
        assert toks[0] == tok.bos_idx
        assert toks[-1] == tok.coord_idx
        # every element token resolved (none fell through to <unk>)
        assert tok.unk_idx not in list(toks[1 : n + 1])

        # the mask spans the full generated layout: 3 lattice + n atom slots
        assert len(mask) == 2 * n + 6
        assert int(mask.sum()) == n + 3
        expected_mask = np.array([0] * (n + 2) + [1] * (n + 3) + [0])
        assert np.array_equal(mask, expected_mask)


def test_material_collate_batch(toy_jsonl):
    """collate() batches infer items; mask is wider than input_ids (fill-rest)."""
    ds, _ = _load_dataset(toy_jsonl, "infer")
    items = [ds.get_infer_item(i) for i in range(len(ds.data))]
    batch = ds.collate(items)

    assert {"input_ids", "coordinates_mask", "attention_mask"} <= set(batch)
    bs = len(items)
    assert batch["input_ids"].shape[0] == bs
    assert batch["coordinates_mask"].shape[0] == bs
    # generate-the-rest: coordinates_mask is at least as wide as the prompt ids
    assert batch["coordinates_mask"].shape[1] >= batch["input_ids"].shape[1]
    # no ground-truth coordinates are fed at inference time
    assert "input_coordinates" not in batch


def test_uni_mat_lattice_first_three_slots(toy_jsonl):
    """uni_mat shares the crystal invariant: lattice in the first 3 slots.

    uni_mat is the stage-1 unified-model material path; this only asserts the
    shared collation invariant, it does not modify that path.
    """
    ds, tok = _load_dataset(toy_jsonl, "train", target="uni_mat")
    for idx in range(len(ds.data)):
        n = len(ds.data[idx]["sites"])
        item = ds.get_train_item(idx)
        coords = item["coordinates"]
        assert coords.shape == (n + 3, 3)
        lattice = np.array(ds.data[idx]["lattice"], dtype=np.float32)
        assert np.allclose(coords[:3], lattice)
        assert int(item["coordinates_mask"].sum()) == n + 3


# --------------------------------------------------------------------------- #
# DoD 6: eval criteria -- StructureMatcher thresholds + smact/structure validity
# --------------------------------------------------------------------------- #
def test_eval_csp_thresholds_source():
    """Pin the paper thresholds without importing (smact-free): values + wiring."""
    text = EVAL_CSP.read_text()
    assert "LTOL = 0.3" in text
    assert "STOL = 0.5" in text
    assert "ANGLE_TOL = 10" in text
    assert "StructureMatcher(stol=STOL, angle_tol=ANGLE_TOL, ltol=LTOL)" in text
    # smact charge-neutrality + Pauling electronegativity criteria
    assert "pauling_test" in text
    assert "def smact_validity" in text
    assert "def structure_validity" in text
    # top-N best-of match-rate mode
    assert "--multiple" in text


def test_eval_csp_match_and_validity():
    """Hand-verifiable toy: prediction == ground truth must match (rms ~ 0).

    ``evaluate_csp.py`` imports smact at module scope and parses argv at import
    time, so it is loaded via importlib with argv patched, behind importorskip.
    smact is not installed here, so this test skips.
    """
    pytest.importorskip("pymatgen")
    pytest.importorskip("smact")

    spec = importlib.util.spec_from_file_location("evaluate_csp", str(EVAL_CSP))
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    # positional "input" + --valid (type=bool: any non-empty string -> True)
    sys.argv = ["evaluate_csp", "toy.jsonl", "--valid", "1"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old_argv

    # thresholds
    assert mod.LTOL == 0.3 and mod.STOL == 0.5 and mod.ANGLE_TOL == 10

    # charge-neutral ionic composition is smact-valid
    assert mod.smact_validity(("Na", "Cl"), (1, 1)) is True

    # a structure matches itself with rms ~ 0 under the CSP thresholds
    lattice = mod.Lattice([[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]])
    struct = mod.Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    matcher = mod.StructureMatcher(
        stol=mod.STOL, angle_tol=mod.ANGLE_TOL, ltol=mod.LTOL
    )
    rms = matcher.get_rms_dist(struct, struct)
    assert rms is not None and rms[0] < 1e-6
    assert mod.structure_validity(struct) is True

    # evaluate_singe on a record whose prediction == ground truth
    record = {
        "sites": [
            {"element": "Na", "fractional_coordinates": [0.0, 0.0, 0.0]},
            {"element": "Cl", "fractional_coordinates": [0.5, 0.5, 0.5]},
        ],
        "lattice": [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
        "prediction": {
            "lattice": [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
            "coordinates": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        },
    }
    record = json.loads(json.dumps(record))  # jsonl round-trip
    rms_dist, is_valid, comp_valid, struct_valid, p1 = mod.evaluate_singe(
        record, None, matcher
    )
    assert rms_dist is not None and rms_dist < 1e-6
    assert is_valid is True
