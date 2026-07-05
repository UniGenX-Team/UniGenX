# -*- coding: utf-8 -*-
"""Stage-4 (molecular conformer ensemble) alignment smoke tests.

Covers RELEASE_PLAN.md Section 3 Definition-of-Done for the unconditional
molecule path (``--target mol`` for both GEOM-QM9 and GEOM-Drugs, which reuse
the same ``mol`` path with different dicts) plus the shared unified molecule
path (``uni_mol``):

  3. dict vocab assertion -- ``vocab == non-empty dict lines + 7`` for the two
                             molecule dicts (dict_qm9 -> 27, dict_drugs -> 38),
                             using only the committed dicts (no checkpoint
                             needed). Plus a skip-if-present check that the
                             ``mol_qm9`` / ``mol_drugs`` checkpoints'
                             ``embed_tokens.weight`` have dim0 == 27 / 38.
  4. collation            -- self-made SMILES records fed through the ``mol``
                             target's ``get_train_item`` / ``get_infer_item``
                             assert the MOLECULE-domain invariant: coordinates
                             carry ONLY atom coordinates (n rows, not n+3) --
                             there is NO lattice prefix, unlike the crystal path
                             where the first 3 coordinate slots are the lattice
                             vectors. Element/SMILES tokens resolve without
                             falling through to ``<unk>`` (incl. the two-char
                             ``Cl`` / ``Br`` merge in the Drugs dict).
  6. eval                 -- COV / MAT (recall + precision) via RDKit
                             ``GetBestRMS`` on hand-verifiable toys: identical
                             conformers give COV=100 / MAT~0; the SAME distorted
                             conformer is rejected at QM9's 0.5 A threshold but
                             accepted at Drugs' 1.25 A threshold; recall vs
                             precision differ on a mixed gen set. RDKit is a hard
                             dependency of this repo, so this test really runs.

The dict-vocab, collation and COV/MAT tests stay green with no external
resources (fixtures are generated at runtime into a temp .jsonl, so nothing is
committed). The checkpoint test skips when the (multi-GB) ``mol_*`` checkpoints
are absent.
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
EVAL_MOL = REPO_ROOT / "eval" / "molecule" / "evaluate_mol.py"

# vocab = non-empty dict lines + 7 special tokens
# (<pad> <bos> <eos> <unk> prepended, <mask> <coord> <sg> appended).
DICT_VOCAB = {"dict_qm9.txt": 27, "dict_drugs.txt": 38}

# Optional local checkpoints live one level above the repo (the internal release
# workspace: a checkpoints/ sibling of the package repo). A public clone won't
# have them, so the dependent test skips. Env vars override for other layouts.
# No machine-absolute paths are committed.
_WORKSPACE = REPO_ROOT.parent
_CKPT_DIR = Path(os.environ.get("UNIGENX_CHECKPOINTS", str(_WORKSPACE / "checkpoints")))
_CKPT_ARGS_PKG = os.environ.get("UNIGENX_CKPT_ARGS_PKG")
# checkpoint file -> expected embedding vocab (see DICT_MAP.md / RELEASE_PLAN.md
# Section 4: mol_qm9 -> dict_qm9 (27), mol_drugs -> dict_drugs (38)). c_mol.pt is
# the conditional stage-5 checkpoint and is intentionally NOT covered here.
MOL_CHECKPOINTS = {"mol_qm9.pt": 27, "mol_drugs.pt": 38}

# CCO (ethanol skeleton, 3 heavy atoms) with an explicit, deterministic
# conformer -- no embedding randomness, so RMSDs are reproducible.
_BASE_COORDS = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.5, 1.0, 0.0)]

# Self-made molecule records. Each carries the fields the mol dataset path reads:
#   train : id, smi, pos (n x 3 atom coords)
#   infer : id, smi, num (atom count)
# get_sequence_length() reads smi + num for every record regardless of mode, so
# num is always present and equals len(pos).
_QM9_RECORDS = [
    {
        "id": 0,
        "smi": "CCO",
        "num": 3,
        "pos": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 1.0, 0.0]],
    },
    {"id": 1, "smi": "CC", "num": 2, "pos": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]},
    {
        "id": 2,
        "smi": "OCC=O",
        "num": 4,
        "pos": [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.1, 1.2, 0.0], [3.4, 1.1, 0.0]],
    },
]
# Drugs records exercise the two-char element merge (Cl, Br) that the SMILES
# tokenizer builds by folding a lowercase char into the previous token.
_DRUGS_RECORDS = [
    {
        "id": 0,
        "smi": "CCCl",
        "num": 3,
        "pos": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
    },
    {
        "id": 1,
        "smi": "CCBr",
        "num": 3,
        "pos": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.1, 0.0, 0.0]],
    },
    {
        "id": 2,
        "smi": "CCO",
        "num": 3,
        "pos": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 1.0, 0.0]],
    },
]


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _write_jsonl(tmp_path_factory, name, records):
    path = tmp_path_factory.mktemp("molecule") / name
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def qm9_jsonl(tmp_path_factory):
    """QM9-style records -> temp .jsonl (not committed; *.jsonl is gitignored)."""
    return _write_jsonl(tmp_path_factory, "qm9_toy.jsonl", _QM9_RECORDS)


@pytest.fixture(scope="module")
def drugs_jsonl(tmp_path_factory):
    return _write_jsonl(tmp_path_factory, "drugs_toy.jsonl", _DRUGS_RECORDS)


def _mol_config(target="mol"):
    """Build a minimal UniGenXConfig for the mol / uni_mol collation path."""
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


def _load_dataset(path, mode, target="mol", dict_name="dict_qm9.txt"):
    from unigenx.data.dataset import MODE, UniGenXDataset
    from unigenx.data.tokenizer import UniGenXTokenizer

    cfg = _mol_config(target)
    tok = UniGenXTokenizer.from_file(str(DATA_DIR / dict_name), cfg)
    m = MODE.TRAIN if mode == "train" else MODE.INFER
    ds = UniGenXDataset(tok, path, args=cfg, shuffle=False, mode=m)
    return ds, tok


def _mol_with_conformer(smi, coords):
    """A molecule with one explicit conformer at the given coordinates."""
    from rdkit import Chem

    m = Chem.MolFromSmiles(smi)
    assert m is not None and m.GetNumAtoms() == len(coords)
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, c in enumerate(coords):
        conf.SetAtomPosition(i, tuple(float(v) for v in c))
    m.AddConformer(conf, assignId=True)
    return m


@pytest.fixture(scope="module")
def eval_mol():
    """Load eval/molecule/evaluate_mol.py by path.

    The argparse / pickle-loading lives under ``if __name__ == '__main__'``, so
    importing the module (name != '__main__') only defines the metric functions
    and does not touch the filesystem or sys.argv. RDKit is a declared repo
    dependency, so this is expected to run (not skip)."""
    pytest.importorskip("rdkit")
    spec = importlib.util.spec_from_file_location("evaluate_mol", str(EVAL_MOL))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# DoD 3: dict vocab assertions (committed dicts, no checkpoint needed)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dict_name,expected", sorted(DICT_VOCAB.items()))
def test_dict_vocab(dict_name, expected):
    from unigenx.data.tokenizer import UniGenXTokenizer

    path = DATA_DIR / dict_name
    assert path.exists(), f"missing committed dict: {path}"
    tok = UniGenXTokenizer.from_file(str(path))
    assert len(tok) == expected, (
        f"{dict_name}: expected vocab {expected}, got {len(tok)} "
        "(vocab must equal non-empty dict lines + 7 special tokens)"
    )


# --------------------------------------------------------------------------- #
# DoD 3 (additional): mol_* checkpoint embedding vocab, skip-if-absent
# --------------------------------------------------------------------------- #
def test_mol_checkpoint_embedding_vocab():
    present = {n: v for n, v in MOL_CHECKPOINTS.items() if (_CKPT_DIR / n).exists()}
    if not present:
        pytest.skip(f"no mol_* checkpoints under {_CKPT_DIR}")
    import torch

    def _load(p):
        return torch.load(str(p), map_location="cpu", weights_only=False, mmap=True)

    for name, expected in present.items():
        ckpt = _CKPT_DIR / name
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
            pytest.skip(f"cannot load {name} ({type(e).__name__}: {e})")

        container = state
        if isinstance(state, dict):
            for key in ("model", "module", "state_dict"):
                if key in state and isinstance(state[key], dict):
                    container = state[key]
                    break

        # Search by suffix (the key carries net./model./module. prefixes) rather
        # than assuming a fixed key name.
        matches = [
            k
            for k in container
            if isinstance(k, str) and k.endswith("embed_tokens.weight")
        ]
        assert matches, f"{name}: no *embed_tokens.weight in state dict"
        for k in matches:
            assert container[k].shape[0] == expected, (
                f"{name}:{k} embedding dim0 {container[k].shape[0]} != {expected} "
                "(mol_qm9 -> dict_qm9/27, mol_drugs -> dict_drugs/38)"
            )


# --------------------------------------------------------------------------- #
# DoD 4: collation -- molecules carry NO lattice slots (unlike crystals)
# --------------------------------------------------------------------------- #
def _coord_pos(toks, tok):
    return list(toks).index(tok.coord_idx)


@pytest.mark.parametrize("dict_name", sorted(DICT_VOCAB.keys()))
def test_mol_train_collation(qm9_jsonl, drugs_jsonl, dict_name):
    """TRAIN mode: coordinates are the n atom coords only -- no lattice prefix."""
    path = qm9_jsonl if dict_name == "dict_qm9.txt" else drugs_jsonl
    ds, tok = _load_dataset(path, "train", target="mol", dict_name=dict_name)
    assert len(ds.data) == 3, "all records pass the length filter"

    for idx in range(len(ds.data)):
        num = ds.data[idx]["num"]
        item = ds.get_train_item(idx)
        assert {"tokens", "coordinates", "coordinates_mask"} <= set(item)

        toks = item["tokens"]
        coords = item["coordinates"]
        mask = item["coordinates_mask"]

        # token layout: <bos> [smiles toks] <coord> [num <mask>] <eos>
        cp = _coord_pos(toks, tok)
        n_smi = cp - 1
        assert toks[0] == tok.bos_idx
        assert toks[cp] == tok.coord_idx
        assert toks[-1] == tok.eos_idx
        assert len(toks) == n_smi + num + 3
        # the num coordinate slots are <mask> in the input ids
        assert all(t == tok.mask_idx for t in toks[cp + 1 : cp + 1 + num])
        # every SMILES/element token resolved (none fell through to <unk>)
        assert tok.unk_idx not in list(toks[1:cp])

        # ---- the MOLECULE-domain invariant (contrast with crystals) ----
        # coordinates = atom coords only, n rows -- there is NO 3-row lattice
        # prefix. A crystal record here would have coords.shape == (n + 3, 3).
        assert coords.shape == (
            num,
            3,
        ), "mol coordinates must be exactly n atom rows (no lattice prefix)"

        # coordinates_mask marks exactly the num atom slots
        assert int(mask.sum()) == num
        expected_mask = np.array([0] * (cp + 1) + [1] * num + [0])
        assert np.array_equal(mask, expected_mask)


@pytest.mark.parametrize("dict_name", sorted(DICT_VOCAB.keys()))
def test_mol_infer_collation(qm9_jsonl, drugs_jsonl, dict_name):
    """INFER mode: prompt stops at <coord>; the mask spans the atom slots."""
    path = qm9_jsonl if dict_name == "dict_qm9.txt" else drugs_jsonl
    ds, tok = _load_dataset(path, "infer", target="mol", dict_name=dict_name)
    assert len(ds.data) == 3

    for idx in range(len(ds.data)):
        num = ds.data[idx]["num"]
        item = ds.get_infer_item(idx)
        assert "tokens" in item and "coordinates_mask" in item
        # coordinates are generated, not provided, at inference time
        assert "coordinates" not in item

        toks = item["tokens"]
        mask = item["coordinates_mask"]

        # inference prompt: <bos> [smiles toks] <coord>
        assert toks[0] == tok.bos_idx
        assert toks[-1] == tok.coord_idx
        n_smi = len(toks) - 2
        assert tok.unk_idx not in list(toks[1 : 1 + n_smi])

        # mask spans the prompt + num atom slots (fill-the-rest), no lattice slots
        assert len(mask) == len(toks) + num
        assert int(mask.sum()) == num
        expected_mask = np.array([0] * len(toks) + [1] * num)
        assert np.array_equal(mask, expected_mask)


def test_drugs_two_char_element_tokens(drugs_jsonl):
    """The Cl / Br two-char elements must resolve to real Drugs-dict tokens."""
    ds, tok = _load_dataset(
        drugs_jsonl, "infer", target="mol", dict_name="dict_drugs.txt"
    )
    assert tok.get_idx("Cl") != tok.unk_idx
    assert tok.get_idx("Br") != tok.unk_idx
    # record 0 is "CCCl": tokens are [<bos>, C, C, Cl, <coord>]
    item = ds.get_infer_item(0)
    toks = list(item["tokens"])
    cp = _coord_pos(np.array(toks), tok)
    assert toks[1:cp] == [tok.get_idx("C"), tok.get_idx("C"), tok.get_idx("Cl")]
    assert tok.unk_idx not in toks[1:cp]


def test_mol_collate_batch(qm9_jsonl):
    """collate() batches items; mask is wider than input_ids (fill-the-rest)."""
    ds, _ = _load_dataset(qm9_jsonl, "infer", target="mol")
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


def test_mol_has_no_lattice_slots_unlike_material(qm9_jsonl, tmp_path):
    """Explicit contrast: the crystal path prepends 3 lattice rows; mol does not.

    Same collation machinery, two targets: material coordinates are (n + 3, 3)
    with the lattice occupying the first 3 slots, while mol coordinates are
    (n, 3) with atom coordinates only. This is the defining molecular-domain
    invariant for stage 4.
    """
    # mol side
    ds_mol, _ = _load_dataset(qm9_jsonl, "train", target="mol")
    for idx in range(len(ds_mol.data)):
        num = ds_mol.data[idx]["num"]
        coords = ds_mol.get_train_item(idx)["coordinates"]
        assert coords.shape == (num, 3)  # no lattice prefix

    # material side (single self-made crystal record)
    mat_rec = {
        "id": 0,
        "formula": "Na1Cl1",
        "lattice": [[5.64, 0.0, 0.0], [0.0, 5.64, 0.0], [0.0, 0.0, 5.64]],
        "sites": [
            {"element": "Na", "fractional_coordinates": [0.0, 0.0, 0.0]},
            {"element": "Cl", "fractional_coordinates": [0.5, 0.5, 0.5]},
        ],
    }
    mat_path = tmp_path / "mat_toy.jsonl"
    with open(mat_path, "w") as f:
        f.write(json.dumps(mat_rec) + "\n")
    ds_mat, _ = _load_dataset(
        str(mat_path), "train", target="material", dict_name="dict_mat.txt"
    )
    n = len(ds_mat.data[0]["sites"])
    mat_coords = ds_mat.get_train_item(0)["coordinates"]
    assert mat_coords.shape == (n + 3, 3), "material DOES have the 3 lattice slots"
    lattice = np.array(mat_rec["lattice"], dtype=np.float32)
    assert np.allclose(mat_coords[:3], lattice)


def test_uni_mol_no_lattice_slots(qm9_jsonl):
    """uni_mol (stage-1 unified molecule path) shares the no-lattice invariant.

    This only asserts the shared collation invariant for the molecule side of
    the unified model; it does not modify that stage-1 path. The <molecule> flag
    token and <s>-prefixed SMILES tokens require the unified dict, so this uses
    dict_uni.txt.
    """
    ds, tok = _load_dataset(
        qm9_jsonl, "train", target="uni_mol", dict_name="dict_uni.txt"
    )
    for idx in range(len(ds.data)):
        num = ds.data[idx]["num"]
        item = ds.get_train_item(idx)
        coords = item["coordinates"]
        # atom coords only -- no 3-row lattice prefix
        assert coords.shape == (num, 3)
        assert int(item["coordinates_mask"].sum()) == num


# --------------------------------------------------------------------------- #
# DoD 6: COV / MAT (recall + precision) on hand-verifiable toys
# --------------------------------------------------------------------------- #
# Distortion that survives best-fit alignment with a known-band RMSD (~0.716 A),
# safely between QM9's 0.5 A and Drugs' 1.25 A thresholds.
_DISTORT = [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]


def _distorted_cco():
    coords = [
        (x + dx, y + dy, z + dz)
        for (x, y, z), (dx, dy, dz) in zip(_BASE_COORDS, _DISTORT)
    ]
    return _mol_with_conformer("CCO", coords)


def test_cov_mat_identical(eval_mol):
    """Two identical conformers -> full coverage (COV 100%) and MAT ~ 0."""
    ref = _mol_with_conformer("CCO", _BASE_COORDS)
    gen = eval_mol.Chem.Mol(ref)
    assert eval_mol.GetBestRMSD(gen, ref) == pytest.approx(0.0, abs=1e-6)

    cov, mat = eval_mol.get_cov_mat([gen], [ref], threshold=0.5)
    assert cov == 100.0
    assert mat == pytest.approx(0.0, abs=1e-6)

    cov_p, mat_p = eval_mol.get_cov_mat_p([gen], [ref], threshold=0.5)
    assert cov_p == 100.0
    assert mat_p == pytest.approx(0.0, abs=1e-6)


def test_cov_mat_translation_invariant(eval_mol):
    """GetBestRMS removes rigid translation -> a shifted copy still matches."""
    ref = _mol_with_conformer("CCO", _BASE_COORDS)
    shifted = _mol_with_conformer(
        "CCO", [(x + 100.0, y, z) for (x, y, z) in _BASE_COORDS]
    )
    assert eval_mol.GetBestRMSD(shifted, ref) == pytest.approx(0.0, abs=1e-6)
    cov, mat = eval_mol.get_cov_mat([shifted], [ref], threshold=0.5)
    assert cov == 100.0
    assert mat == pytest.approx(0.0, abs=1e-6)


def test_cov_mat_threshold_qm9_vs_drugs(eval_mol):
    """The SAME conformer: rejected at QM9 0.5 A, accepted at Drugs 1.25 A."""
    ref = _mol_with_conformer("CCO", _BASE_COORDS)
    gen = _distorted_cco()
    r = eval_mol.GetBestRMSD(gen, ref)
    # the distortion is engineered to bracket both paper thresholds
    assert 0.5 < r < 1.25, f"toy RMSD {r} must sit between 0.5 and 1.25"

    cov_qm9, mat_qm9 = eval_mol.get_cov_mat([gen], [ref], threshold=0.5)
    cov_drugs, mat_drugs = eval_mol.get_cov_mat([gen], [ref], threshold=1.25)
    assert cov_qm9 == 0.0, "QM9 0.5 A: not covered"
    assert cov_drugs == 100.0, "Drugs 1.25 A: covered"
    # MAT is the min RMSD, independent of the coverage threshold
    assert mat_qm9 == pytest.approx(r, abs=1e-6)
    assert mat_drugs == pytest.approx(r, abs=1e-6)


def test_cov_mat_recall_vs_precision(eval_mol):
    """Recall (per-ref) and precision (per-gen) differ on a mixed gen set."""
    ref = _mol_with_conformer("CCO", _BASE_COORDS)
    good = eval_mol.Chem.Mol(ref)  # identical -> RMSD 0
    bad = _distorted_cco()  # RMSD ~0.716 > 0.5
    r = eval_mol.GetBestRMSD(bad, ref)
    assert r > 0.5

    # recall: the single ref is covered by SOME gen (good) -> COV-R 100, MAT-R 0
    cov_r, mat_r = eval_mol.get_cov_mat([good, bad], [ref], threshold=0.5)
    assert cov_r == 100.0
    assert mat_r == pytest.approx(0.0, abs=1e-6)

    # precision: only 1 of 2 gen is within 0.5 A of a ref -> COV-P 50,
    # MAT-P = mean(0, r)
    cov_p, mat_p = eval_mol.get_cov_mat_p([good, bad], [ref], threshold=0.5)
    assert cov_p == 50.0
    assert mat_p == pytest.approx(r / 2, abs=1e-6)


def test_cov_mat_empty_lists(eval_mol):
    """Empty gen / ref lists return (None, None) and are filtered upstream."""
    ref = _mol_with_conformer("CCO", _BASE_COORDS)
    assert eval_mol.get_cov_mat([], [ref]) == (None, None)
    assert eval_mol.get_cov_mat([ref], []) == (None, None)
    assert eval_mol.get_cov_mat_p([], [ref]) == (None, None)
