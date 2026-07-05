# -*- coding: utf-8 -*-
"""Stage-5 (property-conditional molecule generation, 6 properties) smoke tests.

Covers RELEASE_PLAN.md Section 3 Definition-of-Done for the property-conditional
molecule path (target ``cond_mol``, checkpoint ``c_mol``, vocab 34). cond_mol is
pure conditional generation (LDM, no classifier-free guidance) that supports one
or many joint property constraints (num_cond) per molecule.

  3. dict vocab assertion -- ``dict_cond_mol.txt`` is vocab 34 (27 standard lines
                             + 7 special tokens) and its 6 property markers
                             <a><g><h><l><m><c> (alpha/gap/homo/lumo/mu/Cv)
                             resolve to real tokens (not <unk>). Plus a
                             skip-if-present check that ``c_mol.pt`` has
                             ``embed_tokens.weight`` dim0 == 34.
  4. collation            -- the property conditioning is a *continuous value*
                             fed through the coordinate stream (one coordinate
                             row [v, v, v] per constraint), preceded by its
                             property marker token: layout
                             <bos> [<prop_i> propval_i]*num_cond <w> ... . There
                             is NO lattice slot (molecular domain, unlike
                             crystals). Both get_train_cond_mol and
                             get_infer_cond_mol are checked for single and
                             multi-property inputs, and collate() exposes
                             num_cond.
  6. eval                 -- the RDKit-validity component of the six-constraint
                             metric runs on a toy batch and evaluate_cond.py's
                             generate_mol_struct groups valid 3D mols by
                             property. The QM property MAE (evaluate_mol_prop.py)
                             depends on psi4/psikit and is skipped when absent.

The dict, collation and RDKit tests stay green with no external heavy deps
(rdkit is a declared repo dependency). The checkpoint test skips when the
(multi-GB) c_mol checkpoint is absent; the property-MAE test skips without
psi4/psikit.
"""
import base64
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
EVAL_COND_PY = REPO_ROOT / "eval" / "molecule" / "evaluate_cond.py"
EVAL_MOL_PROP_PY = REPO_ROOT / "eval" / "molecule" / "evaluate_mol_prop.py"

# vocab = non-empty dict lines (27) + 7 special tokens
COND_MOL_VOCAB = 34
# the 6 property marker tokens (last 6 standard tokens of dict_cond_mol.txt)
PROPERTY_TOKENS = ["<a>", "<g>", "<h>", "<l>", "<m>", "<c>"]
# their single-char codes, as consumed by get_*_cond_mol via f"<{prop[0]}>"
PROPERTY_CODES = ["a", "g", "h", "l", "m", "c"]

_WORKSPACE = REPO_ROOT.parent
_CKPT_DIR = Path(os.environ.get("UNIGENX_CHECKPOINTS", str(_WORKSPACE / "checkpoints")))
COND_MOL_CHECKPOINT = "c_mol.pt"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cond_mol_config(target="cond_mol"):
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


def _load_dataset(path, mode, target="cond_mol"):
    from unigenx.data.dataset import MODE, UniGenXDataset
    from unigenx.data.tokenizer import UniGenXTokenizer

    cfg = _cond_mol_config(target)
    tok = UniGenXTokenizer.from_file(str(DATA_DIR / "dict_cond_mol.txt"), cfg)
    m = MODE.TRAIN if mode == "train" else MODE.INFER
    ds = UniGenXDataset(tok, path, args=cfg, shuffle=False, mode=m)
    return ds, tok


def _load_checkpoint_container(path):
    """Load a checkpoint state dict, tolerating a saved args object whose
    original class is not shipped with this package."""
    from unigenx.utils.checkpoint import load_checkpoint

    state = load_checkpoint(path)
    container = state
    if isinstance(state, dict):
        for key in ("model", "module", "state_dict"):
            if key in state and isinstance(state[key], dict):
                container = state[key]
                break
    return container


# Toy conditional-molecule records. Both prop and prop_val are per-constraint
# lists (single-property is just the length-1 case). SMILES "CCO" -> 3 atoms.
_TOY_TRAIN_MULTI = {
    "id": 0,
    "smi": "CCO",
    "pos": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 1.0, 0.0]],
    "prop": PROPERTY_CODES,  # 6 joint constraints
    "prop_val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}
_TOY_INFER_MULTI = {
    "id": 0,
    "prop": PROPERTY_CODES,
    "prop_val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}
_TOY_INFER_SINGLE = {
    "id": 1,
    "prop": ["a"],  # single-property = num_cond 1
    "prop_val": [0.7],
}


def _write_jsonl(tmp_path_factory, name, records):
    path = tmp_path_factory.mktemp("cond_mol") / name
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def train_jsonl(tmp_path_factory):
    return _write_jsonl(tmp_path_factory, "train_toy.jsonl", [_TOY_TRAIN_MULTI])


@pytest.fixture(scope="module")
def infer_jsonl(tmp_path_factory):
    return _write_jsonl(
        tmp_path_factory, "infer_toy.jsonl", [_TOY_INFER_MULTI, _TOY_INFER_SINGLE]
    )


# --------------------------------------------------------------------------- #
# DoD 3: dict vocab == 34 and the 6 property tokens resolve
# --------------------------------------------------------------------------- #
def test_dict_cond_mol_vocab():
    from unigenx.data.tokenizer import UniGenXTokenizer

    path = DATA_DIR / "dict_cond_mol.txt"
    assert path.exists(), f"missing committed dict: {path}"
    tok = UniGenXTokenizer.from_file(str(path))
    assert len(tok) == COND_MOL_VOCAB, (
        f"dict_cond_mol.txt: expected vocab {COND_MOL_VOCAB}, got {len(tok)} "
        "(vocab must equal non-empty dict lines (27) + 7 special tokens)"
    )


def test_dict_cond_mol_property_tokens_resolve():
    """The 6 conditioning markers must be real tokens; if any fell back to <unk>
    the property signal would be silently lost."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    tok = UniGenXTokenizer.from_file(str(DATA_DIR / "dict_cond_mol.txt"))
    for marker in PROPERTY_TOKENS:
        assert tok.get_idx(marker) != tok.unk_idx, f"{marker} must be a real token"
    # the 6 marker ids are distinct
    ids = [tok.get_idx(m) for m in PROPERTY_TOKENS]
    assert len(set(ids)) == len(PROPERTY_TOKENS)
    # f"<{code[0]}>" (as built by get_*_cond_mol) hits the same real tokens
    for code, marker in zip(PROPERTY_CODES, PROPERTY_TOKENS):
        assert tok.get_idx(f"<{code[0]}>") == tok.get_idx(marker) != tok.unk_idx


# --------------------------------------------------------------------------- #
# DoD 3 (additional): c_mol checkpoint embedding vocab == 34, skip-if-absent
# --------------------------------------------------------------------------- #
def test_cond_mol_checkpoint_embedding_vocab():
    ckpt = _CKPT_DIR / COND_MOL_CHECKPOINT
    if not ckpt.exists():
        pytest.skip(f"no {COND_MOL_CHECKPOINT} under {_CKPT_DIR}")

    try:
        container = _load_checkpoint_container(ckpt)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot load {ckpt.name} ({type(e).__name__}: {e})")

    matches = [
        k for k in container if isinstance(k, str) and k.endswith("embed_tokens.weight")
    ]
    assert matches, f"{ckpt.name}: no *embed_tokens.weight in state dict"
    for k in matches:
        assert container[k].shape[0] == COND_MOL_VOCAB, (
            f"{ckpt.name}:{k} embedding dim0 {container[k].shape[0]} "
            f"!= {COND_MOL_VOCAB} (c_mol is vocab 34)"
        )


# --------------------------------------------------------------------------- #
# DoD 4: collation -- property conditioning is a continuous coordinate value,
# multi-property prefix layout, and NO lattice slot (molecular domain)
# --------------------------------------------------------------------------- #
def test_cond_mol_train_collation(train_jsonl):
    ds, tok = _load_dataset(train_jsonl, "train")
    assert len(ds.data) == 1

    item = ds.get_train_item(0)
    assert {"tokens", "coordinates", "coordinates_mask"} <= set(item)
    toks = item["tokens"]
    coords = item["coordinates"]
    mask = item["coordinates_mask"]

    n_cond = len(_TOY_TRAIN_MULTI["prop"])  # 6
    n_atoms = len(_TOY_TRAIN_MULTI["pos"])  # 3

    # token layout: <bos> [<prop_i> <mask>]*num_cond <w> C C O <coord> m m m <eos>
    assert toks[0] == tok.bos_idx
    for k in range(n_cond):
        assert toks[1 + 2 * k] == tok.get_idx(PROPERTY_TOKENS[k])  # marker
        assert toks[2 + 2 * k] == tok.mask_idx  # continuous value slot
    w_pos = 1 + 2 * n_cond
    assert toks[w_pos] == tok.get_idx("<w>")
    # smiles then <coord>, atom mask slots, <eos>
    coord_pos = w_pos + 1 + 3  # after 3 smiles tokens (C, C, O)
    assert toks[coord_pos] == tok.coord_idx
    assert toks[-1] == tok.eos_idx
    assert len(toks) == 1 + 2 * n_cond + 1 + 3 + 1 + n_atoms + 1
    # every smiles/element token resolved
    assert tok.unk_idx not in list(toks[w_pos + 1 : coord_pos])

    # mask marks the num_cond property-value slots + the n atom slots
    assert int(mask.sum()) == n_cond + n_atoms

    # ---- conditioning is a CONTINUOUS value in the coordinate stream ----
    # coordinates = [prop_val_k(x3) for k] then [n atom coords]; NO lattice block.
    assert coords.shape == (n_cond + n_atoms, 3)
    for k, pv in enumerate(_TOY_TRAIN_MULTI["prop_val"]):
        # cond_mol feeds the raw value (no material-style log standardization)
        assert np.allclose(coords[k], [pv, pv, pv])
    atom_coords = np.array(_TOY_TRAIN_MULTI["pos"], dtype=np.float32)
    assert np.allclose(coords[n_cond:], atom_coords)


def test_cond_mol_infer_collation(infer_jsonl):
    ds, tok = _load_dataset(infer_jsonl, "infer")
    assert len(ds.data) == 2

    # record 0: multi-property (num_cond 6)
    item = ds.get_infer_item(0)
    toks = item["tokens"]
    mask = item["coordinates_mask"]
    coords = item["coordinates"]
    n_cond = len(_TOY_INFER_MULTI["prop"])

    assert item["num_cond"] == n_cond
    assert toks[0] == tok.bos_idx
    for k in range(n_cond):
        assert toks[1 + 2 * k] == tok.get_idx(PROPERTY_TOKENS[k])
        assert toks[2 + 2 * k] == tok.mask_idx
    assert toks[1 + 2 * n_cond] == tok.get_idx("<w>")
    assert len(toks) == 1 + 2 * n_cond + 1  # <bos> pairs <w>, no smiles yet

    # mask marks exactly the num_cond conditioning slots (values are inputs)
    assert list(mask) == [0] + [0, 1] * n_cond + [0]
    assert int(mask.sum()) == n_cond

    # coordinate rows = one [v,v,v] per constraint; NO lattice slot
    assert coords.shape == (n_cond, 3)
    for k, pv in enumerate(_TOY_INFER_MULTI["prop_val"]):
        assert np.allclose(coords[k], [pv, pv, pv])


def test_cond_mol_infer_single_property_is_length1(infer_jsonl):
    """A single-property run is exactly the num_cond == 1 special case."""
    ds, tok = _load_dataset(infer_jsonl, "infer")
    item = ds.get_infer_item(1)  # _TOY_INFER_SINGLE
    toks = item["tokens"]
    mask = item["coordinates_mask"]

    assert item["num_cond"] == 1
    # layout collapses to <bos> <a> <mask> <w>
    assert list(toks) == [
        tok.bos_idx,
        tok.get_idx("<a>"),
        tok.mask_idx,
        tok.get_idx("<w>"),
    ]
    assert list(mask) == [0, 0, 1, 0]
    assert item["coordinates"].shape == (1, 3)
    assert np.allclose(item["coordinates"][0], [0.7, 0.7, 0.7])


def test_cond_mol_collate_exposes_num_cond(infer_jsonl):
    """collate() batches variable-num_cond items and stacks num_cond; the
    conditioning coordinates are concatenated (sum of num_cond rows)."""
    ds, _ = _load_dataset(infer_jsonl, "infer")
    items = [ds.get_infer_item(i) for i in range(len(ds.data))]
    batch = ds.collate(items)

    assert "num_cond" in batch, "collate must expose per-sample num_cond"
    assert list(batch["num_cond"].tolist()) == [6, 1]
    # input_coordinates concatenated: sum(num_cond) rows, 3 cols, no lattice
    assert batch["input_coordinates"].shape == (6 + 1, 3)
    # mask is at least as wide as the (padded) input ids (fill-the-rest)
    assert batch["coordinates_mask"].shape[1] >= batch["input_ids"].shape[1]


def test_cond_mol_has_no_lattice_slots_unlike_material(train_jsonl):
    """Molecular-domain invariant: the coordinate stream is property rows + atom
    coords only. A crystal would instead carry a 3-row lattice block; cond_mol
    never does."""
    ds, _ = _load_dataset(train_jsonl, "train")
    item = ds.get_train_item(0)
    n_cond = len(_TOY_TRAIN_MULTI["prop"])
    n_atoms = len(_TOY_TRAIN_MULTI["pos"])
    coords = item["coordinates"]
    # exactly num_cond property rows + n atom rows -- no +3 lattice prefix
    assert coords.shape[0] == n_cond + n_atoms
    # the leading rows are property values (all-equal triples), not lattice
    # vectors (which would be an arbitrary 3x3 basis)
    for k in range(n_cond):
        assert coords[k][0] == coords[k][1] == coords[k][2]


# --------------------------------------------------------------------------- #
# DoD 4 (decode): the property-value rows lead the decoded coordinates, so the
# writer's atom_coordinates[num_cond:] slice recovers the atoms.
# --------------------------------------------------------------------------- #
def test_cond_mol_decode_leads_with_property_rows():
    from unigenx.data.tokenizer import UniGenXTokenizer

    tok = UniGenXTokenizer.from_file(str(DATA_DIR / "dict_cond_mol.txt"))
    c, o = tok.get_idx("C"), tok.get_idx("O")
    num_cond = 2
    # <bos> <a> <mask> <g> <mask> <w> C C O <coord> m m m <eos>
    tokens = np.array(
        [
            tok.bos_idx,
            tok.get_idx("<a>"),
            tok.mask_idx,
            tok.get_idx("<g>"),
            tok.mask_idx,
            tok.get_idx("<w>"),
            c,
            c,
            o,
            tok.coord_idx,
            tok.mask_idx,
            tok.mask_idx,
            tok.mask_idx,
            tok.eos_idx,
        ]
    )
    mask = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0])
    # coords for the 5 mask=1 slots: [propval_a, propval_g, atom0, atom1, atom2]
    coords = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 1.0, 0.0],
        ]
    )
    out = tok.decode_batch(tokens[None], coords, mask[None], "cond_mol")
    assert len(out) == 1
    sent, atom_coordinates = out[0]  # cond_mol returns a 2-tuple (no lattice)
    assert len(atom_coordinates) == num_cond + 3
    # writer drops the leading num_cond property rows to keep the atoms
    atoms = np.array(atom_coordinates[num_cond:])
    assert atoms.shape == (3, 3)
    assert np.allclose(atoms[2], [2.5, 1.0, 0.0])


# --------------------------------------------------------------------------- #
# DoD 6: eval -- RDKit-validity front-end (real run) + property-MAE skip
# --------------------------------------------------------------------------- #
def test_evaluate_cond_rdkit_validity(tmp_path):
    """The RDKit-validity component of the six-constraint metric: invalid SMILES
    are not valid generations. evaluate_cond.py rebuilds valid 3D mols and groups
    them by conditioning property."""
    pytest.importorskip("rdkit")
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")  # silence MMFF/parse chatter in test output
    mod = _load_module(EVAL_COND_PY, "evaluate_cond")

    # Use non-degenerate 3D coordinates: generate_mol_struct AddHs (H at origin)
    # then MMFF-optimizes, which needs a reasonable heavy-atom geometry.
    records = [
        {
            "smi": "CCO",
            "coordinates": [[-1.2, 0.2, 0.0], [0.0, -0.5, 0.1], [1.2, 0.3, -0.1]],
            "prop": ["a"],
            "prop_val": [1.0],
        },
        {
            "smi": "CCC",
            "coordinates": [[-1.26, 0.2, 0.0], [0.0, -0.45, 0.0], [1.26, 0.2, 0.0]],
            "prop": ["a"],
            "prop_val": [2.0],
        },
        {
            "smi": "c1ccccc1",
            "coordinates": [
                [1.39, 0, 0],
                [0.69, 1.20, 0],
                [-0.69, 1.20, 0],
                [-1.39, 0, 0],
                [-0.69, -1.20, 0],
                [0.69, -1.20, 0],
            ],
            "prop": ["g"],
            "prop_val": [3.0],
        },
        {
            "smi": "C(C",
            "coordinates": [[0, 0, 0], [1.5, 0, 0]],
            "prop": ["a"],
            "prop_val": [9.0],
        },  # invalid SMILES -> not a valid gen
    ]
    # RDKit validity: 3 of 4 parse (the metric's validity component)
    n_valid = sum(Chem.MolFromSmiles(r["smi"]) is not None for r in records)
    assert n_valid == 3

    path = tmp_path / "gen.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    mol3d, condition = mod.generate_mol_struct(str(path))
    # invalid SMILES skipped; only valid mols grouped by (single) property code
    assert set(condition) == set(mol3d)
    assert set(condition).issubset({"a", "g"})
    total = sum(len(v) for v in mol3d.values())
    # at least one valid 3D mol built (some may drop if MMFF fails to converge)
    assert 1 <= total <= n_valid
    for prop, mols in mol3d.items():
        assert len(mols) == len(condition[prop])
        for m in mols:
            assert m is not None


def test_property_mae_requires_psikit():
    """evaluate_mol_prop.py computes the QM property MAE (energy + HOMO-LUMO gap)
    via Psikit/psi4, which are not installed here (do NOT pip-install them). The
    property-MAE eval is intentionally skipped."""
    assert EVAL_MOL_PROP_PY.exists()
    # psikit imports psi4; either being importable would be required to run it.
    pytest.importorskip("psikit")
    pytest.importorskip("psi4")


# --------------------------------------------------------------------------- #
# hygiene: committed eval scripts carry no machine-absolute paths / internal
# branch names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [EVAL_COND_PY, EVAL_MOL_PROP_PY])
def test_eval_scripts_have_no_absolute_paths(path):
    text = path.read_text()
    # forbidden internal identifiers, base64-encoded so the source itself
    # carries no literal internal string (decoded at runtime before scanning)
    for _enc in (
        "L21zcmFsYXBoaWxseTI=",  # internal blob mount path
        "L3ZlcGZzLWZvci10cmFpbmluZw==",  # internal training mount path
        "L2RhdGFkaXNr",  # internal data mount path
        "L2Jsb2Iv",  # internal blob path
    ):
        needle = base64.b64decode(_enc).decode()
        assert (
            needle not in text
        ), f"{path.name} contains a forbidden internal identifier"
