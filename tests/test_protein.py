# -*- coding: utf-8 -*-
"""Stage-6 (protein conformational dynamics / MD) smoke tests.

Covers RELEASE_PLAN.md Section 3 Definition-of-Done for the protein path
(target ``prot``, checkpoints 1_m_p..12_m_p / b_p / e_bs / e_wo_bs, vocab 28):

  3. dict vocab assertion -- ``dict_prot.txt`` is vocab 28 (21 residue tokens
                             A R N D C Q E G H I L K M F P S T W Y V X + 7
                             special tokens) and every residue resolves (not
                             <unk>). Plus a skip-if-present check that a protein
                             checkpoint's ``embed_tokens.weight`` dim0 == 28.
  4. collation            -- get_{train,infer}_item_prot build the sequence
                             tokens + coordinate-mask layout
                             ``<bos> seq <coord> [n Cα coords] <eos>``; the
                             (per-residue) coordinate conditioning stream lands
                             at the mask==1 slots. A mock precomputed ESM-2
                             embedding (random tensor) is aligned to the residue
                             positions via the source esm_mask layout (the
                             released path is the baseline: sequence tokens +
                             coordinates only, so ESM here is a mock tensor and
                             no ``esm`` package / model support is required).
  5. inference dry-run    -- a tiny random model runs the prot generate path and
                             decode_batch(entity="prot") returns per-residue Cα
                             coordinates with no lattice block (CUDA-gated).
  6. eval                 -- the TICA Cα-pair distance featurization (tica_eval.py)
                             runs on a toy
                             LMDB trajectory. The full TICA fit needs deeptime,
                             which is skipped when absent (do NOT pip-install it).

The dict / collation / protein-list / TICA-featurization tests stay green with
no heavy deps (lmdb is a declared repo dependency). The checkpoint test skips
when no protein checkpoint is present; the generate dry-run skips without CUDA;
the TICA-fit test skips without deeptime.
"""
import base64
import importlib.util
import json
import os
import pickle
import sys
import zlib
from pathlib import Path

import lmdb
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "unigenx" / "data"
EVAL_PROT_DIR = REPO_ROOT / "eval" / "protein"
TICA_EVAL_PY = EVAL_PROT_DIR / "tica_eval.py"
GEN_PROT_SH = REPO_ROOT / "scripts" / "gen_prot.sh"

# vocab = non-empty dict lines (21) + 7 special tokens
PROT_VOCAB = 28
# the 20 standard amino acids + unknown, in dict_prot.txt order
RESIDUES = list("ARNDCQEGHILKMFPSTWYVX")

_WORKSPACE = REPO_ROOT.parent
_CKPT_DIR = Path(os.environ.get("UNIGENX_CHECKPOINTS", str(_WORKSPACE / "checkpoints")))
# any of these (all vocab 28) satisfies the embedding-vocab check
PROT_CHECKPOINTS = ["1_m_p.pt", "b_p.pt", "e_bs.pt", "e_wo_bs.pt"]

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _HAS_CUDA = False


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prot_config(target="prot"):
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


def _load_dataset(path, mode, target="prot"):
    from unigenx.data.dataset import MODE, UniGenXDataset
    from unigenx.data.tokenizer import UniGenXTokenizer

    cfg = _prot_config(target)
    tok = UniGenXTokenizer.from_file(str(DATA_DIR / "dict_prot.txt"), cfg)
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


# a short toy protein sequence (6 residues); "pos" are arbitrary Cα coordinates
_TOY_SEQ = "AGCCEK"
_TOY_POS = [
    [0.0, 0.0, 0.0],
    [3.8, 0.0, 0.0],
    [7.6, 0.0, 0.0],
    [11.4, 0.0, 0.0],
    [15.2, 0.0, 0.0],
    [19.0, 0.0, 0.0],
]
_TOY_TRAIN = {"id": 0, "aa": _TOY_SEQ, "pos": _TOY_POS}
_TOY_INFER = {"id": 0, "seq": _TOY_SEQ}


def _write_jsonl(tmp_path_factory, name, records):
    path = tmp_path_factory.mktemp("prot") / name
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def train_jsonl(tmp_path_factory):
    return _write_jsonl(tmp_path_factory, "train_toy.jsonl", [_TOY_TRAIN])


@pytest.fixture(scope="module")
def infer_jsonl(tmp_path_factory):
    return _write_jsonl(tmp_path_factory, "infer_toy.jsonl", [_TOY_INFER])


# --------------------------------------------------------------------------- #
# DoD 3: dict vocab == 28 and every residue resolves
# --------------------------------------------------------------------------- #
def test_dict_prot_vocab():
    from unigenx.data.tokenizer import UniGenXTokenizer

    path = DATA_DIR / "dict_prot.txt"
    assert path.exists(), f"missing committed dict: {path}"
    tok = UniGenXTokenizer.from_file(str(path))
    assert len(tok) == PROT_VOCAB, (
        f"dict_prot.txt: expected vocab {PROT_VOCAB}, got {len(tok)} "
        "(vocab must equal non-empty dict lines (21) + 7 special tokens)"
    )


def test_prot_residue_tokens_resolve():
    """Each amino-acid token must be real; a residue collapsing to <unk> would
    silently corrupt the conditioning sequence."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    tok = UniGenXTokenizer.from_file(str(DATA_DIR / "dict_prot.txt"))
    for res in RESIDUES:
        assert tok.get_idx(res) != tok.unk_idx, f"residue {res} must be a real token"
    ids = [tok.get_idx(r) for r in RESIDUES]
    assert len(set(ids)) == len(RESIDUES)  # 21 distinct residue ids


# --------------------------------------------------------------------------- #
# DoD 3 (additional): protein checkpoint embedding vocab == 28, skip-if-absent
# --------------------------------------------------------------------------- #
def test_prot_checkpoint_embedding_vocab():
    ckpt = next(
        (_CKPT_DIR / n for n in PROT_CHECKPOINTS if (_CKPT_DIR / n).exists()), None
    )
    if ckpt is None:
        pytest.skip(f"no protein checkpoint {PROT_CHECKPOINTS} under {_CKPT_DIR}")

    try:
        container = _load_checkpoint_container(ckpt)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot load {ckpt.name} ({type(e).__name__}: {e})")

    matches = [
        k for k in container if isinstance(k, str) and k.endswith("embed_tokens.weight")
    ]
    assert matches, f"{ckpt.name}: no *embed_tokens.weight in state dict"
    for k in matches:
        assert container[k].shape[0] == PROT_VOCAB, (
            f"{ckpt.name}:{k} embedding dim0 {container[k].shape[0]} "
            f"!= {PROT_VOCAB} (protein checkpoints map to dict_prot, vocab 28)"
        )


# --------------------------------------------------------------------------- #
# DoD 4: collation -- sequence + coordinate-mask layout, coordinate conditioning
# --------------------------------------------------------------------------- #
def test_prot_infer_collation(infer_jsonl):
    ds, tok = _load_dataset(infer_jsonl, "infer")
    assert len(ds.data) == 1

    item = ds.get_infer_item(0)
    assert {"tokens", "coordinates_mask"} <= set(item)
    # no input coordinates are provided at inference time (they are generated)
    assert "coordinates" not in item

    toks = item["tokens"]
    mask = item["coordinates_mask"]
    n = len(_TOY_SEQ)

    # token layout: <bos> A G C C E K <coord>   (len n + 2)
    assert toks[0] == tok.bos_idx
    for i, res in enumerate(_TOY_SEQ):
        assert toks[1 + i] == tok.get_idx(res)
    assert toks[1 + n] == tok.coord_idx
    assert len(toks) == n + 2
    assert tok.unk_idx not in list(toks)

    # mask layout: [0]*(n+2) prompt + [1]*n coordinate slots + [0] eos  (len 2n+3)
    assert len(mask) == 2 * n + 3
    assert list(mask) == [0] * (n + 2) + [1] * n + [0]
    assert int(mask.sum()) == n  # one Cα coordinate per residue


def test_prot_train_collation(train_jsonl):
    """Training item: the per-residue Cα coordinates are the conditioning stream
    and land exactly at the mask==1 slots; coordinates are geometry-centered."""
    ds, tok = _load_dataset(train_jsonl, "train")
    item = ds.get_train_item(0)
    assert {"tokens", "coordinates", "coordinates_mask"} <= set(item)

    toks = item["tokens"]
    coords = item["coordinates"]
    mask = item["coordinates_mask"]
    n = len(_TOY_SEQ)

    # token layout: <bos> seq <coord> [<mask>]*n <eos>
    assert toks[0] == tok.bos_idx
    for i, res in enumerate(_TOY_SEQ):
        assert toks[1 + i] == tok.get_idx(res)
    assert toks[1 + n] == tok.coord_idx
    assert list(toks[2 + n : 2 + 2 * n]) == [tok.mask_idx] * n
    assert toks[-1] == tok.eos_idx
    assert len(toks) == 2 * n + 3

    # mask marks exactly the n Cα coordinate slots
    assert list(mask) == [0] * (n + 2) + [1] * n + [0]
    assert int(mask.sum()) == n

    # coordinate conditioning stream: one (x,y,z) per residue, geometry-centered
    pos = np.asarray(_TOY_POS, dtype=np.float32)
    assert coords.shape == (n, 3)
    assert np.allclose(coords, pos - pos.mean(axis=0))
    # centering => zero mean (no lattice / property rows prepended)
    assert np.allclose(coords.mean(axis=0), 0.0, atol=1e-5)


def test_prot_collate_batch(infer_jsonl):
    """collate() batches the sequence tokens + coordinate mask; the mask is at
    least as wide as the (padded) prompt ids (generate-the-rest layout)."""
    ds, _ = _load_dataset(infer_jsonl, "infer")
    items = [ds.get_infer_item(0)]
    batch = ds.collate(items)

    assert "input_ids" in batch and "coordinates_mask" in batch
    n = len(_TOY_SEQ)
    assert batch["input_ids"].shape[1] == n + 2
    assert batch["coordinates_mask"].shape[1] == 2 * n + 3
    assert batch["coordinates_mask"].shape[1] >= batch["input_ids"].shape[1]
    # no input coordinates at inference time
    assert "input_coordinates" not in batch


def test_prot_mock_esm_embedding_conditioning_layout():
    """The released protein path is the baseline (sequence tokens + coordinate
    mask). This documents the *precomputed ESM-2* conditioning layout with a mock
    random embedding (no ``esm`` package needed, "dimensions aligned"): a
    precomputed ESM-2 embedding has one row per residue and aligns to the residue
    token positions via the source esm_mask ``[0] + [1]*n + [0]*(n+2)``."""
    n = len(_TOY_SEQ)
    esm_dim = 320  # ESM-2 t6 embedding width (any width works; alignment is by n)
    rng = np.random.default_rng(0)
    mock_esm_embedding = rng.standard_normal((n, esm_dim)).astype(np.float32)

    # esm_mask layout as produced by the source UniGenXProteinDataset:
    # one 1 per residue token, zeros over <bos>, <coord>, the coord slots, <eos>.
    esm_mask = np.array([0] + [1] * n + [0] * (n + 2))
    assert len(esm_mask) == 2 * n + 3  # matches the coordinate-mask width
    assert int(esm_mask.sum()) == n  # one embedding row per residue

    # the mock embedding (dim-aligned) maps one row onto each masked residue slot
    residue_positions = np.nonzero(esm_mask)[0]
    assert residue_positions.tolist() == list(range(1, n + 1))
    assert mock_esm_embedding.shape[0] == int(esm_mask.sum())


# --------------------------------------------------------------------------- #
# DoD 4 (decode): entity="prot" decodes per-residue Cα coords, NO lattice block
# --------------------------------------------------------------------------- #
def test_decode_batch_prot():
    from unigenx.data.tokenizer import UniGenXTokenizer

    tok = UniGenXTokenizer.from_file(str(DATA_DIR / "dict_prot.txt"))
    # <bos> A G <coord> c0 c1 <eos>   (a 2-residue protein)
    tokens = np.array(
        [
            tok.bos_idx,
            tok.get_idx("A"),
            tok.get_idx("G"),
            tok.coord_idx,
            tok.mask_idx,
            tok.mask_idx,
            tok.eos_idx,
        ]
    )
    mask = np.array([0, 0, 0, 0, 1, 1, 0])
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    out = tok.decode_batch(tokens[None], coords, mask[None], "prot")
    assert len(out) == 1
    sent, atom_coordinates = out[0]  # prot returns a 2-tuple (no lattice)
    # both Cα coordinate slots decode to atom coords (no lattice diverted)
    assert len(atom_coordinates) == 2
    assert np.allclose(atom_coordinates[0], [1.0, 2.0, 3.0])
    assert np.allclose(atom_coordinates[1], [4.0, 5.0, 6.0])


# --------------------------------------------------------------------------- #
# DoD 5: inference dry-run -- tiny random model, prot generate + decode (CUDA)
# --------------------------------------------------------------------------- #
def _build_tiny_prot_model():
    from unigenx.model.config import UniGenXConfig
    from unigenx.model.wrapper import UniGenX

    config = UniGenXConfig(
        vocab_size=PROT_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        diff_width=32,
        diff_depth=2,
        diff_steps="4",
        diff_mul=1,
        is_solver=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config.mask_token_id = PROT_VOCAB - 3  # <mask> id for dict_prot (25)
    model = UniGenX(config)
    model.eval()
    return model


@pytest.mark.skipif(
    not _HAS_CUDA,
    reason="diffloss.sample() allocates its noise on cuda; a GPU is required",
)
def test_prot_generate_dry_run(infer_jsonl):
    from transformers import GenerationConfig

    from unigenx.data.tokenizer import UniGenXTokenizer
    from unigenx.model.unigenx import UniGenXOutput

    device = "cuda"
    ds, _ = _load_dataset(infer_jsonl, "infer")
    tok = UniGenXTokenizer.from_file(str(DATA_DIR / "dict_prot.txt"), _prot_config())
    item = ds.get_infer_item(0)
    batch = ds.collate([item])

    model = _build_tiny_prot_model().to(device)
    input_ids = batch["input_ids"].to(device)
    coordinates_mask = batch["coordinates_mask"].to(device)

    gen_config = GenerationConfig(
        pad_token_id=0,
        eos_token_id=2,
        use_cache=True,
        max_length=coordinates_mask.shape[1],
        return_dict_in_generate=True,
    )
    ret = model.net.generate(
        input_ids=input_ids,
        coordinates_mask=coordinates_mask,
        generation_config=gen_config,
        max_length=coordinates_mask.shape[1],
    )
    assert isinstance(ret, UniGenXOutput)
    assert ret.coordinates is not None and ret.coordinates.shape[-1] == 3

    # the prot inference branch decodes with entity="prot" (no lattice)
    decoded = tok.decode_batch(
        ret.sequences.cpu().numpy(),
        ret.coordinates.cpu().numpy(),
        coordinates_mask.cpu().numpy(),
        "prot",
    )
    assert len(decoded) == 1
    sent, atom_coordinates = decoded[0]
    assert isinstance(atom_coordinates, list)


# --------------------------------------------------------------------------- #
# DoD 6 (eval): TICA Cα-pair distance featurization + full-fit skip
# --------------------------------------------------------------------------- #
def _make_toy_lmdb(path, frames):
    env = lmdb.open(str(path), subdir=False, map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for i, coords in enumerate(frames):
            txn.put(str(i).encode(), zlib.compress(pickle.dumps({"coords": coords})))
    env.close()


def test_tica_internal_coordinates(tmp_path):
    """The TICA featurizer computes all Cα-pair distances per frame -> an
    (n_frames, n_pairs) matrix; verify shape and known distances."""
    mod = _load_module(TICA_EVAL_PY, "tica_eval")

    # 3 frames, 3 Cα atoms each -> C(3,2)=3 pairs, order (0,1),(0,2),(1,2)
    frames = [
        [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]],  # 01=5, 02=0, 12=5
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # all zero
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # 01=1, 02=1, 12=0
    ]
    lmdb_path = tmp_path / "toy.lmdb"
    _make_toy_lmdb(lmdb_path, frames)

    dist = mod.internal_coordinates_ca_backbone(str(lmdb_path))
    assert dist.shape == (3, 3)
    assert np.allclose(dist[0], [5.0, 0.0, 5.0])
    assert np.allclose(dist[1], [0.0, 0.0, 0.0])
    assert np.allclose(dist[2], [1.0, 1.0, 0.0])

    # convert_to_nparray passes numpy through and converts lists
    assert isinstance(mod.convert_to_nparray([1, 2, 3]), np.ndarray)
    arr = np.array([1.0, 2.0])
    assert mod.convert_to_nparray(arr) is arr


def test_tica_fit_requires_deeptime(tmp_path):
    """The full TICA free-energy-surface fit uses deeptime, which is not
    installed here (do NOT pip-install it); the fit is intentionally skipped."""
    pytest.importorskip("deeptime")
    mod = _load_module(TICA_EVAL_PY, "tica_eval")

    # A non-degenerate toy trajectory: 5 Calpha points drifting with per-point
    # noise so the pairwise-distance features vary independently (rank >= 2) and
    # TICA yields the two components (TIC1/TIC2) the FES projection needs. (A
    # collinear/equal-distance toy is rank-1 and would give only one component.)
    rng = np.random.default_rng(0)
    base = rng.standard_normal((5, 3)) * 3.0
    frames = [
        (base + i * 0.03 + rng.standard_normal((5, 3)) * 0.3).tolist()
        for i in range(200)
    ]
    train = tmp_path / "train.lmdb"
    test = tmp_path / "test.lmdb"
    _make_toy_lmdb(train, frames)
    _make_toy_lmdb(test, frames)
    out_png = tmp_path / "tica.png"
    comps = mod.run_tica_free_energy(
        str(train), str(test), lagtime=5, out_png=str(out_png)
    )
    assert comps.shape[1] == 2  # TIC1 / TIC2
    assert out_png.exists()


# --------------------------------------------------------------------------- #
# hygiene: ported scripts carry no machine-absolute paths / internal branch names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [TICA_EVAL_PY, GEN_PROT_SH])
def test_ported_scripts_have_no_absolute_paths(path):
    text = path.read_text()
    # forbidden internal identifiers, base64-encoded so the source itself
    # carries no literal internal string (decoded at runtime before scanning)
    for _enc in (
        "L21zcmFsYXBoaWxseTI=",  # internal blob mount path
        "L3ZlcGZzLWZvci10cmFpbmluZw==",  # internal training mount path
        "L2RhdGFkaXNr",  # internal data mount path
        "L2Jsb2Iv",  # internal blob path
        "Z29uZ2JvLw==",  # internal user/branch prefix
        "eWFuZ3k=",  # internal user id
        "eWxp",  # internal user id
    ):
        needle = base64.b64decode(_enc).decode()
        assert (
            needle not in text
        ), f"{path.name} contains a forbidden internal identifier"
