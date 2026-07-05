# -*- coding: utf-8 -*-
"""T4 verification for the UniGenX unified training path (uni_mat / uni_mol).

The two unified sub-targets share the unified dict ``dict_uni.txt`` (vocab 193
per DICT_MAP/T4: 186 non-empty lines + 7 specials). This file verifies the real
unified data paths end to end:

  * ``dict_uni.txt`` loads to vocab 193 (dependency-free contract),
  * the single ``uni_mat`` path (material-format jsonl -> ``get_train_uni_mat``)
    and single ``uni_mol`` path (mol-format jsonl -> ``get_train_uni_mol``) each
    collate into a coherent TRAIN batch whose coordinate slots line up 1:1 with
    the flat ``(N, 3)`` coordinates, and a tiny random ``UniGenX`` (vocab 193)
    takes optimizer steps over that batch with a finite (non-NaN/Inf) loss, and
  * the unified *mixed* comma-path: a comma-joined
    ``"material_path,mol_path"`` under ``--target uni_mat`` is dispatched by
    ``unigenx_train._build_dataset`` to ``UnifiedUniGenXDataset``, which
    forks one ``uni_mat`` sub-dataset and one ``uni_mol`` sub-dataset; a batch
    spanning BOTH trains to a finite loss.

Conventions mirror tests/test_training.py and tests/test_train_diffcore.py:
``REPO_ROOT`` on ``sys.path``, repo-relative dict paths, the same tiny model
recipe, and a CUDA skipif on the train steps (``diffloss`` allocates its noise
on the active CUDA device). The vocab and dispatch-wiring contracts are checked
in dependency-free tests that never touch CUDA.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _HAS_CUDA = False

_EOS_ID = 2
_PAD_ID = 0
_BOS_ID = 1

DATA_DIR = REPO_ROOT / "unigenx" / "data"

# uni_mat and uni_mol share dict_uni.txt (186 non-empty lines + 7 specials).
_UNI_DICT = "dict_uni.txt"
_UNI_VOCAB = 193

# material-format TRAIN records (uni_mat): id/formula/lattice/sites. Elements map
# to <m>Elem tokens present in dict_uni.txt. input_coordinates for each record is
# concat[lattice(3), fractional(n)], so coordinate slots = 3 + n_sites.
_UNI_MAT_RECORDS = [
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
        "formula": "Fe2O3",
        "lattice": [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 13.7]],
        "sites": [
            {"element": "Fe", "fractional_coordinates": [0.0, 0.0, 0.35]},
            {"element": "Fe", "fractional_coordinates": [0.0, 0.0, 0.65]},
            {"element": "O", "fractional_coordinates": [0.30, 0.0, 0.25]},
            {"element": "O", "fractional_coordinates": [0.0, 0.30, 0.25]},
            {"element": "O", "fractional_coordinates": [0.70, 0.70, 0.25]},
        ],
    },
    {
        "id": 2,
        "formula": "Ti1O2",
        "lattice": [[4.59, 0.0, 0.0], [0.0, 4.59, 0.0], [0.0, 0.0, 2.96]],
        "sites": [
            {"element": "Ti", "fractional_coordinates": [0.0, 0.0, 0.0]},
            {"element": "O", "fractional_coordinates": [0.30, 0.30, 0.0]},
            {"element": "O", "fractional_coordinates": [0.70, 0.70, 0.0]},
        ],
    },
]

# mol-format TRAIN records (uni_mol): id/smi/pos/num. SMILES chars map to <s>char
# tokens present in dict_uni.txt; jsonl length filtering reads ``num``.
_UNI_MOL_RECORDS = [
    {
        "id": 0,
        "smi": "CCO",
        "pos": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 1.0, 0.0]],
        "num": 3,
    },
    {"id": 1, "smi": "CO", "pos": [[0.0, 0.0, 0.0], [1.4, 0.2, 0.1]], "num": 2},
    {
        "id": 2,
        "smi": "CCC",
        "pos": [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
        "num": 3,
    },
]

# Trimmed material/mol record sets for the mixed comma-path fixture.
_MIXED_MAT_RECORDS = [_UNI_MAT_RECORDS[0], _UNI_MAT_RECORDS[2]]
_MIXED_MOL_RECORDS = _UNI_MOL_RECORDS[:2]


def _build_config(target="uni_mat"):
    from unigenx.model.config import UniGenXConfig

    config = UniGenXConfig(
        vocab_size=_UNI_VOCAB,
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
        pad_token_id=_PAD_ID,
        bos_token_id=_BOS_ID,
        eos_token_id=_EOS_ID,
    )
    config.target = target
    config.tokenizer = "num"
    config.total_num_steps = 10
    config.warmup_num_steps = 2
    config.max_lr = 1e-4
    return config


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _write_mixed_fixtures(tmp_path):
    """Write a material-format jsonl + a mol-format jsonl, return "mat,mol"."""
    mat = _write_jsonl(tmp_path / "unified_mat.jsonl", _MIXED_MAT_RECORDS)
    mol = _write_jsonl(tmp_path / "unified_mol.jsonl", _MIXED_MOL_RECORDS)
    return f"{mat},{mol}"


def _train_two_steps(config, batch):
    """Two optimizer steps of a tiny UniGenX over ``batch`` on CUDA.

    Returns the list of per-step float losses. Exercises the ported hooks
    (config_optimizer / before_batch / after_batch), mirroring T1/T2/T3.
    """
    from unigenx.model.wrapper import UniGenX

    device = "cuda"
    model = UniGenX(config).to(device)
    model.train()
    optimizer, lr_scheduler = model.config_optimizer()
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    losses = []
    for _ in range(2):
        model.before_batch()
        out = model.compute_loss(model(batch), batch)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        model.after_batch()
        losses.append(float(loss.item()))
    return losses


# --------------------------------------------------------------------------- #
# Dependency-free: DICT_MAP vocab contract (never touches CUDA).
# --------------------------------------------------------------------------- #
def test_dict_uni_vocab():
    """dict_uni.txt loads to the DICT_MAP vocab shared by uni_mat/uni_mol."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    config = _build_config()
    tokenizer = UniGenXTokenizer.from_file(str(DATA_DIR / _UNI_DICT), config)
    assert (
        len(tokenizer) == _UNI_VOCAB
    ), f"dict_uni.txt vocab {len(tokenizer)} != DICT_MAP {_UNI_VOCAB}"


# --------------------------------------------------------------------------- #
# uni_mat single-path tiny-train gate: real jsonl collate + finite loss.
# CUDA-gated: diffloss allocates its noise on the active CUDA device.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _HAS_CUDA,
    reason="diffloss allocates its diffusion noise on cuda; a GPU is required",
)
def test_uni_mat_tiny_train_finite_loss(tmp_path):
    from unigenx.data.dataset import MODE, UniGenXDataset
    from unigenx.data.tokenizer import UniGenXTokenizer

    config = _build_config("uni_mat")
    tokenizer = UniGenXTokenizer.from_file(str(DATA_DIR / _UNI_DICT), config)
    assert len(tokenizer) == _UNI_VOCAB
    config.mask_token_id = tokenizer.mask_idx

    jsonl = _write_jsonl(tmp_path / "uni_mat.jsonl", _UNI_MAT_RECORDS)
    ds = UniGenXDataset(tokenizer, str(jsonl), config, shuffle=False, mode=MODE.TRAIN)
    assert len(ds.data) == len(_UNI_MAT_RECORDS)
    batch = ds.collate([ds[i] for i in range(len(ds.data))])
    # coordinate slots (mask==1) line up 1:1 with the flat (N,3) coords;
    # input_coordinates == concat[lattice(3), fractional(n)] per record.
    assert int(batch["coordinates_mask"].sum()) == batch["input_coordinates"].shape[0]
    assert batch["input_coordinates"].shape[-1] == 3

    losses = _train_two_steps(config, batch)
    assert all(np.isfinite(losses)), f"non-finite uni_mat training loss: {losses}"


# --------------------------------------------------------------------------- #
# uni_mol single-path tiny-train gate: real jsonl collate + finite loss.
# CUDA-gated: diffloss allocates its noise on the active CUDA device.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _HAS_CUDA,
    reason="diffloss allocates its diffusion noise on cuda; a GPU is required",
)
def test_uni_mol_tiny_train_finite_loss(tmp_path):
    from unigenx.data.dataset import MODE, UniGenXDataset
    from unigenx.data.tokenizer import UniGenXTokenizer

    config = _build_config("uni_mol")
    tokenizer = UniGenXTokenizer.from_file(str(DATA_DIR / _UNI_DICT), config)
    assert len(tokenizer) == _UNI_VOCAB
    config.mask_token_id = tokenizer.mask_idx

    jsonl = _write_jsonl(tmp_path / "uni_mol.jsonl", _UNI_MOL_RECORDS)
    ds = UniGenXDataset(tokenizer, str(jsonl), config, shuffle=False, mode=MODE.TRAIN)
    assert len(ds.data) == len(_UNI_MOL_RECORDS)
    batch = ds.collate([ds[i] for i in range(len(ds.data))])
    # coordinate slots (mask==1) line up 1:1 with the flat (N,3) coords.
    assert int(batch["coordinates_mask"].sum()) == batch["input_coordinates"].shape[0]
    assert batch["input_coordinates"].shape[-1] == 3

    losses = _train_two_steps(config, batch)
    assert all(np.isfinite(losses)), f"non-finite uni_mol training loss: {losses}"


# --------------------------------------------------------------------------- #
# Dependency-free: _build_dataset comma-path dispatch wiring (never uses CUDA).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", ["uni_mat", "uni_mol"])
def test_build_dataset_dispatch_mixed_vs_single(tmp_path, target):
    """A comma-path -> UnifiedUniGenXDataset with forked sub-targets;
    a single path -> plain UniGenXDataset. Trainer config stays pristine."""
    import unigenx_train as ut
    from unigenx.data.dataset import MODE, UnifiedUniGenXDataset, UniGenXDataset
    from unigenx.data.tokenizer import UniGenXTokenizer

    config = _build_config(target)
    tokenizer = UniGenXTokenizer.from_file(str(DATA_DIR / _UNI_DICT), config)
    config.mask_token_id = tokenizer.mask_idx

    comma_path = _write_mixed_fixtures(tmp_path)
    unified = ut._build_dataset(
        tokenizer, comma_path, config, shuffle=False, mode=MODE.TRAIN
    )
    assert isinstance(unified, UnifiedUniGenXDataset)
    # Sub-datasets are always forked to uni_mat / uni_mol regardless of --target.
    assert unified.material_dataset.args.target == "uni_mat"
    assert unified.molecule_dataset.args.target == "uni_mol"
    assert len(unified.material_dataset.data) == len(_MIXED_MAT_RECORDS)
    assert len(unified.molecule_dataset.data) == len(_MIXED_MOL_RECORDS)
    assert len(unified) == len(_MIXED_MAT_RECORDS) + len(_MIXED_MOL_RECORDS)

    # The Trainer config must remain pristine: UnifiedUniGenXDataset forks
    # its sub-configs via config.copy() (a fresh deep copy), so the Trainer's own
    # config is not mutated (target unchanged) and stays picklable for the
    # checkpoint ["args"] round-trip.
    assert config.target == target
    assert config.copy() is not config  # UniGenXConfig.copy() -> fresh deep copy
    asdict(config)  # must not raise

    # A single (comma-free) path falls back to the plain dataset. Pick the path
    # whose format matches --target (mat=elem 0, mol=elem 1).
    single_path = comma_path.split(",")[0 if target == "uni_mat" else 1]
    single = ut._build_dataset(
        tokenizer, single_path, config, shuffle=False, mode=MODE.TRAIN
    )
    assert isinstance(single, UniGenXDataset)
    assert not isinstance(single, UnifiedUniGenXDataset)


# --------------------------------------------------------------------------- #
# Unified mixed comma-path tiny-train gate: real comma-path collate spanning
# BOTH sub-datasets + finite loss.
# CUDA-gated: diffloss allocates its noise on the active CUDA device.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _HAS_CUDA,
    reason="diffloss allocates its diffusion noise on cuda; a GPU is required",
)
def test_unified_mixed_comma_path_finite_loss(tmp_path):
    import unigenx_train as ut
    from unigenx.data.dataset import MODE
    from unigenx.data.tokenizer import UniGenXTokenizer

    config = _build_config("uni_mat")
    tokenizer = UniGenXTokenizer.from_file(str(DATA_DIR / _UNI_DICT), config)
    assert len(tokenizer) == _UNI_VOCAB
    config.mask_token_id = tokenizer.mask_idx

    comma_path = _write_mixed_fixtures(tmp_path)
    ds = ut._build_dataset(
        tokenizer, comma_path, config, shuffle=False, mode=MODE.TRAIN
    )

    # Collate a batch spanning BOTH sub-datasets (material items then mol items).
    batch = ds.collate([ds[i] for i in range(len(ds))])
    # coordinate slots (mask==1) must line up 1:1 with the flat (N,3) coords.
    assert int(batch["coordinates_mask"].sum()) == batch["input_coordinates"].shape[0]
    assert batch["input_coordinates"].shape[-1] == 3

    losses = _train_two_steps(config, batch)
    assert all(np.isfinite(losses)), f"non-finite unified-mixed training loss: {losses}"
