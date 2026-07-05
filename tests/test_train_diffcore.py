# -*- coding: utf-8 -*-
"""T2 per-target training gates for the UniGenX diffusion-core training path.

For each generation target (material / mol / prot / cond_mat / cond_mol) this
verifies the *real* per-target data path end to end:

  * the domain dict loads to the vocab size pinned in DICT_MAP,
  * ``get_train_item_<target>`` + ``collate`` produce a coherent TRAIN batch
    (coordinate slots line up with the ``coordinates_mask``), and
  * a tiny random ``UniGenX`` takes optimizer steps over that batch with a
    finite (non-NaN/Inf) loss.

Conventions mirror tests/test_training.py: ``REPO_ROOT`` on ``sys.path``,
repo-relative dict paths, the same tiny model recipe, and a CUDA skipif on the
train step (``diffloss`` allocates its noise on the active CUDA device). The
vocab-per-target contract is additionally checked in a dependency-free test that
never touches CUDA, so the DICT_MAP contract is always exercised.
"""
import json
import sys
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


# --------------------------------------------------------------------------- #
# Per-target fixtures: verified TRAIN records + dict + DICT_MAP vocab.
# --------------------------------------------------------------------------- #
_MATERIAL_RECORDS = [
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

_MOL_RECORDS = [
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

_PROT_RECORDS = [
    {
        "id": 0,
        "aa": "AGCCEK",
        "pos": [
            [4.898, -0.62, 1.947],
            [-5.435, 2.526, -3.847],
            [-0.858, -3.771, 2.226],
            [-1.746, -0.579, 0.717],
            [-2.632, -0.464, -3.19],
            [3.185, 2.75, -4.507],
        ],
    },
    {
        "id": 1,
        "aa": "MKLV",
        "pos": [
            [-0.393, 1.339, 3.883],
            [3.328, -2.706, -1.267],
            [-0.685, -1.615, -1.107],
            [3.559, -2.951, -0.63],
        ],
    },
    {
        "id": 2,
        "aa": "WYFPST",
        "pos": [
            [2.981, -4.196, -2.86],
            [-3.14, -2.756, -1.242],
            [0.386, 0.966, -1.472],
            [3.375, 0.293, -1.328],
            [0.858, 1.868, -2.573],
            [-4.442, -4.264, -1.783],
        ],
    },
]

_COND_MAT_RECORDS = [
    {
        "id": 0,
        "formula": "Fe2O3",
        "property": {"dft_bulk_modulus": 120.0},
        "lattice": [[3.1, 0.0, 0.0], [0.0, 3.2, 0.0], [0.0, 0.0, 3.3]],
        "sites": [
            {"element": "Fe", "fractional_coordinates": [0.0, 0.0, 0.0]},
            {"element": "Fe", "fractional_coordinates": [0.5, 0.5, 0.5]},
            {"element": "O", "fractional_coordinates": [0.25, 0.25, 0.25]},
        ],
    },
    {
        "id": 1,
        "formula": "LiO",
        "property": {"dft_bulk_modulus": 80.0},
        "lattice": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        "sites": [
            {"element": "Li", "fractional_coordinates": [0.1, 0.1, 0.1]},
            {"element": "O", "fractional_coordinates": [0.6, 0.6, 0.6]},
        ],
    },
]

_COND_MOL_RECORDS = [
    {
        "id": 0,
        "smi": "CCO",
        "prop": ["a"],
        "prop_val": [1.23],
        "pos": [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
    },
    {
        "id": 1,
        "smi": "CC",
        "prop": ["h", "l"],
        "prop_val": [0.5, -0.7],
        "pos": [[0.3, 0.4, 0.5], [1.3, 1.4, 1.5]],
    },
    {
        "id": 2,
        "smi": "CN",
        "prop": ["g"],
        "prop_val": [2.0],
        "pos": [[0.9, 0.8, 0.7], [1.9, 1.8, 1.7]],
    },
]

# target -> (dict filename, DICT_MAP vocab, TRAIN records, per-target config
# overrides). Vocab = non-empty dict lines + 7 specials.
_TARGETS = {
    "material": ("dict_mat.txt", 355, _MATERIAL_RECORDS, {}),
    "mol": ("dict_qm9.txt", 27, _MOL_RECORDS, {}),
    "prot": (
        "dict_prot.txt",
        28,
        _PROT_RECORDS,
        {
            "space_group": False,
            "reorder": False,
            "rotation_augmentation": False,
            "translation_augmentation": False,
            "scale_coords": None,
            "max_sites": None,
        },
    ),
    "cond_mat": ("dict_cond_mat.txt", 355, _COND_MAT_RECORDS, {}),
    "cond_mol": ("dict_cond_mol.txt", 34, _COND_MOL_RECORDS, {}),
}
_TARGET_IDS = list(_TARGETS)


def _build_config(target, vocab, overrides):
    from unigenx.model.config import UniGenXConfig

    config = UniGenXConfig(
        vocab_size=vocab,
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
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# --------------------------------------------------------------------------- #
# Dependency-free: DICT_MAP vocab contract per target (never touches CUDA).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", _TARGET_IDS)
def test_target_dict_vocab(target):
    from unigenx.data.tokenizer import UniGenXTokenizer

    dict_file, vocab, _, overrides = _TARGETS[target]
    config = _build_config(target, vocab, overrides)
    tokenizer = UniGenXTokenizer.from_file(str(DATA_DIR / dict_file), config)
    assert (
        len(tokenizer) == vocab
    ), f"{target}: dict {dict_file} vocab {len(tokenizer)} != DICT_MAP {vocab}"


# --------------------------------------------------------------------------- #
# Per-target tiny-train gate: real collate + finite loss over optimizer steps.
# CUDA-gated: diffloss allocates its noise on the active CUDA device.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _HAS_CUDA,
    reason="diffloss allocates its diffusion noise on cuda; a GPU is required",
)
@pytest.mark.parametrize("target", _TARGET_IDS)
def test_target_tiny_train_finite_loss(target, tmp_path):
    from unigenx.data.dataset import MODE, UniGenXDataset
    from unigenx.data.tokenizer import UniGenXTokenizer
    from unigenx.model.wrapper import UniGenX

    dict_file, vocab, records, overrides = _TARGETS[target]
    config = _build_config(target, vocab, overrides)

    tokenizer = UniGenXTokenizer.from_file(str(DATA_DIR / dict_file), config)
    assert len(tokenizer) == vocab, f"{target}: vocab {len(tokenizer)} != {vocab}"
    config.mask_token_id = tokenizer.mask_idx

    jsonl = tmp_path / f"{target}.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    ds = UniGenXDataset(tokenizer, str(jsonl), config, shuffle=False, mode=MODE.TRAIN)
    assert len(ds.data) == len(records)
    batch = ds.collate([ds[i] for i in range(len(ds.data))])
    # coordinate slots (mask==1) must line up 1:1 with the flat (N,3) coords.
    assert int(batch["coordinates_mask"].sum()) == batch["input_coordinates"].shape[0]
    assert batch["input_coordinates"].shape[-1] == 3

    device = "cuda" if _HAS_CUDA else "cpu"
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

    assert all(np.isfinite(losses)), f"non-finite {target} training loss: {losses}"
