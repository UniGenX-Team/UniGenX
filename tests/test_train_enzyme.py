# -*- coding: utf-8 -*-
"""T5 training gate for the EC-number conditioned enzyme (``ecnum``) path.

The enzyme target is *sequence-only*: an EC number (split on "." into its first
three levels) conditions the generation of an amino-acid sequence, with no 3D
coordinates at all --

    <bos> <ec1> L1 <ec2> L2 <ec3> L3 <prot> {residues} <eos>

so ``get_train_item_ecnum`` lays down an all-zero ``coordinates_mask`` and never
emits a ``coordinates`` field.

This file mirrors the conventions of tests/test_train_diffcore.py and
tests/test_training.py (``REPO_ROOT`` on ``sys.path``, repo-relative dict paths,
the same tiny model recipe, a CUDA skipif on the train step) and the
checkpoint-embedding check of tests/test_enzyme.py.

Gates (TRAINING_PLAN.md Section 6 T5):
  * vocab == 64                 -- dict_ecnum.txt (57 non-empty lines + 7
                                   specials); matches the ``e`` / ``e_wo``
                                   checkpoints' embedding dim0. Dependency-free,
                                   never touches CUDA, always runs.
  * collation                   -- TRAIN collate is sequence-only: an all-zero
                                   ``coordinates_mask`` and no ``input_coordinates``
                                   tensor. Dependency-free.
  * e / e_wo embed dim0 == 64   -- ground truth, skip-if-absent.
  * tiny train -> finite loss   -- CUDA-gated. Sequence-only training works
                                   because two guards handle the no-coordinate
                                   case: ``UniGenX.forward`` skips the diffusion
                                   branch when ``input_coordinates`` is None
                                   (unigenx.py), and ``CrystalCriterions`` returns
                                   ``loss_words`` only when ``loss_coord`` is None
                                   (criterions.py). Both are shared with inference
                                   but only add None-checks, so inference is
                                   unaffected (its generate path never hits them).
"""
import json
import os
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
DICT_ECNUM = DATA_DIR / "dict_ecnum.txt"

# ground truth from the checkpoints (e / e_wo embedding dim0 == 64) and the
# vocab-counting rule: dict_ecnum.txt has 57 non-empty lines -> vocab 64.
ECNUM_VOCAB = 64
ECNUM_DICT_LINES = 57

# 20 standard amino acids + unknown, in dict_ecnum.txt order.
RESIDUES = list("ARNDCQEGHILKMFPSTWYVX")

_WORKSPACE = REPO_ROOT.parent
_CKPT_DIR = Path(os.environ.get("UNIGENX_CHECKPOINTS", str(_WORKSPACE / "checkpoints")))
EC_CHECKPOINTS = ["e.pt", "e_wo.pt"]

# Synthetic TRAIN records: EC numbers split on "." to their first three levels,
# short amino-acid sequences. The residue field may be keyed "seq" or "aa".
_ECNUM_RECORDS = [
    {"id": 0, "EC_number": "1.1.1.1", "seq": "AGCCEK"},
    {"id": 1, "EC_number": "2.7.11.1", "aa": "MKLVWY"},
    {"id": 2, "EC_number": "3.4.21.4", "seq": "PSTQEND"},
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _build_ecnum_config(vocab=ECNUM_VOCAB):
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
    config.target = "ecnum"
    config.tokenizer = "num"
    # sequence-only target: no coordinate augmentation / crystal knobs apply.
    config.space_group = False
    config.reorder = False
    config.rotation_augmentation = False
    config.translation_augmentation = False
    config.scale_coords = None
    config.max_sites = None
    config.total_num_steps = 10
    config.warmup_num_steps = 2
    config.max_lr = 1e-4
    return config


def _load_checkpoint_container(path):
    """Load a checkpoint state dict, tolerating a saved args object whose
    original class is not shipped with this package (mirrors
    tests/test_enzyme.py)."""
    from unigenx.utils.checkpoint import load_checkpoint

    state = load_checkpoint(path)
    container = state
    if isinstance(state, dict):
        for key in ("model", "module", "state_dict"):
            if key in state and isinstance(state[key], dict):
                container = state[key]
                break
    return container


def _load_train_dataset(records, tmp_path):
    from unigenx.data.dataset import MODE, UniGenXDataset
    from unigenx.data.tokenizer import UniGenXTokenizer

    config = _build_ecnum_config()
    tokenizer = UniGenXTokenizer.from_file(str(DICT_ECNUM), config)
    jsonl = tmp_path / "ecnum.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    ds = UniGenXDataset(tokenizer, str(jsonl), config, shuffle=False, mode=MODE.TRAIN)
    return ds, tokenizer, config


# --------------------------------------------------------------------------- #
# vocab == 64 (dependency-free; the T5 dict gate, always runs)
# --------------------------------------------------------------------------- #
def test_ecnum_dict_vocab():
    from unigenx.data.tokenizer import UniGenXTokenizer

    config = _build_ecnum_config()
    tokenizer = UniGenXTokenizer.from_file(str(DICT_ECNUM), config)
    assert len(tokenizer) == ECNUM_VOCAB, (
        f"dict_ecnum vocab {len(tokenizer)} != {ECNUM_VOCAB}; the e / e_wo "
        "checkpoints need a 57-line dict (vocab = lines + 7 special tokens)"
    )
    # also without a config, since from_file(args=None) must give the same vocab
    assert len(UniGenXTokenizer.from_file(str(DICT_ECNUM))) == ECNUM_VOCAB


def test_ecnum_dict_vocab_rule_holds():
    """vocab == non-empty lines + 7. dict_ecnum.txt has 57 non-empty lines."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    assert DICT_ECNUM.exists(), f"missing committed dict: {DICT_ECNUM}"
    lines = [ln for ln in DICT_ECNUM.read_text().splitlines() if ln.strip()]
    tokenizer = UniGenXTokenizer.from_file(str(DICT_ECNUM))
    assert len(tokenizer) == len(lines) + 7
    assert len(lines) == ECNUM_DICT_LINES, (
        f"dict_ecnum has {len(lines)} non-empty lines; expected "
        f"{ECNUM_DICT_LINES} (=> vocab {ECNUM_VOCAB})"
    )


# --------------------------------------------------------------------------- #
# collation: sequence-only TRAIN batch (dependency-free; the T5 collate gate)
# --------------------------------------------------------------------------- #
def test_ecnum_train_collation_is_sequence_only(tmp_path):
    from unigenx.data.dataset import MODE  # noqa: F401

    ds, tokenizer, _ = _load_train_dataset(_ECNUM_RECORDS, tmp_path)
    assert len(ds.data) == len(_ECNUM_RECORDS)

    # per-item: an all-zero coordinate mask, and no coordinate field at all.
    for i in range(len(ds.data)):
        item = ds[i]
        assert "coordinates" not in item
        assert int(np.asarray(item["coordinates_mask"]).sum()) == 0
        assert len(item["coordinates_mask"]) == len(item["tokens"])

    batch = ds.collate([ds[i] for i in range(len(ds.data))])
    assert "input_ids" in batch and "coordinates_mask" in batch
    assert batch["input_ids"].shape[0] == len(_ECNUM_RECORDS)
    # sequence-only: the collate emits an all-zero coordinate mask and NO
    # coordinate tensors (this all-zero-but-present mask is exactly what trips
    # the forward/criterion guards documented at the top of this file).
    assert int(batch["coordinates_mask"].sum()) == 0
    assert "input_coordinates" not in batch
    assert "label_coordinates" not in batch
    # TRAIN pads on the right, so the mask is as wide as the padded token ids.
    assert batch["coordinates_mask"].shape[1] == batch["input_ids"].shape[1]


def test_ecnum_train_item_layout(tmp_path):
    """<bos> <ec1> L1 <ec2> L2 <ec3> L3 <prot> {residues} <eos>, all-zero mask."""
    ds, tok, _ = _load_train_dataset([_ECNUM_RECORDS[0]], tmp_path)
    item = ds[0]  # EC 1.1.1.1, seq AGCCEK
    toks = list(item["tokens"])
    seq = _ECNUM_RECORDS[0]["seq"]

    prefix = [
        tok.bos_idx,
        tok.get_idx("<ec1>"),
        tok.get_idx("1"),
        tok.get_idx("<ec2>"),
        tok.get_idx("1"),
        tok.get_idx("<ec3>"),
        tok.get_idx("1"),
        tok.get_idx("<prot>"),
    ]
    assert toks[:8] == prefix
    assert toks[8 : 8 + len(seq)] == [tok.get_idx(r) for r in seq]
    assert toks[-1] == tok.eos_idx
    assert len(toks) == 8 + len(seq) + 1
    assert tok.unk_idx not in toks
    assert int(np.asarray(item["coordinates_mask"]).sum()) == 0


def test_ecnum_residue_and_marker_tokens_resolve():
    """EC markers / levels / residues must be real tokens (no silent <unk>)."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    tok = UniGenXTokenizer.from_file(str(DICT_ECNUM))
    for marker in ("<ec1>", "<ec2>", "<ec3>", "<prot>"):
        assert tok.get_idx(marker) != tok.unk_idx, f"{marker} must be a real token"
    for level in ("1", "2", "7", "11"):
        assert tok.get_idx(level) != tok.unk_idx, f"EC level {level} must resolve"
    for res in RESIDUES:
        assert tok.get_idx(res) != tok.unk_idx, f"residue {res} must be a real token"


# --------------------------------------------------------------------------- #
# e / e_wo checkpoint embedding vocab == 64 (ground truth, skip-if-absent)
# --------------------------------------------------------------------------- #
def test_ecnum_checkpoint_embedding_vocab():
    ckpt = next(
        (_CKPT_DIR / n for n in EC_CHECKPOINTS if (_CKPT_DIR / n).exists()), None
    )
    if ckpt is None:
        pytest.skip(f"no enzyme checkpoint {EC_CHECKPOINTS} under {_CKPT_DIR}")

    try:
        container = _load_checkpoint_container(ckpt)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot load {ckpt.name} ({type(e).__name__}: {e})")

    matches = [
        k for k in container if isinstance(k, str) and k.endswith("embed_tokens.weight")
    ]
    assert matches, f"{ckpt.name}: no *embed_tokens.weight in state dict"
    for k in matches:
        assert container[k].shape[0] == ECNUM_VOCAB, (
            f"{ckpt.name}:{k} embedding dim0 {container[k].shape[0]} "
            f"!= {ECNUM_VOCAB} (enzyme checkpoints require dict_ecnum vocab 64)"
        )


# --------------------------------------------------------------------------- #
# tiny train -> finite loss. CUDA-gated (diffloss allocates its noise on cuda).
#
# Sequence-only ecnum training: the forward diffusion branch is skipped
# (input_coordinates is None) and CrystalCriterions returns loss_words only
# (loss_coord is None), so a tiny train step yields a finite loss.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _HAS_CUDA,
    reason="diffloss allocates its diffusion noise on cuda; a GPU is required",
)
def test_ecnum_tiny_train_finite_loss(tmp_path):
    from unigenx.model.wrapper import UniGenX

    ds, tokenizer, config = _load_train_dataset(_ECNUM_RECORDS, tmp_path)
    config.mask_token_id = tokenizer.mask_idx

    batch = ds.collate([ds[i] for i in range(len(ds.data))])
    # precondition: the batch really is sequence-only (all-zero mask, no coords).
    assert int(batch["coordinates_mask"].sum()) == 0
    assert "input_coordinates" not in batch

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

    assert all(np.isfinite(losses)), f"non-finite ecnum training loss: {losses}"
