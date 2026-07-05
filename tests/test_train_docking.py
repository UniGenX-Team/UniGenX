# -*- coding: utf-8 -*-
"""T3 docking training gates (dock + misato) for the UniGenX training path.

Both docking targets share ``dict_dock.txt`` (vocab 126 per DICT_MAP: 119
non-empty lines + 7 specials). For each target this verifies the *real*
per-target data path end to end:

  * ``dict_dock.txt`` loads to vocab 126,
  * the per-target ``get_train_item_*`` + ``collate`` produce a coherent TRAIN
    batch (coordinate slots line up 1:1 with ``coordinates_mask``), from a
    minimal synthetic on-disk fixture (LMDB for dock, pkl+hdf5 for misato), and
  * a tiny random ``UniGenX`` takes optimizer steps over that batch with a
    finite (non-NaN/Inf) loss.

A separate guard test asserts that exercising the dock data path pulls no
``modules.dock`` / ``dataset_dock`` module into ``sys.modules`` (the docking
results in the paper do not flow through that legacy code).

Conventions mirror tests/test_training.py + tests/test_train_diffcore.py:
``REPO_ROOT`` on ``sys.path``, repo-relative dict paths, the same tiny model
recipe, and a CUDA skipif on the train step (``diffloss`` allocates its noise on
the active CUDA device). The vocab contract is additionally checked in a
dependency-free test that never touches CUDA.
"""
import os
import pickle
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

# dock and misato share dict_dock.txt (119 non-empty lines + 7 specials).
_DOCK_DICT = "dict_dock.txt"
_DOCK_VOCAB = 126

# misato pocket atoms: C N O C S -> all present in dict_dock.txt.
_MISATO_POCKET_Z = [6, 7, 8, 6, 16]
# self.data enumerates frame%100 x 200; reading dataset indices 0..k requires
# the mol to carry conformers 0..k and the trajectory that many frames.
_MISATO_NFRAMES = 3


def _build_config(target):
    from unigenx.model.config import UniGenXConfig

    config = UniGenXConfig(
        vocab_size=_DOCK_VOCAB,
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


# --------------------------------------------------------------------------- #
# Synthetic on-disk fixtures.
# --------------------------------------------------------------------------- #
def _build_dock_dir(root):
    """Minimal dock TRAIN dir: only ``ligands/train.lmdb`` (subdir=False file).

    The ligand branch of ``get_train_item_docking`` is the cheapest of the four
    dock sources: it uses only ``<LIG_*>`` + SMILES-regex tokens (all in
    dict_dock.txt), needs no ``<PROT_*>`` tokens and no RDKit mol building. Each
    record carries 10 conformers (the branch loops ``rep in range(10)``), so 3
    keys -> 30 TRAIN items.
    """
    import lmdb

    lig_dir = os.path.join(root, "ligands")
    os.makedirs(lig_dir, exist_ok=True)
    lmdb_path = os.path.join(lig_dir, "train.lmdb")  # split='train' for MODE.TRAIN
    rng = np.random.default_rng(0)
    atoms = ["C", "C", "O"]  # heavy atoms match SMILES "CCO"
    env = lmdb.open(lmdb_path, subdir=False, map_size=1 << 24)
    with env.begin(write=True) as txn:
        for i in range(3):
            coords10 = (rng.standard_normal((10, len(atoms), 3)) * 2.0).astype(
                np.float32
            )
            txn.put(
                f"lig{i}".encode(),
                pickle.dumps(
                    {"smi": "CCO", "atoms": list(atoms), "coordinates": coords10}
                ),  # indexable [0..9]
            )
    env.sync()
    env.close()
    return root


def _build_misato_dir(root):
    """Minimal synthetic misato dir: ``train_mols.pkl`` + ``MD_pockets.hdf5``."""
    import h5py
    from rdkit import Chem

    d = Path(root)
    d.mkdir(parents=True, exist_ok=True)
    n_pocket = len(_MISATO_POCKET_Z)
    rng = np.random.default_rng(0)

    mol = Chem.MolFromSmiles("CCO")  # 3 heavy atoms == n_lig
    natoms = mol.GetNumAtoms()
    for _ in range(_MISATO_NFRAMES):
        conf = Chem.Conformer(natoms)
        for a in range(natoms):
            conf.SetAtomPosition(a, tuple(float(x) for x in rng.normal(size=3)))
        mol.AddConformer(conf, assignId=True)  # conformer ids 0.._NFRAMES-1
    with open(d / "train_mols.pkl", "wb") as fh:
        pickle.dump({"PDB0": {"smi": "CCO", "mol": mol}}, fh)

    with h5py.File(d / "MD_pockets.hdf5", "w") as h5:
        g = h5.create_group("PDB0")
        # [-1] = ligand-begin = n_pocket
        g["molecules_begin_atom_index"] = np.array([n_pocket], dtype=np.int64)
        g["atoms_number"] = np.array(_MISATO_POCKET_Z, dtype=np.int64)
        g["trajectory_coordinates"] = rng.normal(
            size=(_MISATO_NFRAMES, n_pocket, 3)
        ).astype(np.float32)
        # exactly n_pocket rows (train assert len(coordinates)==len(tags))
        g["apo_pocket_coordinates"] = rng.normal(size=(n_pocket, 3)).astype(np.float32)
    return str(d)


# --------------------------------------------------------------------------- #
# Dependency-free: DICT_MAP vocab contract (never touches CUDA).
# --------------------------------------------------------------------------- #
def test_dict_dock_vocab():
    """dict_dock.txt loads to the DICT_MAP vocab (dock/misato share it)."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    config = _build_config("dock")
    tokenizer = UniGenXTokenizer.from_file(str(DATA_DIR / _DOCK_DICT), config)
    assert (
        len(tokenizer) == _DOCK_VOCAB
    ), f"dict_dock.txt vocab {len(tokenizer)} != DICT_MAP {_DOCK_VOCAB}"


# --------------------------------------------------------------------------- #
# Guard: the dock data path must not pull the legacy docking module.
# --------------------------------------------------------------------------- #
def test_dock_no_dock_module_import():
    """Exercising the dock path pulls no modules.dock / dataset_dock module."""
    from unigenx.data.dataset import MODE, UniGenXDataset  # noqa: F401
    from unigenx.data.tokenizer import UniGenXTokenizer  # noqa: F401

    leaked = [m for m in sys.modules if "modules.dock" in m or "dataset_dock" in m]
    assert not leaked, f"dock path pulled forbidden legacy module(s): {leaked}"


# --------------------------------------------------------------------------- #
# dock tiny-train gate: real LMDB ligand collate + finite loss.
# CUDA-gated: diffloss allocates its noise on the active CUDA device.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _HAS_CUDA,
    reason="diffloss allocates its diffusion noise on cuda; a GPU is required",
)
def test_dock_tiny_train_finite_loss(tmp_path):
    from unigenx.data.dataset import MODE, UniGenXDataset
    from unigenx.data.tokenizer import UniGenXTokenizer
    from unigenx.model.wrapper import UniGenX

    config = _build_config("dock")
    tokenizer = UniGenXTokenizer.from_file(str(DATA_DIR / _DOCK_DICT), config)
    assert len(tokenizer) == _DOCK_VOCAB
    config.mask_token_id = tokenizer.mask_idx

    _build_dock_dir(str(tmp_path))
    ds = UniGenXDataset(
        tokenizer, str(tmp_path), config, shuffle=False, mode=MODE.TRAIN
    )
    assert len(ds.data) == 30  # 3 keys * 10 reps
    items = [ds[i] for i in range(len(ds.data))]
    assert all(it is not None for it in items)
    batch = ds.collate(items)
    # Only mask==1 rows carry coordinates (the ligand branch also writes mask==0
    # slots for tokens), so compare eq(1) count to the flat (N,3) coord rows.
    assert (
        int(batch["coordinates_mask"].eq(1).sum())
        == batch["input_coordinates"].shape[0]
    )
    assert batch["input_coordinates"].shape[-1] == 3
    assert not [m for m in sys.modules if "modules.dock" in m or "dataset_dock" in m]

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

    assert all(np.isfinite(losses)), f"non-finite dock training loss: {losses}"


# --------------------------------------------------------------------------- #
# misato tiny-train gate: real pkl+hdf5 collate + finite loss.
# CUDA-gated: diffloss allocates its noise on the active CUDA device.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _HAS_CUDA,
    reason="diffloss allocates its diffusion noise on cuda; a GPU is required",
)
def test_misato_tiny_train_finite_loss(tmp_path):
    from unigenx.data.dataset import MODE, UniGenXDataset
    from unigenx.data.tokenizer import UniGenXTokenizer
    from unigenx.model.wrapper import UniGenX

    config = _build_config("misato")
    tokenizer = UniGenXTokenizer.from_file(str(DATA_DIR / _DOCK_DICT), config)
    assert len(tokenizer) == _DOCK_VOCAB
    config.mask_token_id = tokenizer.mask_idx

    ddir = _build_misato_dir(tmp_path / "misato")
    ds = UniGenXDataset(tokenizer, ddir, config, shuffle=False, mode=MODE.TRAIN)
    # self.data enumerates frame%100 x 200; indices 0..2 -> frames 0,1,2.
    items = [ds[i] for i in range(_MISATO_NFRAMES)]
    assert all(it is not None for it in items)
    batch = ds.collate(items)
    assert (
        int(batch["coordinates_mask"].bool().sum())
        == batch["input_coordinates"].shape[0]
    )
    assert batch["input_coordinates"].shape[-1] == 3

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

    assert all(np.isfinite(losses)), f"non-finite misato training loss: {losses}"
