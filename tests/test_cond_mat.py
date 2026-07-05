# -*- coding: utf-8 -*-
"""Stage-3 (conditional multi-property material design) smoke tests.

Covers RELEASE_PLAN.md Section 3 Definition-of-Done for the multi-property
material design path (target ``cond_mat``). Classifier-free guidance (CFG) was
dropped from the release by request, so cond_mat runs the conditional model
directly; these tests cover the dict, collation, and eval pieces.

  3. dict vocab assertion -- ``dict_cond_mat.txt`` (the property-token material
                             dict) is vocab 355 and its property markers
                             (<band>/<bulk>/... at ids 122-128) resolve to real
                             tokens; ``dict_mat.txt`` stays vocab 355 and is left
                             untouched by this stage. Plus a skip-if-absent check
                             that the cond_mat checkpoints
                             (mc_mat / bs_mat_1 / ms_mat_1) have
                             ``embed_tokens.weight`` dim0 == 355.
  4. collation            -- the property conditioning is a *continuous value*
                             fed through the coordinate stream (coordinate slot
                             carrying [prop_val]*3), preceded by a property
                             marker token. The lattice still occupies material-
                             style coordinate slots (right after the single
                             property slot). Both get_train_cond_mat and
                             get_infer_cond_mat are checked, including the
                             ``<prop> propval <bos>`` prefix order and the
                             training-consistent value standardization.
  6. eval                 -- the ASE EOS / Murnaghan bulk-modulus fit recovers a
                             known modulus from a toy energy-volume curve; the
                             CHGNet relax utilities import behind importorskip.

The dict and collation tests stay green with no external dependencies. The
checkpoint test skips when the (multi-GB) cond_mat checkpoints are absent; the
relax test skips without CHGNet.
"""
import importlib.util
import json
import os
import sys
from math import log
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "unigenx" / "data"
COMPUTE_BULK_PY = REPO_ROOT / "eval" / "material" / "compute_bulk.py"
RELAX_PY = REPO_ROOT / "eval" / "material" / "relax.py"

# vocab = non-empty dict lines + 7 special tokens
DICT_MAT_VOCAB = 355

_WORKSPACE = REPO_ROOT.parent
_CKPT_DIR = Path(os.environ.get("UNIGENX_CHECKPOINTS", str(_WORKSPACE / "checkpoints")))
COND_MAT_CHECKPOINTS = ["mc_mat.pt", "bs_mat_1.pt", "ms_mat_1.pt"]

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


def _cond_mat_config(target="cond_mat"):
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


def _load_dataset(path, mode, target="cond_mat"):
    from unigenx.data.dataset import MODE, UniGenXDataset
    from unigenx.data.tokenizer import UniGenXTokenizer

    cfg = _cond_mat_config(target)
    tok = UniGenXTokenizer.from_file(str(DATA_DIR / "dict_cond_mat.txt"), cfg)
    m = MODE.TRAIN if mode == "train" else MODE.INFER
    ds = UniGenXDataset(tok, path, args=cfg, shuffle=False, mode=m)
    return ds, tok


def _load_checkpoint_container(path):
    """Load a (DeepSpeed) checkpoint state dict, tolerating a saved args object
    whose original class is not shipped with this package."""
    from unigenx.utils.checkpoint import load_checkpoint

    state = load_checkpoint(path)
    container = state
    if isinstance(state, dict):
        for key in ("model", "module", "state_dict"):
            if key in state and isinstance(state[key], dict):
                container = state[key]
                break
    return container


# Toy conditional-material records. TRAIN reads data_item["property"] (a dict
# whose key matches dft_(.*?)_); INFER reads data_item["prop"]/["prop_val"].
_TOY_TRAIN = [
    {
        "id": 0,
        "formula": "Na1Cl1",
        "lattice": [[5.64, 0.0, 0.0], [0.0, 5.64, 0.0], [0.0, 0.0, 5.64]],
        "sites": [
            {"element": "Na", "fractional_coordinates": [0.0, 0.0, 0.0]},
            {"element": "Cl", "fractional_coordinates": [0.5, 0.5, 0.5]},
        ],
        "property": {"dft_bulk_modulus": 100.0},
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
        "property": {"dft_bulk_modulus": 250.0},
    },
]

_TOY_INFER = [
    {"id": 0, "prop": "bulk", "prop_val": 100.0},
    {"id": 1, "prop": "bulk", "prop_val": 250.0},
]


@pytest.fixture(scope="module")
def toy_train_jsonl(tmp_path_factory):
    path = tmp_path_factory.mktemp("cond_mat") / "train_toy.jsonl"
    with open(path, "w") as f:
        for rec in _TOY_TRAIN:
            f.write(json.dumps(rec) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def toy_infer_jsonl(tmp_path_factory):
    path = tmp_path_factory.mktemp("cond_mat") / "infer_toy.jsonl"
    with open(path, "w") as f:
        for rec in _TOY_INFER:
            f.write(json.dumps(rec) + "\n")
    return str(path)


# --------------------------------------------------------------------------- #
# DoD 3: dict vocab must stay 355 (invariant preserved by this stage)
# --------------------------------------------------------------------------- #
def test_dict_mat_vocab_unchanged():
    from unigenx.data.tokenizer import UniGenXTokenizer

    path = DATA_DIR / "dict_mat.txt"
    assert path.exists(), f"missing committed dict: {path}"
    tok = UniGenXTokenizer.from_file(str(path))
    assert len(tok) == DICT_MAT_VOCAB, (
        f"dict_mat.txt: expected vocab {DICT_MAT_VOCAB}, got {len(tok)} "
        "(stage 3 must NOT add tokens to dict_mat.txt)"
    )


def test_dict_cond_mat_markers_resolve():
    """cond_mat uses the property-token dict ``dict_cond_mat.txt`` (also vocab
    355): the property markers <band>/<bulk>/<mag>/... are real tokens (ids
    122-128), so the conditioning marker is not silently lost. In ``dict_mat.txt``
    those same ids are <sgn>1..7, so the markers would fall back to <unk> -- which
    is exactly why cond_mat needs its own vocab-355 dict."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    tok = UniGenXTokenizer.from_file(str(DATA_DIR / "dict_cond_mat.txt"))
    assert len(tok) == DICT_MAT_VOCAB  # property-token dict is also vocab 355
    for marker in ("<band>", "<bulk>", "<mag>", "<heat_capacity_300K>"):
        assert tok.get_idx(marker) != tok.unk_idx, f"{marker} must be a real token"

    # contrast: the plain material dict does NOT carry the property markers
    tok_mat = UniGenXTokenizer.from_file(str(DATA_DIR / "dict_mat.txt"))
    assert tok_mat.get_idx("<bulk>") == tok_mat.unk_idx


# --------------------------------------------------------------------------- #
# DoD 3 (additional): cond_mat checkpoint embedding vocab == 355, skip-if-absent
# --------------------------------------------------------------------------- #
def test_cond_mat_checkpoint_embedding_vocab():
    present = [_CKPT_DIR / n for n in COND_MAT_CHECKPOINTS if (_CKPT_DIR / n).exists()]
    if not present:
        pytest.skip(f"no cond_mat checkpoints under {_CKPT_DIR}")

    for ckpt in present:
        try:
            container = _load_checkpoint_container(ckpt)
        except Exception as e:  # pragma: no cover - environment dependent
            pytest.skip(f"cannot load {ckpt.name} ({type(e).__name__}: {e})")

        matches = [
            k
            for k in container
            if isinstance(k, str) and k.endswith("embed_tokens.weight")
        ]
        assert matches, f"{ckpt.name}: no *embed_tokens.weight in state dict"
        for k in matches:
            assert container[k].shape[0] == DICT_MAT_VOCAB, (
                f"{ckpt.name}:{k} embedding dim0 {container[k].shape[0]} "
                f"!= {DICT_MAT_VOCAB} (cond_mat checkpoints are vocab 355)"
            )


# --------------------------------------------------------------------------- #
# DoD 4: collation -- property conditioning is an MLP-embedded continuous value
# --------------------------------------------------------------------------- #
def test_cond_mat_train_collation(toy_train_jsonl):
    ds, tok = _load_dataset(toy_train_jsonl, "train")
    assert len(ds.data) == len(_TOY_TRAIN)

    for idx in range(len(ds.data)):
        n = len(ds.data[idx]["sites"])
        item = ds.get_train_item(idx)
        assert {"tokens", "coordinates", "coordinates_mask"} <= set(item)

        toks = item["tokens"]
        coords = item["coordinates"]
        mask = item["coordinates_mask"]

        # token layout:
        #   <prop> <mask(propval)> <bos> [n elems] <coord> [3 lat mask] [n mask] <eos>
        assert len(toks) == 2 * n + 8
        # property prefix: marker token then a coordinate (mask=1) value slot
        assert toks[0] == tok.get_idx(
            "<bulk>"
        )  # property marker (real id in dict_cond_mat)
        assert toks[1] == tok.mask_idx
        assert toks[2] == tok.bos_idx
        assert toks[n + 3] == tok.coord_idx
        assert toks[-1] == tok.eos_idx

        # mask marks: the propval slot (idx 1) + the 3 lattice + n atom slots
        assert mask[0] == 0 and mask[1] == 1 and mask[2] == 0
        assert int(mask.sum()) == n + 4

        # ---- conditioning is a CONTINUOUS value in the coordinate stream ----
        # coordinates = [prop_val(x3), lattice(3x3), atom fractional coords(n x3)]
        assert coords.shape == (n + 4, 3)
        prop_val = log(ds.data[idx]["property"]["dft_bulk_modulus"] + 1)  # bulk std.
        assert np.allclose(coords[0], [prop_val, prop_val, prop_val])
        # lattice still occupies material-style slots (right after the prop slot)
        lattice = np.array(ds.data[idx]["lattice"], dtype=np.float32)
        assert np.allclose(coords[1:4], lattice)


def test_cond_mat_infer_collation(toy_infer_jsonl):
    ds, tok = _load_dataset(toy_infer_jsonl, "infer")
    assert len(ds.data) == len(_TOY_INFER)

    for idx in range(len(ds.data)):
        item = ds.get_infer_item(idx)
        toks = item["tokens"]
        mask = item["coordinates_mask"]
        coords = item["coordinates"]

        # prompt order MUST be <prop> propval <bos> (matches training), not bos-first
        assert len(toks) == 3
        assert toks[0] == tok.get_idx("<bulk>")
        assert toks[1] == tok.mask_idx
        assert toks[2] == tok.bos_idx
        assert list(mask) == [0, 1, 0]

        # value is standardized exactly as in training (bulk -> log(v+1))
        prop_val = log(_TOY_INFER[idx]["prop_val"] + 1)
        assert coords.shape == (1, 3)
        assert np.allclose(coords[0], [prop_val, prop_val, prop_val])


def test_cond_mat_decode_splits_cond_val():
    """decode_batch(entity='cond_mat') returns (sent, cond_val, lattice, atoms):
    coordinate slot 0 is the conditioning value, slots 1-3 the lattice."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    tok = UniGenXTokenizer.from_file(str(DATA_DIR / "dict_cond_mat.txt"))
    na, cl = tok.get_idx("Na"), tok.get_idx("Cl")
    # tokens: <prop> <mask> <bos> Na Cl <coord> m m m m m <eos>
    tokens = np.array(
        [
            tok.get_idx("<bulk>"),
            tok.mask_idx,
            tok.bos_idx,
            na,
            cl,
            tok.coord_idx,
            tok.mask_idx,
            tok.mask_idx,
            tok.mask_idx,
            tok.mask_idx,
            tok.mask_idx,
            tok.eos_idx,
        ]
    )
    mask = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
    # coordinate stream for the 6 mask=1 slots: [prop, lat1, lat2, lat3, a1, a2]
    coords = np.array(
        [
            [7.0, 7.0, 7.0],  # cond_val
            [5.64, 0.0, 0.0],
            [0.0, 5.64, 0.0],
            [0.0, 0.0, 5.64],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ]
    )
    out = tok.decode_batch(tokens[None], coords, mask[None], "cond_mat")
    assert len(out) == 1
    sent, cond_val, lattice, atoms = out[0]
    assert np.allclose(cond_val, [7.0, 7.0, 7.0])
    assert len(lattice) == 3 and np.allclose(lattice[0], [5.64, 0.0, 0.0])
    assert len(atoms) == 2 and np.allclose(atoms[1], [0.5, 0.5, 0.5])


# --------------------------------------------------------------------------- #
# DoD 6: eval -- ASE EOS/Murnaghan bulk-modulus fit + relax utilities
# --------------------------------------------------------------------------- #
def _murnaghan_energy(V, E0, B0, B0p, V0):
    # standard Murnaghan EOS energy (B0 in energy/volume units)
    return E0 + B0 * V / B0p * ((V0 / V) ** B0p / (B0p - 1) + 1) - B0 * V0 / (B0p - 1)


def test_compute_bulk_toy_eos_fit():
    pytest.importorskip("ase")
    mod = _load_module(COMPUTE_BULK_PY, "compute_bulk")

    from ase.units import GPa

    # Generate exact Murnaghan energy-volume points with a known bulk modulus,
    # then fit and check recovery.
    V0, E0, B0p = 40.0, -10.0, 4.0
    B0 = 0.5  # eV/Angstrom^3  (~80 GPa)
    volumes = np.linspace(0.94 * V0, 1.06 * V0, 7)
    energies = [_murnaghan_energy(v, E0, B0, B0p, V0) for v in volumes]

    v0_fit, e0_fit, B_gpa = mod.fit_bulk_modulus(volumes, energies, eos="murnaghan")
    expected_gpa = B0 / GPa
    assert B_gpa > 0
    assert abs(B_gpa - expected_gpa) / expected_gpa < 0.05
    assert abs(v0_fit - V0) / V0 < 0.02


def test_compute_bulk_pipeline_with_injected_energy():
    """compute_bulk_modulus end to end with an injected (toy) energy predictor,
    so no ML potential / GPU is needed."""
    pytest.importorskip("ase")
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure

    mod = _load_module(COMPUTE_BULK_PY, "compute_bulk")

    struct = Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    V0 = struct.volume
    B0, B0p, E0 = 0.5, 4.0, -10.0

    def energy_predictor(atoms_list, batch_size):
        return [_murnaghan_energy(a.get_volume(), E0, B0, B0p, V0) for a in atoms_list]

    bulk_moduli, failed = mod.compute_bulk_modulus(
        [struct], energy_predictor=energy_predictor, npoints=5, eps_max=0.03
    )
    assert failed == []
    assert bulk_moduli.shape == (1,)
    assert bulk_moduli[0] > 0


def test_relax_helper_builds_structure():
    """The pymatgen-only helper works without CHGNet installed."""
    pytest.importorskip("pymatgen")
    mod = _load_module(RELAX_PY, "relax_mod")
    data = {
        "sites": [
            {"element": "Na", "fractional_coordinates": [0.0, 0.0, 0.0]},
            {"element": "Cl", "fractional_coordinates": [0.5, 0.5, 0.5]},
        ],
        "prediction": {
            "lattice": [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
            "coordinates": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        },
    }
    structure = mod.get_pred_structure_from_coords(data)
    assert len(structure) == 2
    assert [str(s.specie) for s in structure] == ["Na", "Cl"]


def test_relax_requires_chgnet():
    """CHGNet is an optional external dependency (not installed here)."""
    pytest.importorskip("chgnet")
    mod = _load_module(RELAX_PY, "relax_mod2")
    assert hasattr(mod, "main")
