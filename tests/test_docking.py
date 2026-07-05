# -*- coding: utf-8 -*-
"""Stage-7 (protein-ligand docking) smoke tests.

Covers RELEASE_PLAN.md Section 3 Definition-of-Done for the docking path
(targets ``dock`` / ``misato``, checkpoints pld / pld_u, vocab 126):

  3. dict vocab assertion -- ``dict_dock.txt`` is vocab 126 (119 dict lines + 7
                             special tokens <pad><bos><eos><unk> + <mask><coord>
                             <sg>). Plus a skip-if-present check that pld / pld_u
                             ``embed_tokens.weight`` dim0 == 126.
  4. collation            -- self-made dock + misato fixtures pass through
                             get_{infer,simple}_item_{docking,misato}; the
                             apo (mask 2) / holo (mask 1) / ligand (mask 3)
                             layout and the "generate-the-rest" coordinate mask
                             are asserted.
  5. inference dry-run    -- a tiny random model runs the docking generate path
                             for the pocket-given (misato) and dock setups
                             (CUDA-gated).
  6. eval                 -- calc_rmsd (naive RMSD): identical coords -> 0, a
                             known offset -> hand-computed value; the best-of-N
                             per-target aggregation and the evaluate_docking
                             thresholds [2,4,6,8,10] are checked on a toy jsonl.
                             The optional symmetry-corrected RMSD path is
                             importorskip'd (spyrmsd is not installed).

Notable dict gap (see report): the plain ``dock``/unimol inference path uses
``<PROT_COORDS_START>`` / ``<PROT_COORDS_END>`` which are ABSENT from
dict_dock.txt (they collapse to <unk> -> get_infer_item_docking returns None).
The misato path -- the one the paper docking numbers are evaluated on -- uses
<PROT_APO_COORDS_*> / <PROT_HOLO_COORDS_*> which all resolve. The dock-layout
test therefore adds the two missing tokens to verify the ported layout, and a
separate test documents the real-dict None behaviour.
"""
import base64
import importlib.util
import json
import os
import pickle
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# misato collation maps pocket atomic numbers -> element symbols via
# ``periodictable`` (a pure lookup table). It is an optional runtime dependency
# for the misato path and is not installed in this env; stub the tiny slice the
# collation needs so the apo/holo/ligand layout stays testable. (The same is
# true of ``h5py``: fixtures set ``ds.h5`` to an in-memory dict instead.)
try:
    import periodictable  # noqa: F401
except ImportError:
    _Z2SYM = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S"}

    class _FakeElement:
        def __init__(self, sym):
            self.symbol = sym

    class _FakeElements:
        def __getitem__(self, z):
            return _FakeElement(_Z2SYM[int(z)])

    sys.modules["periodictable"] = types.SimpleNamespace(elements=_FakeElements())

DATA_DIR = REPO_ROOT / "unigenx" / "data"
EVAL_DOCK_DIR = REPO_ROOT / "eval" / "docking"
DICT_DOCK = DATA_DIR / "dict_dock.txt"
GEN_DOCK_SH = REPO_ROOT / "scripts" / "gen_dock.sh"

DOCK_VOCAB = 126  # 119 dict lines + 7 special tokens

_WORKSPACE = REPO_ROOT.parent
_CKPT_DIR = Path(os.environ.get("UNIGENX_CHECKPOINTS", str(_WORKSPACE / "checkpoints")))
DOCK_CHECKPOINTS = ["pld.pt", "pld_u.pt"]

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


def _dock_config(target="misato"):
    from unigenx.model.config import UniGenXConfig

    cfg = UniGenXConfig()
    cfg.target = target
    cfg.space_group = False
    cfg.reorder = False
    cfg.rotation_augmentation = False
    cfg.translation_augmentation = False
    cfg.scale_coords = None
    cfg.max_sites = None
    cfg.max_position_embeddings = 2048
    cfg.tokenizer = "num"
    cfg.smi_rand_aug = 0.0
    return cfg


def _make_dataset(target, tok, cfg, mode=None):
    """Build a UniGenXDataset without touching load_data_from_file (there
    is no on-disk misato dir here -- h5py is not installed -- so fixtures set
    self.data / self.mols / self.h5 / self.txns directly and exercise the real
    get_* methods)."""
    from unigenx.data.dataset import MODE, UniGenXDataset

    ds = UniGenXDataset.__new__(UniGenXDataset)
    ds.tokenizer = tok
    ds.args = cfg
    ds.mode = MODE.INFER if mode is None else mode
    ds.max_position_embeddings = cfg.max_position_embeddings
    ds.data = []
    ds.sizes = []
    ds.env = None
    ds.keys = None
    ds.lig_regex = re.compile(UniGenXDataset._DOCK_SMILES_PATTERN)
    ds.coords_max, ds.coords_min = 20, -20
    return ds


def _make_ligand(smiles="CCO"):
    """A tiny RDKit ligand (no explicit H) with one conformer at known coords."""
    mol = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(float(i) * 1.5, 0.0, 0.0))
    mol.AddConformer(conf, assignId=True)
    return mol


class _FakeTxn:
    """Minimal LMDB-transaction stand-in: get(key) -> pickled bytes."""

    def __init__(self, store):
        self._store = store

    def get(self, key):
        return self._store[key]


# ligand / pocket sizes used across the fixtures
_N_POCKET = 4
_LIG_SMILES = "CCO"  # 3 heavy atoms


def _misato_fixture(ds):
    """Populate a misato dataset: 4-atom pocket (N,C,C,O), 3-atom ligand."""
    key = "1abc"
    ds.mols = {key: {"smi": _LIG_SMILES, "mol": _make_ligand(_LIG_SMILES)}}
    ds.h5 = {
        key: {
            "molecules_begin_atom_index": np.array([_N_POCKET]),
            "atoms_number": np.array([7, 6, 6, 8]),  # N C C O
            "trajectory_coordinates": np.arange(
                2 * _N_POCKET * 3, dtype=np.float64
            ).reshape(2, _N_POCKET, 3),
            "apo_pocket_coordinates": np.arange(
                _N_POCKET * 3, dtype=np.float64
            ).reshape(_N_POCKET, 3)
            + 100.0,
        }
    }
    ds.data = [{"pdb_id": key, "frame": 0}]
    return key


def _dock_fixture(ds):
    """Populate a dock/unimol dataset: 4-atom pocket, 3-atom ligand."""
    lig = _make_ligand(_LIG_SMILES)
    data = {
        "config": {"cx": 0.0, "cy": 0.0, "cz": 0.0},
        "holo_pocket_coordinates": [
            np.arange(_N_POCKET * 3, dtype=np.float64).reshape(_N_POCKET, 3)
        ],
        "holo_mol": lig,
        "pocket_atoms": ["N", "CA", "C", "O"],
    }
    ds.txns = {"unimol": _FakeTxn({b"0": pickle.dumps(data)})}
    ds.data = [{"txn": "unimol", "key": b"0"}]


# --------------------------------------------------------------------------- #
# DoD 3: dict vocab == 126 + docking special tokens resolve
# --------------------------------------------------------------------------- #
def test_dict_dock_line_count():
    lines = [ln for ln in DICT_DOCK.read_text().splitlines() if ln.strip()]
    assert (
        len(lines) == 119
    ), f"dict_dock.txt: expected 119 non-empty lines, got {len(lines)}"


def test_dict_dock_vocab():
    from unigenx.data.tokenizer import UniGenXTokenizer

    tok = UniGenXTokenizer.from_file(str(DICT_DOCK))
    assert len(tok) == DOCK_VOCAB, (
        f"dict_dock.txt: expected vocab {DOCK_VOCAB}, got {len(tok)} "
        "(vocab must equal 119 non-empty lines + 7 special tokens)"
    )


def test_dock_special_tokens_resolve():
    """The misato path's structural tokens must all be real (not <unk>)."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    tok = UniGenXTokenizer.from_file(str(DICT_DOCK))
    for t in (
        "<PROT_START>",
        "<PROT_END>",
        "<LIG_START>",
        "<LIG_END>",
        "<LIG_COORDS_START>",
        "<LIG_COORDS_END>",
        "<PROT_APO_COORDS_START>",
        "<PROT_APO_COORDS_END>",
        "<PROT_HOLO_COORDS_START>",
        "<PROT_HOLO_COORDS_END>",
    ):
        assert tok.get_idx(t) != tok.unk_idx, f"{t} must be a real dict token"


def test_dock_plain_pocket_tokens_absent():
    """Documents the dict gap: the plain dock/unimol path's
    <PROT_COORDS_START>/<PROT_COORDS_END> are NOT in dict_dock (they collapse to
    <unk>). Only the misato APO/HOLO variants are present."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    tok = UniGenXTokenizer.from_file(str(DICT_DOCK))
    assert tok.get_idx("<PROT_COORDS_START>") == tok.unk_idx
    assert tok.get_idx("<PROT_COORDS_END>") == tok.unk_idx


def test_dock_checkpoint_embedding_vocab():
    ckpt = next(
        (_CKPT_DIR / n for n in DOCK_CHECKPOINTS if (_CKPT_DIR / n).exists()), None
    )
    if ckpt is None:
        pytest.skip(f"no docking checkpoint {DOCK_CHECKPOINTS} under {_CKPT_DIR}")

    from unigenx.utils.checkpoint import load_checkpoint

    try:
        state = load_checkpoint(str(ckpt))
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot load {ckpt.name} ({type(e).__name__}: {e})")

    container = state
    if isinstance(state, dict):
        for key in ("module", "model", "state_dict"):
            if key in state and isinstance(state[key], dict):
                container = state[key]
                break
    matches = [
        k for k in container if isinstance(k, str) and k.endswith("embed_tokens.weight")
    ]
    assert matches, f"{ckpt.name}: no *embed_tokens.weight in state dict"
    for k in matches:
        assert container[k].shape[0] == DOCK_VOCAB, (
            f"{ckpt.name}:{k} embedding dim0 {container[k].shape[0]} "
            f"!= {DOCK_VOCAB} (docking checkpoints map to dict_dock, vocab 126)"
        )


# --------------------------------------------------------------------------- #
# DoD 4: collation -- misato apo(2)/holo(1)/ligand(3) layout
# --------------------------------------------------------------------------- #
def test_misato_infer_collation():
    from unigenx.data.tokenizer import UniGenXTokenizer

    cfg = _dock_config("misato")
    tok = UniGenXTokenizer.from_file(str(DICT_DOCK), cfg)
    ds = _make_dataset("misato", tok, cfg)
    _misato_fixture(ds)

    item = ds.get_infer_item_misato(0)
    toks = item["tokens"]
    mask = item["coordinates_mask"]
    n = _N_POCKET
    n_lig = 3

    # apo (given, mask 2) and holo (given, mask 1) are inside the prompt;
    # the ligand (mask 3) + trailing separators are "generate-the-rest".
    assert int(np.sum(mask == 2)) == n, "apo pocket -> mask 2 (given)"
    assert int(np.sum(mask == 1)) == n, "holo pocket -> mask 1 (given, routed to atoms)"
    assert int(np.sum(mask == 3)) == n_lig, "docked ligand -> mask 3 (generated)"

    # prompt ends at <LIG_COORDS_START>; the ligand mask-3 slots are beyond it.
    assert toks[0] == tok.bos_idx
    assert toks[-1] == tok.get_idx("<LIG_COORDS_START>")
    prompt_len = len(toks)
    # one trailing separator slot (release generate contract: eos-fill + stop)
    assert list(mask[prompt_len:]) == [3] * n_lig + [0]
    # given coordinate stream = apo ++ holo (one row per masked prompt slot)
    assert item["coordinates"].shape == (2 * n, 3)
    # gt coordinate stream = apo ++ holo ++ ligand (centered)
    assert item["gt_coords"].shape == (2 * n + n_lig, 3)
    assert tok.unk_idx not in list(toks)


def test_misato_simple_item():
    from unigenx.data.tokenizer import UniGenXTokenizer

    cfg = _dock_config("misato")
    tok = UniGenXTokenizer.from_file(str(DICT_DOCK), cfg)
    ds = _make_dataset("misato", tok, cfg)
    _misato_fixture(ds)

    item = ds.get_simple_item_misato(0)
    # the ground-truth keys the docking eval reads
    assert {"lig_coords", "holo_coords", "apo_coords", "smiles", "pdb_id"} <= set(item)
    assert item["apo_coords"].shape == (_N_POCKET, 3)
    assert item["holo_coords"].shape == (_N_POCKET, 3)
    assert item["lig_coords"].shape == (3, 3)
    # centered on the frame-0 ligand centroid
    assert np.allclose(item["lig_coords"].mean(axis=0), 0.0, atol=1e-4)


def test_misato_decode_split_roundtrip():
    """decode_batch(entity='misato') routes apo->lattice, holo+ligand->atoms;
    the inference split holo=atoms[:n_pocket], ligand=atoms[n_pocket:] recovers
    the three coordinate sets."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    cfg = _dock_config("misato")
    tok = UniGenXTokenizer.from_file(str(DICT_DOCK), cfg)
    ds = _make_dataset("misato", tok, cfg)
    _misato_fixture(ds)
    item = ds.get_infer_item_misato(0)

    mask = item["coordinates_mask"]
    tokens = item["tokens"]
    # pad tokens out to full mask width (as generate would), coords at masked slots
    full_tokens = np.concatenate(
        [tokens, np.full(len(mask) - len(tokens), tok.padding_idx)]
    )
    n_coords = int(np.sum(mask != 0))
    coords = np.arange(n_coords * 3, dtype=np.float64).reshape(n_coords, 3)

    out = tok.decode_batch(full_tokens[None], coords, mask[None], "misato")
    assert len(out) == 1
    sent, lattice, atom_coordinates = out[0]
    n = _N_POCKET
    # lattice = apo (mask 2); atoms = holo (mask 1) ++ ligand (mask 3)
    assert len(lattice) == n
    assert len(atom_coordinates) == n + 3
    holo = atom_coordinates[:n]
    ligand = atom_coordinates[n:]
    assert len(holo) == n and len(ligand) == 3


def test_dock_simple_item():
    from unigenx.data.tokenizer import UniGenXTokenizer

    cfg = _dock_config("dock")
    tok = UniGenXTokenizer.from_file(str(DICT_DOCK), cfg)
    ds = _make_dataset("dock", tok, cfg)
    _dock_fixture(ds)

    item = ds.get_simple_item_docking(0)
    assert {"smiles", "lig_coords", "prot_coords", "pocket", "center"} <= set(item)
    assert item["prot_coords"].shape == (_N_POCKET, 3)
    assert item["lig_coords"].shape == (3, 3)


def test_dock_infer_returns_none_with_real_dict():
    """With the committed dict_dock, the dock/unimol path's <PROT_COORDS_START>
    is <unk>, so get_infer_item_docking returns None (documented dict gap)."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    cfg = _dock_config("dock")
    tok = UniGenXTokenizer.from_file(str(DICT_DOCK), cfg)
    ds = _make_dataset("dock", tok, cfg)
    _dock_fixture(ds)
    assert ds.get_infer_item_docking(0) is None


def test_dock_infer_layout_with_pocket_tokens():
    """Add the two missing tokens to verify the ported dock layout: protein
    pocket coords -> mask 2 (given); ligand coords -> mask 1 (generate-the-rest,
    beyond the prompt which ends at <LIG_COORDS_START>)."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    cfg = _dock_config("dock")
    tok = UniGenXTokenizer.from_file(str(DICT_DOCK), cfg)
    tok.add_tok("<PROT_COORDS_START>")
    tok.add_tok("<PROT_COORDS_END>")
    ds = _make_dataset("dock", tok, cfg)
    _dock_fixture(ds)

    item = ds.get_infer_item_docking(0)
    assert item is not None
    toks = item["tokens"]
    mask = item["coordinates_mask"]
    n_lig = 3

    assert int(np.sum(mask == 2)) == _N_POCKET, "protein pocket -> mask 2 (given)"
    assert int(np.sum(mask == 1)) == n_lig, "ligand -> mask 1 (generated)"
    assert toks[0] == tok.bos_idx
    assert toks[-1] == tok.get_idx("<LIG_COORDS_START>")
    # ligand coordinate slots + trailing separator are beyond the prompt
    assert list(mask[len(toks) :]) == [1] * n_lig + [0]
    assert item["coordinates"].shape == (_N_POCKET, 3)  # given pocket coords
    assert item["gt_coords"].shape == (n_lig, 3)  # ground-truth ligand


def test_docking_collate_batches_given_coords():
    """collate() exposes input_coordinates (given pocket) and gt_coords (dock
    ground-truth ligand) for the docking generate path."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    cfg = _dock_config("misato")
    tok = UniGenXTokenizer.from_file(str(DICT_DOCK), cfg)
    ds = _make_dataset("misato", tok, cfg)
    _misato_fixture(ds)
    batch = ds.collate([ds.get_infer_item_misato(0)])
    assert "input_ids" in batch and "coordinates_mask" in batch
    assert "input_coordinates" in batch  # apo ++ holo given
    assert batch["input_coordinates"].shape[0] == 2 * _N_POCKET


# --------------------------------------------------------------------------- #
# DoD 6 (eval): calc_rmsd + best-of-N + evaluate_docking thresholds
# --------------------------------------------------------------------------- #
def test_calc_rmsd_identical_is_zero():
    from unigenx.data.docking_utils import calc_rmsd

    a = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert calc_rmsd(a, a) == pytest.approx(0.0)


def test_calc_rmsd_known_offset():
    """A rigid shift of (3,4,0) on every atom: per-atom squared distance is
    3^2+4^2 = 25, so RMSD = sqrt(mean(25)) = 5."""
    from unigenx.data.docking_utils import calc_rmsd

    a = np.zeros((4, 3))
    b = np.tile(np.array([3.0, 4.0, 0.0]), (4, 1))
    assert calc_rmsd(a, b) == pytest.approx(5.0)

    # a single-axis unit shift on N atoms -> RMSD 1.0
    c = np.tile(np.array([0.0, 0.0, 1.0]), (5, 1))
    assert calc_rmsd(np.zeros((5, 3)), c) == pytest.approx(1.0)


def test_calc_rmsd_shape_mismatch_raises():
    from unigenx.data.docking_utils import calc_rmsd

    with pytest.raises(ValueError):
        calc_rmsd(np.zeros((3, 3)), np.zeros((4, 3)))


def test_best_of_n_rmsd():
    """best-of-N = minimum RMSD over all gt x pred sample pairs for a target."""
    ev = _load_module(EVAL_DOCK_DIR / "evaluate_docking.py", "evaluate_docking")

    gt = [np.zeros((3, 3))]
    preds = [
        np.full((3, 3), 5.0),  # far
        np.zeros((3, 3)) + np.array([0.0, 0.0, 1.0]),  # RMSD 1.0
        np.full((3, 3), 9.0),  # far
    ]
    assert ev.best_of_n_rmsd(gt, preds) == pytest.approx(1.0)


def test_evaluate_docking_thresholds_ligand():
    """Toy 2-target x 2-sample ligand jsonl: target-0 has a perfect match
    (best RMSD 0), target-1's best is a (0,0,1) shift (RMSD 1). Both are < 2,
    so ratio<2 == 1.0; mean == 0.5; median == 0.5."""
    ev = _load_module(EVAL_DOCK_DIR / "evaluate_docking.py", "evaluate_docking")

    zeros = np.zeros((3, 3)).tolist()
    unit = (np.zeros((3, 3)) + np.array([0.0, 0.0, 1.0])).tolist()
    far = np.full((3, 3), 7.0).tolist()

    def rec(gt, pred):
        return {"ligand_gt": gt, "prediction": {"ligand_coords": pred}}

    records = [
        # target 0: a perfect gt/pred pair exists -> best 0
        rec(zeros, far),
        rec(zeros, zeros),
        # target 1: best pairing is the (0,0,1) unit shift -> best 1.0
        rec(zeros, far),
        rec(zeros, unit),
    ]
    res = ev.evaluate(records, samples_per_target=2)
    assert "ligand" in res
    lig = res["ligand"]
    assert lig["n_targets"] == 2
    assert lig["min_rmsd"][0] == pytest.approx(0.0)
    assert lig["min_rmsd"][1] == pytest.approx(1.0)
    assert lig["ratios"][2] == pytest.approx(1.0)
    assert lig["mean"] == pytest.approx(0.5)
    assert lig["median"] == pytest.approx(0.5)


def test_evaluate_docking_misato_variants():
    """misato records expose ligand + holo -> ligand / holo / combined variants
    are all scored."""
    ev = _load_module(EVAL_DOCK_DIR / "evaluate_docking.py", "evaluate_docking")

    lig = np.zeros((3, 3)).tolist()
    holo = np.zeros((4, 3)).tolist()

    def rec():
        return {
            "lig_coords": lig,
            "holo_coords": holo,
            "prediction": {"ligand_coords": lig, "holo_coords": holo},
        }

    records = [rec(), rec()]  # 1 target, 2 samples
    res = ev.evaluate(records, samples_per_target=2)
    assert set(res) == {"ligand", "holo", "combined"}
    for name in ("ligand", "holo", "combined"):
        assert res[name]["min_rmsd"][0] == pytest.approx(0.0)


def test_symmetric_rmsd_requires_spyrmsd():
    """--symmetry uses spyrmsd, which is not installed here (importorskip)."""
    pytest.importorskip("spyrmsd")
    ev = _load_module(EVAL_DOCK_DIR / "evaluate_docking.py", "evaluate_docking")
    sym = ev._make_symmetric_rmsd()
    a = np.zeros((3, 3))
    assert sym(a, a) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# DoD 5: inference dry-run -- tiny random model, docking generate (CUDA)
# --------------------------------------------------------------------------- #
def _build_tiny_dock_model(vocab_size, mask_token_id):
    from unigenx.model.config import UniGenXConfig
    from unigenx.model.wrapper import UniGenX

    config = UniGenXConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        diff_width=32,
        diff_depth=2,
        diff_steps="4",
        diff_mul=1,
        is_solver=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config.mask_token_id = mask_token_id
    model = UniGenX(config)
    model.eval()
    return model


@pytest.mark.skipif(
    not _HAS_CUDA,
    reason="diffloss.sample() allocates its noise on cuda; a GPU is required",
)
@pytest.mark.parametrize("setup", ["given_pocket_misato", "given_pocket_dock"])
def test_docking_generate_dry_run(setup):
    """Two docking setups (pocket-given misato: apo+holo given; dock: protein
    pocket given) both run the generate + decode_batch path on a tiny model."""
    from transformers import GenerationConfig

    from unigenx.data.tokenizer import UniGenXTokenizer
    from unigenx.model.unigenx import UniGenXOutput

    device = "cuda"
    if setup == "given_pocket_misato":
        cfg = _dock_config("misato")
        tok = UniGenXTokenizer.from_file(str(DICT_DOCK), cfg)
        ds = _make_dataset("misato", tok, cfg)
        _misato_fixture(ds)
        item = ds.get_infer_item_misato(0)
        entity = "misato"
    else:
        cfg = _dock_config("dock")
        tok = UniGenXTokenizer.from_file(str(DICT_DOCK), cfg)
        tok.add_tok("<PROT_COORDS_START>")
        tok.add_tok("<PROT_COORDS_END>")
        ds = _make_dataset("dock", tok, cfg)
        _dock_fixture(ds)
        item = ds.get_infer_item_docking(0)
        entity = "dock"

    batch = ds.collate([item])
    # the tiny model's vocab matches the (possibly token-extended) tokenizer
    model = _build_tiny_dock_model(len(tok), tok.mask_idx).to(device)

    gen_config = GenerationConfig(
        pad_token_id=0,
        eos_token_id=2,
        use_cache=True,
        max_length=batch["coordinates_mask"].shape[1],
        return_dict_in_generate=True,
    )
    ret = model.net.generate(
        input_ids=batch["input_ids"].to(device),
        coordinates_mask=batch["coordinates_mask"].to(device),
        input_coordinates=batch["input_coordinates"].to(device),
        generation_config=gen_config,
        max_length=batch["coordinates_mask"].shape[1],
    )
    assert isinstance(ret, UniGenXOutput)
    assert ret.coordinates is not None and ret.coordinates.shape[-1] == 3

    decoded = tok.decode_batch(
        ret.sequences.cpu().numpy(),
        ret.coordinates.cpu().numpy(),
        batch["coordinates_mask"].cpu().numpy(),
        entity,
    )
    assert len(decoded) == 1
    sent, lattice, atom_coordinates = decoded[0]  # dock/misato -> 3-tuple
    assert isinstance(atom_coordinates, list)
    assert len(lattice) == _N_POCKET  # given pocket coords echoed into lattice


# --------------------------------------------------------------------------- #
# hygiene: ported files carry no machine-absolute paths / internal branch names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path",
    [
        DATA_DIR / "docking_utils.py",
        EVAL_DOCK_DIR / "evaluate_docking.py",
        GEN_DOCK_SH,
    ],
)
def test_ported_files_have_no_absolute_paths(path):
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
        "U0ZNX2FsbA==",  # internal monorepo name
    ):
        needle = base64.b64decode(_enc).decode()
        assert (
            needle not in text
        ), f"{path.name} contains a forbidden internal identifier"
