# -*- coding: utf-8 -*-
"""Stage-8 (EC-number conditioned enzyme design) smoke tests.

Covers RELEASE_PLAN.md Section 3 Definition-of-Done for the enzyme path
(target ``ecnum``, checkpoints ``e`` / ``e_wo``, embedding vocab 64):

  4. collation            -- get_{train,infer}_item_ecnum split the EC number on
                             "." into (up to) its first three levels and lay them
                             out as  <bos> <ec1> L1 <ec2> L2 <ec3> L3 <prot> ...
                             The infer prefix stops at <prot> (the model samples
                             the amino-acid sequence); the train item continues
                             with the residues and an <eos>. Sequence-only target,
                             so coordinates_mask is all zeros (no coordinate
                             slots).
  3. dict vocab assertion -- ``dict_ecnum.txt`` has 57 non-empty lines =>
                             tokenizer vocab 64 (vocab = lines + 7 special
                             tokens), which matches the ``e`` / ``e_wo``
                             checkpoints' ``embed_tokens.weight`` dim0 == 64
                             (ground truth, skip-if-absent).
  5. inference dry-run    -- a tiny random model runs the ecnum generate path
                             (all-zeros coordinate mask, sample-until-<coord>) and
                             returns a UniGenXOutput (CUDA-gated, as
                             diffloss.sample allocates its noise on cuda).
"""
import base64
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "unigenx" / "data"
DICT_ECNUM = DATA_DIR / "dict_ecnum.txt"
GEN_ENZYME_SH = REPO_ROOT / "scripts" / "gen_enzyme.sh"

# ground truth from the checkpoints (e / e_wo embedding dim0 == 64)
ECNUM_CKPT_VOCAB = 64
# dict_ecnum.txt has 57 non-empty lines -> tokenizer vocab 64 (57 + 7 specials)
ECNUM_DICT_LINES = 57

# 20 standard amino acids + unknown, in dict_ecnum.txt order
RESIDUES = list("ARNDCQEGHILKMFPSTWYVX")

_WORKSPACE = REPO_ROOT.parent
_CKPT_DIR = Path(os.environ.get("UNIGENX_CHECKPOINTS", str(_WORKSPACE / "checkpoints")))
EC_CHECKPOINTS = ["e.pt", "e_wo.pt"]

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _HAS_CUDA = False


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ecnum_config(target="ecnum"):
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


def _load_dataset(records, mode, tmp_path_factory):
    import json

    from unigenx.data.dataset import MODE, UniGenXDataset
    from unigenx.data.tokenizer import UniGenXTokenizer

    path = tmp_path_factory.mktemp("ecnum") / f"{mode}_toy.jsonl"
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    cfg = _ecnum_config()
    tok = UniGenXTokenizer.from_file(str(DICT_ECNUM), cfg)
    m = MODE.TRAIN if mode == "train" else MODE.INFER
    ds = UniGenXDataset(tok, str(path), args=cfg, shuffle=False, mode=m)
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


# toy fixtures: standard EC numbers + a short amino-acid sequence
_TOY_SEQ = "AGCCEK"
_TOY_TRAIN = {"id": 0, "EC_number": "1.1.1.1", "seq": _TOY_SEQ}
# a second infer record with a 4-level EC to exercise the [:3] truncation and
# multi-character level tokens ("7", "11")
_TOY_INFER = [
    {"id": 0, "EC_number": "1.1.1.1"},
    {"id": 1, "EC_number": "2.7.11.1"},
]


@pytest.fixture(scope="module")
def infer_ds(tmp_path_factory):
    return _load_dataset(_TOY_INFER, "infer", tmp_path_factory)


@pytest.fixture(scope="module")
def train_ds(tmp_path_factory):
    return _load_dataset([_TOY_TRAIN], "train", tmp_path_factory)


# --------------------------------------------------------------------------- #
# DoD 3 (ground truth): e / e_wo checkpoint embedding vocab == 64, skip-if-absent
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
        assert container[k].shape[0] == ECNUM_CKPT_VOCAB, (
            f"{ckpt.name}:{k} embedding dim0 {container[k].shape[0]} "
            f"!= {ECNUM_CKPT_VOCAB} (enzyme checkpoints require dict_ecnum vocab 64)"
        )


# --------------------------------------------------------------------------- #
# DoD 3: dict_ecnum vocab == 64, matching the e / e_wo checkpoint embeddings.
# --------------------------------------------------------------------------- #
def test_dict_ecnum_vocab_matches_checkpoint():
    from unigenx.data.tokenizer import UniGenXTokenizer

    tok = UniGenXTokenizer.from_file(str(DICT_ECNUM))
    assert len(tok) == ECNUM_CKPT_VOCAB, (
        f"dict_ecnum vocab {len(tok)} != {ECNUM_CKPT_VOCAB}; the e / e_wo "
        "checkpoints need a 57-line dict (vocab = lines + 7 special tokens)"
    )


def test_dict_ecnum_vocab_rule_holds():
    """The vocab-counting invariant (vocab == non-empty lines + 7) must hold:
    dict_ecnum.txt has 57 non-empty lines => tokenizer vocab 64, matching the
    e / e_wo checkpoints."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    assert DICT_ECNUM.exists(), f"missing committed dict: {DICT_ECNUM}"
    # splitlines() counts non-empty lines correctly even without a trailing
    # newline (unlike ``wc -l``), so this is the 57 that yields vocab 64.
    lines = [ln for ln in DICT_ECNUM.read_text().splitlines() if ln.strip()]
    tok = UniGenXTokenizer.from_file(str(DICT_ECNUM))
    assert len(tok) == len(lines) + 7
    assert len(lines) == ECNUM_DICT_LINES, (
        f"dict_ecnum has {len(lines)} non-empty lines; expected "
        f"{ECNUM_DICT_LINES} (=> vocab {ECNUM_CKPT_VOCAB})"
    )


def test_ecnum_prefix_tokens_resolve():
    """The EC-marker, EC-level and residue tokens used by the enzyme path must
    be real tokens; any collapsing to <unk> would silently corrupt the
    conditioning prefix (the exact failure mode the dict rule guards against)."""
    from unigenx.data.tokenizer import UniGenXTokenizer

    tok = UniGenXTokenizer.from_file(str(DICT_ECNUM))
    for marker in ("<ec1>", "<ec2>", "<ec3>", "<prot>"):
        assert tok.get_idx(marker) != tok.unk_idx, f"{marker} must be a real token"
    for level in ("1", "2", "7", "11"):  # EC level tokens used in the fixtures
        assert tok.get_idx(level) != tok.unk_idx, f"EC level {level} must resolve"
    for res in RESIDUES:
        assert tok.get_idx(res) != tok.unk_idx, f"residue {res} must be a real token"


# --------------------------------------------------------------------------- #
# DoD 4: collation -- EC split into 3-level token prefix + all-zero coord mask
# --------------------------------------------------------------------------- #
def test_ecnum_infer_prefix_layout(infer_ds):
    ds, tok = infer_ds
    item = ds.get_infer_item(0)  # EC 1.1.1.1
    assert {"tokens", "coordinates_mask"} <= set(item)
    # sequence-only inference prompt: no coordinates are provided
    assert "coordinates" not in item

    toks = list(item["tokens"])
    mask = item["coordinates_mask"]

    # prefix layout: <bos> <ec1> 1 <ec2> 1 <ec3> 1 <prot>
    expected = [
        tok.bos_idx,
        tok.get_idx("<ec1>"),
        tok.get_idx("1"),
        tok.get_idx("<ec2>"),
        tok.get_idx("1"),
        tok.get_idx("<ec3>"),
        tok.get_idx("1"),
        tok.get_idx("<prot>"),
    ]
    assert toks == expected
    assert len(toks) == 8
    assert tok.unk_idx not in toks  # every prefix token resolved

    # sequence-only: coordinate mask is all zeros and as wide as the tokens
    assert len(mask) == len(toks)
    assert int(np.asarray(mask).sum()) == 0


def test_ecnum_infer_truncates_to_three_levels(infer_ds):
    """EC "2.7.11.1" -> only the first three levels (2, 7, 11) are prefixed; the
    4th level (".1") is dropped by the ``.split(".")[:3]`` rule."""
    ds, tok = infer_ds
    item = ds.get_infer_item(1)  # EC 2.7.11.1
    toks = list(item["tokens"])

    expected = [
        tok.bos_idx,
        tok.get_idx("<ec1>"),
        tok.get_idx("2"),
        tok.get_idx("<ec2>"),
        tok.get_idx("7"),
        tok.get_idx("<ec3>"),
        tok.get_idx("11"),  # multi-character EC level token
        tok.get_idx("<prot>"),
    ]
    assert toks == expected
    assert len(toks) == 8  # exactly 3 levels kept, not 4


def test_ecnum_train_layout(train_ds):
    """Training item: <bos> <ec1> L1 <ec2> L2 <ec3> L3 <prot> {residues} <eos>,
    with an all-zero coordinate mask (no coordinate slots)."""
    ds, tok = train_ds
    item = ds.get_train_item(0)  # EC 1.1.1.1, seq AGCCEK
    assert {"tokens", "coordinates_mask"} <= set(item)
    # sequence-only target: no coordinate conditioning stream
    assert "coordinates" not in item

    toks = list(item["tokens"])
    mask = item["coordinates_mask"]
    n = len(_TOY_SEQ)

    # EC 3-level prefix
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
    # residues follow the <prot> separator
    assert toks[8 : 8 + n] == [tok.get_idx(r) for r in _TOY_SEQ]
    # then eos
    assert toks[-1] == tok.eos_idx
    assert len(toks) == 8 + n + 1  # prefix(8) + residues(n) + eos(1)
    assert tok.unk_idx not in toks

    # sequence-only: all-zero coordinate mask spanning the whole token stream
    assert len(mask) == len(toks)
    assert int(np.asarray(mask).sum()) == 0


def test_ecnum_train_seq_aa_fallback(tmp_path_factory):
    """The residue field may be keyed "seq" or "aa" -- both must work."""
    ds, tok = _load_dataset(
        [{"id": 0, "EC_number": "1.1.1.1", "aa": _TOY_SEQ}], "train", tmp_path_factory
    )
    item = ds.get_train_item(0)
    toks = list(item["tokens"])
    assert toks[8 : 8 + len(_TOY_SEQ)] == [tok.get_idx(r) for r in _TOY_SEQ]


def test_ecnum_collate_batch(infer_ds):
    """collate() batches the EC prefixes + all-zero coordinate mask; the mask is
    as wide as the (padded) prompt ids and carries no coordinate slot."""
    ds, _ = infer_ds
    items = [ds.get_infer_item(0), ds.get_infer_item(1)]
    batch = ds.collate(items)

    assert "input_ids" in batch and "coordinates_mask" in batch
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] == 8  # both prefixes are 8 tokens
    assert batch["coordinates_mask"].shape[1] >= batch["input_ids"].shape[1]
    assert int(batch["coordinates_mask"].sum()) == 0  # no coordinate slots
    # sequence-only target: no coordinate tensors are collated
    assert "input_coordinates" not in batch


# --------------------------------------------------------------------------- #
# DoD 5: inference dry-run -- tiny random model, ecnum generate path (CUDA)
# --------------------------------------------------------------------------- #
def _build_tiny_ecnum_model(vocab):
    from unigenx.model.config import UniGenXConfig
    from unigenx.model.wrapper import UniGenX

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
        is_solver=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config.mask_token_id = vocab - 3  # <mask> = first of the appended specials
    model = UniGenX(config)
    model.eval()
    return model


@pytest.mark.skipif(
    not _HAS_CUDA,
    reason="diffloss.sample() allocates its noise on cuda; a GPU is required",
)
def test_ecnum_generate_dry_run(infer_ds):
    from transformers import GenerationConfig

    from unigenx.model.unigenx import UniGenXOutput

    device = "cuda"
    ds, tok = infer_ds
    item = ds.get_infer_item(0)
    batch = ds.collate([item])

    model = _build_tiny_ecnum_model(len(tok)).to(device)
    input_ids = batch["input_ids"].to(device)

    # mirror the ecnum inference branch: an all-zeros coordinate mask spanning
    # the full generation length, sampling until <coord> (coord_idx as EOS).
    gen_len = 24
    coordinates_mask = torch.zeros(
        (input_ids.shape[0], gen_len), dtype=torch.long, device=device
    )
    sample_config = GenerationConfig(
        pad_token_id=tok.padding_idx,
        eos_token_id=tok.coord_idx,
        use_cache=True,
        max_length=gen_len,
        return_dict_in_generate=True,
    )
    ret = model.net.generate(
        input_ids=input_ids,
        coordinates_mask=coordinates_mask,
        generation_config=sample_config,
        max_length=coordinates_mask.shape[1],
        do_sample=True,
        top_p=0.95,
        temperature=1.0,
    )
    assert isinstance(ret, UniGenXOutput)
    assert ret.sequences.shape[0] == 1
    # prompt is preserved as the generation prefix
    assert ret.sequences.shape[1] >= input_ids.shape[1]


# --------------------------------------------------------------------------- #
# hygiene: ported files carry no machine-absolute paths / internal branch names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [GEN_ENZYME_SH, DICT_ECNUM])
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
    ):
        needle = base64.b64decode(_enc).decode()
        assert (
            needle not in text
        ), f"{path.name} contains a forbidden internal identifier"


def test_gen_enzyme_script_shape():
    """The gen script ships with blank CKPT/INPUT and the ecnum target + dict."""
    text = GEN_ENZYME_SH.read_text()
    assert "--target ecnum" in text
    assert "unigenx/data/dict_ecnum.txt" in text
    assert "\nCKPT=\n" in text and "\nINPUT=\n" in text  # left blank
    assert "CFG" not in text and "cfg_scale" not in text  # no CFG for enzyme
