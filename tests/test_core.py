# -*- coding: utf-8 -*-
"""Stage-1 (spine) smoke tests for the UniGenX core.

Covers RELEASE_PLAN.md Section 3 Definition-of-Done items 1/2/3/5 for the
unified model + diffusion coordinate sampler:

  1. import               -- ``import unigenx`` succeeds.
  2. config round-trip    -- rebuild ``UniGenXConfig`` from a committed
                             saved-args fixture (vocab_size=193, hidden=1024, ...)
                             the same way ``unigenx_infer.py`` does, and assert
                             key fields survive.
  3. dict vocab assertion -- ``vocab == non-empty dict lines + 7`` for the two
                             spine dicts (dict_uni -> 193, dict_prot -> 28), using
                             only committed dicts (no checkpoint required). Plus a
                             skip-if-present check that the ``unified_pretrain``
                             checkpoint's ``embed_tokens.weight`` has dim0 == 193.
  5. inference dry-run     -- a tiny random model runs ``model.net.generate`` for
                             one step down BOTH sampler paths (DDPM default and
                             DPM-Solver ``is_solver=True, solver_order=2``) and
                             returns a ``UniGenXOutput(sequences, coordinates)``.

The dict-vocab and config round-trip tests must stay green with no checkpoints
present; the checkpoint and generate tests skip when their prerequisites (a
678MB checkpoint / a CUDA device) are unavailable.
"""
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "core"
DATA_DIR = REPO_ROOT / "unigenx" / "data"

# vocab = non-empty dict lines + 7 special tokens
# (<pad> <bos> <eos> <unk> prepended, <mask> <coord> <sg> appended).
DICT_VOCAB = {"dict_uni.txt": 193, "dict_prot.txt": 28}

# No machine-absolute paths are committed. These optional local resources live
# one level above the repo (a checkpoints/ sibling of the package
# repo in the internal release workspace). A public clone won't have them, so
# the dependent test skips. Override with env vars for other layouts.
_WORKSPACE = REPO_ROOT.parent
UNIFIED_PRETRAIN_CKPT = (
    Path(os.environ.get("UNIGENX_CHECKPOINTS", str(_WORKSPACE / "checkpoints")))
    / "unified_pretrain.pt"
)
_CKPT_ARGS_PKG = os.environ.get("UNIGENX_CKPT_ARGS_PKG")

try:
    import torch

    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _HAS_CUDA = False


# --------------------------------------------------------------------------- #
# DoD 1: import
# --------------------------------------------------------------------------- #
def test_import_unigenx():
    import unigenx  # noqa: F401


# --------------------------------------------------------------------------- #
# DoD 2: config round-trip from committed saved-args fixture
# --------------------------------------------------------------------------- #
def test_config_round_trip():
    from unigenx.model.config import UniGenXConfig
    from unigenx.utils import arg_utils

    with open(FIXTURE_DIR / "unified_pretrain_args.json") as f:
        saved_args = json.load(f)

    # Mirror unigenx_infer.py: config is rebuilt from the checkpoint's saved
    # args via arg_utils.from_args (which keeps only declared dataclass fields,
    # so stale source-only fields like "entity"/"max_length" are dropped safely).
    config = arg_utils.from_args(saved_args, UniGenXConfig)
    assert isinstance(config, UniGenXConfig)
    assert config.vocab_size == 193
    assert config.hidden_size == 1024
    assert config.num_hidden_layers == saved_args["num_hidden_layers"]
    assert config.num_attention_heads == saved_args["num_attention_heads"]
    assert config.diff_width == saved_args["diff_width"]
    assert config.diff_depth == saved_args["diff_depth"]
    assert config.diff_type == "diffloss"

    # Direct construction (**kwargs, incl. source-only fields) must agree on the
    # arch-critical fields.
    config2 = UniGenXConfig(**saved_args)
    assert config2.vocab_size == config.vocab_size
    assert config2.hidden_size == config.hidden_size
    assert config2.num_hidden_layers == config.num_hidden_layers


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
# DoD 3 (additional): checkpoint embedding vocab, skip-if-absent
# --------------------------------------------------------------------------- #
def test_unified_pretrain_embedding_vocab():
    if not UNIFIED_PRETRAIN_CKPT.exists():
        pytest.skip(f"checkpoint not present: {UNIFIED_PRETRAIN_CKPT}")
    import torch as _torch

    def _load():
        return _torch.load(
            str(UNIFIED_PRETRAIN_CKPT),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )

    try:
        try:
            state = _load()
        except ModuleNotFoundError:
            # Unpickling the saved args needs the internal training package on
            # the path; if that is unavailable we skip rather than fail.
            if _CKPT_ARGS_PKG and _CKPT_ARGS_PKG not in sys.path:
                sys.path.insert(0, _CKPT_ARGS_PKG)
            state = _load()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot load checkpoint ({type(e).__name__}: {e})")

    container = state
    if isinstance(state, dict):
        for key in ("model", "module", "state_dict"):
            if key in state and isinstance(state[key], dict):
                container = state[key]
                break

    # Search for the embedding weight by suffix (it may carry net./model./module.
    # prefixes) rather than assuming a fixed key name.
    matches = [
        k for k in container if isinstance(k, str) and k.endswith("embed_tokens.weight")
    ]
    assert matches, "no *embed_tokens.weight found in checkpoint state dict"
    for k in matches:
        assert container[k].shape[0] == 193, (
            f"{k}: embedding dim0 {container[k].shape[0]} != 193 "
            "(unified checkpoints must map to dict_uni, vocab 193)"
        )


# --------------------------------------------------------------------------- #
# DoD 5: inference dry-run (tiny random model), both sampler paths
# --------------------------------------------------------------------------- #
_TINY_VOCAB = 32
_MASK_ID = _TINY_VOCAB - 2  # a valid in-range id standing in for <mask>
_COORD_ID = _TINY_VOCAB - 1
_EOS_ID = 2
_PAD_ID = 0
_BOS_ID = 1


def _build_tiny_model(is_solver):
    from unigenx.model.config import UniGenXConfig
    from unigenx.model.wrapper import UniGenX

    config = UniGenXConfig(
        vocab_size=_TINY_VOCAB,
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
        is_solver=is_solver,
        solver_order=2,
        solver_type="dpmsolver",
        solver_steps=4,
        algorithm_type="dpmsolver++",
        pad_token_id=_PAD_ID,
        bos_token_id=_BOS_ID,
        eos_token_id=_EOS_ID,
    )
    config.mask_token_id = _MASK_ID
    model = UniGenX(config)  # wrapper (nn.Module) holding .net = the network
    model.eval()
    return model


def _run_one_step(is_solver):
    from transformers import GenerationConfig

    from unigenx.model.unigenx import UniGenXOutput

    device = "cuda"
    model = _build_tiny_model(is_solver).to(device)

    seq_len = 5
    input_ids = torch.tensor(
        [[_BOS_ID, 4, 5, 6, _COORD_ID]], dtype=torch.long, device=device
    )
    # For exactly one decode step: max_length == coordinates_mask width == seq_len + 1
    # (the final coordinate tensor length must match the mask width, see the tail
    # of UniGenX._greedy_search).
    coordinates_mask = torch.zeros((1, seq_len + 1), dtype=torch.long, device=device)
    coordinates_mask[0, -1] = 1  # the next slot is a coordinate slot

    gen_config = GenerationConfig(
        pad_token_id=_PAD_ID,
        eos_token_id=_EOS_ID,
        use_cache=True,
        max_length=seq_len + 1,
        return_dict_in_generate=True,
    )
    ret = model.net.generate(
        input_ids=input_ids,
        coordinates_mask=coordinates_mask,
        generation_config=gen_config,
        max_length=coordinates_mask.shape[1],
    )
    assert isinstance(ret, UniGenXOutput)
    assert ret.sequences is not None and ret.sequences.shape[0] == 1
    assert ret.coordinates is not None  # a coordinate was sampled at the masked slot
    assert ret.coordinates.shape[-1] == 3
    return ret


@pytest.mark.skipif(
    not _HAS_CUDA,
    reason="diffloss.sample() allocates its noise on cuda; a GPU is required for the dry-run",
)
def test_generate_ddpm():
    _run_one_step(is_solver=False)


@pytest.mark.skipif(
    not _HAS_CUDA,
    reason="diffloss.sample() allocates its noise on cuda; a GPU is required for the dry-run",
)
def test_generate_dpm_solver():
    _run_one_step(is_solver=True)
