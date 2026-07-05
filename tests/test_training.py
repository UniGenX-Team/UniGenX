# -*- coding: utf-8 -*-
"""T1 (spine) gate tests for the UniGenX training path.

Covers TRAINING_PLAN.md Section 6 T1 gates G1/G2/G3/G4:

  G1 test_engine_imports_clean       -- Trainer + DeepSpeedAccelerator import
                                        without pulling any of the heavier
                                        distributed-training backends that the
                                        single-node release does not ship
                                        (megatron/nnscaler) into sys.modules.
  G2 test_tiny_train_finite_loss     -- a tiny random UniGenX runs a couple of
                                        optimizer steps over synthetic collated
                                        batches with a finite loss, exercising
                                        before_batch / after_batch /
                                        config_optimizer (CPU / Single is fine).
  G3 test_ckpt_roundtrip_carries_args-- a DeepSpeed-shaped checkpoint
                                        ({'module': state_dict, 'args': asdict})
                                        round-trips and its 'args' still rebuild
                                        the architecture via arg_utils.from_args.
  G4 test_trained_ckpt_loads_into_inference
                                     -- a training-produced checkpoint loads
                                        back through unigenx_infer.py's load path
                                        (from_args -> UniGenX -> generate).
                                        CUDA-gated (diffloss.sample uses cuda).

These mirror tests/test_core.py conventions (REPO_ROOT on sys.path, repo-relative
paths, CUDA skipif). No checkpoints or real data are required.
"""
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

# Heavier distributed-training backends that the single-node release does not
# ship; the released training code must never pull them.
_FORBIDDEN_PKGS = {"megatron", "nnscaler"}

# Tiny special-token ids reused across the model-building helpers.
_TINY_VOCAB = 32
_MASK_ID = _TINY_VOCAB - 2
_COORD_ID = _TINY_VOCAB - 1
_EOS_ID = 2
_PAD_ID = 0
_BOS_ID = 1


def _tiny_config():
    from unigenx.model.config import UniGenXConfig

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
        pad_token_id=_PAD_ID,
        bos_token_id=_BOS_ID,
        eos_token_id=_EOS_ID,
    )
    config.mask_token_id = _MASK_ID
    config.total_num_steps = 10
    config.warmup_num_steps = 2
    config.max_lr = 1e-4
    return config


def _synthetic_batch(bs=2, seqlen=6, ncoord=2, device="cpu"):
    """A collated training batch matching dataset.collate_fn's output layout.

    ``input_coordinates`` is a flat (N, 3) tensor (N == number of coordinate
    slots across the batch), and ``coordinates_mask`` marks those slots; slot 0
    is never a coordinate (it is <bos>), mirroring the real data layout.
    """
    input_ids = torch.randint(4, _TINY_VOCAB, (bs, seqlen), device=device)
    input_ids[:, 0] = _BOS_ID
    coordinates_mask = torch.zeros(bs, seqlen, dtype=torch.long, device=device)
    coordinates_mask[:, -ncoord:] = 1
    n = int(coordinates_mask.sum().item())
    coords = torch.randn(n, 3, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(bs, seqlen, dtype=torch.long, device=device),
        "coordinates_mask": coordinates_mask,
        "input_coordinates": coords,
        "label_ids": input_ids.clone(),
        "label_coordinates": coords.clone(),
    }


# --------------------------------------------------------------------------- #
# G1: engine imports are clean (no megatron/nnscaler pulled)
# --------------------------------------------------------------------------- #
def test_engine_imports_clean():
    from unigenx.pipeline.accelerator.accelerator import (  # noqa: F401
        DeepSpeedAccelerator,
    )
    from unigenx.pipeline.accelerator.trainer import Trainer  # noqa: F401

    leaked = sorted(m for m in sys.modules if m.split(".")[0] in _FORBIDDEN_PKGS)
    assert not leaked, f"training engine pulled forbidden packages: {leaked}"


# --------------------------------------------------------------------------- #
# G2: tiny train -> finite loss, exercising the ported model hooks
# --------------------------------------------------------------------------- #
def test_tiny_train_finite_loss():
    from unigenx.model.wrapper import UniGenX

    config = _tiny_config()
    model = UniGenX(config)
    model.train()

    # config_optimizer is the ported hook (myAdamW + groupWarmupDecayLR).
    optimizer, lr_scheduler = model.config_optimizer()
    from torch.optim import Optimizer

    assert isinstance(optimizer, Optimizer)
    assert lr_scheduler is not None

    losses = []
    for _ in range(2):
        batch = _synthetic_batch()
        model.before_batch()  # engine calls this unconditionally per micro-batch
        model_output = model(batch)
        model_output = model.compute_loss(model_output, batch)
        loss = model_output.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        model.after_batch()
        losses.append(float(loss.item()))

    assert all(np.isfinite(losses)), f"non-finite training loss: {losses}"


# --------------------------------------------------------------------------- #
# G3: DeepSpeed-shaped checkpoint round-trips and carries arch args
# --------------------------------------------------------------------------- #
def test_ckpt_roundtrip_carries_args(tmp_path):
    from unigenx.model.config import UniGenXConfig
    from unigenx.model.wrapper import UniGenX
    from unigenx.utils import arg_utils

    config = _tiny_config()
    model = UniGenX(config)

    # Shape mirrors the DeepSpeed model-states file: weights under "module",
    # the full config under "args" (what unigenx_infer.py reads back).
    ckpt = {"module": model.state_dict(), "args": asdict(config)}
    ckpt_path = tmp_path / "mp_rank_00_model_states.pt"
    torch.save(ckpt, ckpt_path)

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "module" in loaded and "args" in loaded
    saved_args = loaded["args"]
    for key in ("num_hidden_layers", "hidden_size", "vocab_size"):
        assert key in saved_args, f"checkpoint args missing arch field: {key}"

    rebuilt = arg_utils.from_args(saved_args, UniGenXConfig)
    assert rebuilt.num_hidden_layers == config.num_hidden_layers
    assert rebuilt.hidden_size == config.hidden_size
    assert rebuilt.vocab_size == config.vocab_size


# --------------------------------------------------------------------------- #
# G4: a training-produced checkpoint loads back through the inference path
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not _HAS_CUDA,
    reason="diffloss.sample() allocates its noise on cuda; a GPU is required",
)
def test_trained_ckpt_loads_into_inference(tmp_path):
    from transformers import GenerationConfig

    from unigenx.model.config import UniGenXConfig
    from unigenx.model.unigenx import UniGenXOutput
    from unigenx.model.wrapper import UniGenX
    from unigenx.utils import arg_utils

    # 1. "Train": build a tiny model and save a DeepSpeed-shaped checkpoint.
    config = _tiny_config()
    trained = UniGenX(config)
    ckpt_path = tmp_path / "mp_rank_00_model_states.pt"
    torch.save({"module": trained.state_dict(), "args": asdict(config)}, ckpt_path)

    # 2. Drive unigenx_infer.py's load path: rebuild arch from ckpt["args"]
    #    BEFORE loading weights, then load tolerantly.
    saved_args = torch.load(ckpt_path, map_location="cpu", weights_only=False)["args"]
    saved_config = arg_utils.from_args(saved_args, UniGenXConfig)
    saved_config.mask_token_id = _MASK_ID
    model = UniGenX(saved_config)
    model.eval()
    model.load_pretrained_weights(str(ckpt_path))
    model.cuda()

    # 3. Generate one step and confirm a coordinate was produced.
    seq_len = 5
    input_ids = torch.tensor(
        [[_BOS_ID, 4, 5, 6, _COORD_ID]], dtype=torch.long, device="cuda"
    )
    coordinates_mask = torch.zeros((1, seq_len + 1), dtype=torch.long, device="cuda")
    coordinates_mask[0, -1] = 1
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
    assert ret.coordinates is not None
    assert ret.coordinates.shape[-1] == 3
