# -*- coding: utf-8 -*-
"""Single-node training entry point for UniGenX.

Mirrors ``unigenx_infer.py``'s CLI / config bootstrap so that the checkpoint
produced here can be loaded straight back by the inference entry point: the
FULL :class:`UniGenXConfig` (carrying all model-architecture fields) is handed to
the ``Trainer`` as its ``args``, so the saved checkpoint's ``["args"]`` lets
``unigenx_infer.py`` rebuild the exact architecture via
``arg_utils.from_args(ckpt["args"], UniGenXConfig)`` before loading weights.

Example (single node, 1 GPU)::

    torchrun --nproc_per_node=1 unigenx_train.py \\
        --strategy Zero1 --target material \\
        --dict_path unigenx/data/dict_mat.txt \\
        --train_data_path <train.jsonl> --save_dir <out_dir>

``--strategy`` accepts Single / DDP / Zero0-3 (default Zero1). For Zero3, run
DeepSpeed's ``zero_to_fp32.py`` on the saved checkpoint before inference so the
consolidated fp32 weights live under the ``"module"`` key that
``load_pretrained_weights`` expects.
"""
from unigenx.data.dataset import MODE, UnifiedUniGenXDataset, UniGenXDataset
from unigenx.data.tokenizer import UniGenXTokenizer
from unigenx.logging import logger
from unigenx.model.config import UniGenXConfig
from unigenx.model.wrapper import UniGenX
from unigenx.pipeline.accelerator.trainer import Trainer
from unigenx.utils import arg_utils
from unigenx.utils.cli_utils import cli

# Unified targets whose training data may be a comma-joined
# "material_path,mol_path" pair, dispatched to UnifiedUniGenXDataset.
_UNIFIED_TARGETS = {"uni_mat", "uni_mol"}


def _build_dataset(tokenizer, data_path, config, shuffle, mode):
    """Build the training/validation dataset for the requested target.

    Single-path (default): a plain ``UniGenXDataset``.

    Unified mixed-path: when ``--target`` is ``uni_mat``/``uni_mol`` and
    ``data_path`` is a comma-joined ``"material_path,mol_path"`` pair, dispatch
    to ``UnifiedUniGenXDataset`` (dataset.py), which builds one uni_mat
    sub-dataset and one uni_mol sub-dataset (each a ``UniGenXDataset`` with
    its own target + tokenization) and interleaves them. Path order is
    ``[material, mol]`` (mirrors the source ``ARDiffDataset(..., data_path[0]``
    material, ``data_path[1]`` mol)).

    UnifiedUniGenXDataset forks per-sub-target configs via ``config.copy()``
    (``UniGenXConfig.copy`` returns a deep copy), so the Trainer's ``config`` stays
    intact and the checkpoint ``["args"]`` round-trip is unaffected.
    """
    if (
        getattr(config, "target", None) in _UNIFIED_TARGETS
        and data_path is not None
        and "," in data_path
    ):
        paths = [p.strip() for p in data_path.split(",")]
        assert len(paths) == 2, (
            "unified target expects a comma-joined 'material_path,mol_path' pair, "
            f"got {len(paths)} paths: {paths}"
        )
        return UnifiedUniGenXDataset(
            tokenizer, paths, config, shuffle=shuffle, mode=mode
        )
    return UniGenXDataset(tokenizer, data_path, config, shuffle=shuffle, mode=mode)


@cli(UniGenXConfig)
def main(args):
    # region initial config --------
    config = arg_utils.from_args(args, UniGenXConfig)
    logger.info(f"Initializing training with seed: {config.seed}")

    # region tokenizer + dict-vocab iron rule --------
    logger.info(f"Loading tokenizer from {config.dict_path}")
    tokenizer = UniGenXTokenizer.from_file(config.dict_path, config)
    # The released dict iron rule: vocab == non-empty dict lines + 7 special
    # tokens. The model embedding must match the tokenizer vocab exactly, else
    # generation/loading against DICT_MAP checkpoints silently produces garbage.
    config.vocab_size = len(tokenizer)
    config.mask_token_id = tokenizer.mask_idx
    # endregion --------

    # region model --------
    model = UniGenX(config)
    embed_vocab = model.net.get_input_embeddings().weight.shape[0]
    assert embed_vocab == len(tokenizer), (
        f"embedding vocab {embed_vocab} != tokenizer vocab {len(tokenizer)} "
        "(vocab must equal non-empty dict lines + 7 special tokens)"
    )
    logger.info(
        f"Built UniGenX with vocab_size={config.vocab_size}, "
        f"hidden_size={config.hidden_size}, num_hidden_layers={config.num_hidden_layers}"
    )
    # endregion --------

    # region data --------
    logger.info(f"Loading training data from {config.train_data_path}")
    train_data = _build_dataset(
        tokenizer,
        config.train_data_path,
        config,
        shuffle=True,
        mode=MODE.TRAIN,
    )
    valid_data = None
    if config.valid_data_path is not None:
        logger.info(f"Loading validation data from {config.valid_data_path}")
        valid_data = _build_dataset(
            tokenizer,
            config.valid_data_path,
            config,
            shuffle=False,
            mode=MODE.VAL,
        )
    # endregion --------

    # region trainer --------
    # Pass the FULL UniGenXConfig as the Trainer args so the saved checkpoint's
    # ["args"] carries every model-architecture field (see module docstring).
    # optimizer / lr_scheduler are left as None; the engine calls
    # model.config_optimizer() to build them.
    trainer = Trainer(
        args=config,
        model=model,
        train_data=train_data,
        valid_data=valid_data,
    )
    trainer.train()
    # endregion --------


if __name__ == "__main__":
    main()
