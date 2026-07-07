# -*- coding: utf-8 -*-
"""Committed example-input smoke tests.

These tests keep the public ``examples/data`` JSONL files aligned with the
release inference readers. They intentionally do not require checkpoints or run
generation; the end-to-end checkpoint-backed runs are done manually before
release updates.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXAMPLES = REPO_ROOT / "examples" / "data"
DATA_DIR = REPO_ROOT / "unigenx" / "data"


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _config(target, *, space_group=False):
    from unigenx.model.config import UniGenXConfig

    cfg = UniGenXConfig()
    cfg.target = target
    cfg.space_group = space_group
    cfg.reorder = False
    cfg.rotation_augmentation = False
    cfg.translation_augmentation = False
    cfg.scale_coords = None
    cfg.max_sites = None
    cfg.tokenizer = "num"
    return cfg


def _dataset(filename, target, dict_name, *, space_group=False):
    from unigenx.data.dataset import MODE, UniGenXDataset
    from unigenx.data.tokenizer import UniGenXTokenizer

    cfg = _config(target, space_group=space_group)
    tokenizer = UniGenXTokenizer.from_file(str(DATA_DIR / dict_name), cfg)
    dataset = UniGenXDataset(
        tokenizer,
        str(EXAMPLES / filename),
        args=cfg,
        shuffle=False,
        mode=MODE.INFER,
    )
    return dataset, tokenizer


def test_mp20_example_reader():
    records = _jsonl(EXAMPLES / "mp20_10.jsonl")
    assert len(records) == 10
    assert all(
        {"id", "formula", "lattice", "sites"} <= record.keys() for record in records
    )

    dataset, tokenizer = _dataset("mp20_10.jsonl", "material", "dict_mat.txt")
    assert len(dataset) == 10
    for i, record in enumerate(records):
        item = dataset.get_infer_item(i)
        assert item["id"] == record["id"]
        assert len(record["lattice"]) == 3
        assert int(item["coordinates_mask"].sum()) == 3 + len(record["sites"])
        assert tokenizer.unk_idx not in item["tokens"]


def test_drugs_example_reader():
    records = _jsonl(EXAMPLES / "drugs_10.jsonl")
    assert len(records) == 10
    assert len({record["smi"] for record in records}) == 10
    assert all({"id", "smi", "num", "pos"} <= record.keys() for record in records)

    dataset, tokenizer = _dataset("drugs_10.jsonl", "mol", "dict_drugs.txt")
    assert len(dataset) == 10
    for i, record in enumerate(records):
        item = dataset.get_infer_item(i)
        assert item["id"] == record["id"]
        assert (
            int(item["coordinates_mask"].sum()) == record["num"] == len(record["pos"])
        )
        assert tokenizer.unk_idx not in item["tokens"]


def test_protein_md_example_reader():
    records = _jsonl(EXAMPLES / "protein_md_2.jsonl")
    assert len(records) == 2
    assert all(
        {"id", "seq", "aa", "coords", "pos"} <= record.keys() for record in records
    )

    dataset, tokenizer = _dataset("protein_md_2.jsonl", "prot", "dict_prot.txt")
    assert len(dataset) == 2
    for i, record in enumerate(records):
        seq = record["seq"]
        item = dataset.get_infer_item(i)
        raw = dataset.get_infer_record_prot(i)
        assert raw["id"] == record["id"]
        assert seq == record["aa"]
        assert len(seq) <= 256
        assert int(item["coordinates_mask"].sum()) == len(seq) == len(record["coords"])
        assert record["coords"] == record["pos"]
        assert tokenizer.unk_idx not in item["tokens"]
