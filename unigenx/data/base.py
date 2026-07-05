# -*- coding: utf-8 -*-
"""Minimal base classes referenced by the training engine's type annotations.

The released ``UniGenXDataset`` is a self-contained ``torch.utils.data.Dataset``
with its own ``.collate``; these lightweight base classes exist only so the
accelerator / trainer type hints (``Data``, ``Batch``, ``FoundationModelDataset``)
resolve without pulling in the Cython ``data_utils_fast`` machinery.
"""
from dataclasses import dataclass
from typing import List

from torch.utils.data import Dataset


@dataclass
class Data:
    pass


@dataclass
class Batch(Data):
    batch_size: int


class FoundationModelDataset(Dataset[Data]):
    def __init__(self) -> None:
        super().__init__()

    def collate(self, batch: List[Data]) -> Data:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def num_tokens(self, index: int) -> int:
        raise NotImplementedError

    def num_tokens_vec(self, indices):
        raise NotImplementedError

    def get_batch_shapes(self):
        return None
