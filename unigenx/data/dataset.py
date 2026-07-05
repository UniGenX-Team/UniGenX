# -*- coding: utf-8 -*-
import json
import os
import pickle
import random
import re
import zlib
from collections import OrderedDict
from enum import Enum
from functools import cmp_to_key
from typing import List, Union

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from unigenx.data.tokenizer import normalize_frac_coordinate
from unigenx.logging import logger


class MODE(Enum):
    TRAIN = 1
    VAL = 2
    INFER = 3


# allow pad_num to be int or float
def pad_1d_unsqueeze(
    x: torch.Tensor, padlen: int, start: int, pad_num: Union[int, float]
):
    # (N) -> (1, padlen)
    xlen = x.size(0)
    assert (
        start + xlen <= padlen
    ), f"padlen {padlen} is too small for xlen {xlen} and start point {start}"
    new_x = x.new_full([padlen], pad_num, dtype=x.dtype)
    new_x[start : start + xlen] = x
    x = new_x
    return x.unsqueeze(0)


def collate_fn(samples: List[dict], tokenizer, mode=MODE.TRAIN):
    """
    Overload BaseWrapperDataset.collater

    By default, the collater pads and batch all torch.Tensors (np.array will be converted) in the sample dicts
    """

    max_tokens = max(len(s["tokens"]) for s in samples)
    max_masks = max(
        len(s["coordinates_mask"]) + max_tokens - len(s["tokens"]) for s in samples
    )

    batch = dict()

    if "id" in samples[0]:
        batch["id"] = torch.tensor([s["id"] for s in samples], dtype=torch.long)

    batch["ntokens"] = torch.tensor(
        [len(s["tokens"]) for s in samples], dtype=torch.long
    )

    batch["input_ids"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["tokens"]).long(),
                max_tokens,
                0 if mode != MODE.INFER else max_tokens - len(s["tokens"]),
                tokenizer.padding_idx,
            )
            for s in samples
        ]
    )

    batch["attention_mask"] = batch["input_ids"].ne(tokenizer.padding_idx).long()

    if mode != MODE.INFER:
        batch["label_ids"] = batch["input_ids"].clone()

    if "coordinates" in samples[0]:
        batch["input_coordinates"] = torch.cat(
            [torch.from_numpy(s["coordinates"]) for s in samples]
        ).to(torch.float32)
        if mode != MODE.INFER:
            batch["label_coordinates"] = batch["input_coordinates"].clone()

    # cond_mol carries a variable number of property constraints per sample;
    # expose the per-sample count so two-phase inference can locate the
    # [<prop_i> propval_i]*num_cond prefix.
    if "num_cond" in samples[0]:
        batch["num_cond"] = torch.tensor(
            [s["num_cond"] for s in samples], dtype=torch.long
        )

    batch["coordinates_mask"] = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.from_numpy(s["coordinates_mask"]).long(),
                max_masks,
                0 if mode != MODE.INFER else max_tokens - len(s["tokens"]),
                tokenizer.padding_idx,
            )
            for s in samples
        ]
    )

    # docking / misato: `tags` marks (train) which coordinate rows get the
    # diffusion loss; `gt_coords` carries the ground-truth ligand coordinates
    # (dock inference) used to build the `ligand_gt` field of the output jsonl.
    if "tags" in samples[0]:
        batch["tags"] = torch.cat([torch.from_numpy(s["tags"]) for s in samples]).to(
            torch.long
        )

    if "gt_coords" in samples[0] and samples[0]["gt_coords"] is not None:
        batch["gt_coords"] = torch.cat(
            [torch.from_numpy(np.asarray(s["gt_coords"])) for s in samples]
        ).to(torch.float32)
    return batch


def normalize_frac_coordinates(coordinates: list, margin: float = 1e-4):
    return [normalize_frac_coordinate(x, margin) for x in coordinates]


def compare_by_coords(order=None):
    def innfer_f(a, b):
        frac_a = a["fractional_coordinates"]
        frac_b = b["fractional_coordinates"]
        if order == "<orderxyz>" or order is None:
            pass
        elif order == "<orderxzy>":
            frac_a = [frac_a[0], frac_a[2], frac_a[1]]
            frac_b = [frac_b[0], frac_b[2], frac_b[1]]
        elif order == "<orderyxz>":
            frac_a = [frac_a[1], frac_a[0], frac_a[2]]
            frac_b = [frac_b[1], frac_b[0], frac_b[2]]
        elif order == "<orderyzx>":
            frac_a = [frac_a[1], frac_a[2], frac_a[0]]
            frac_b = [frac_b[1], frac_b[2], frac_b[0]]
        elif order == "<orderzxy>":
            frac_a = [frac_a[2], frac_a[0], frac_a[1]]
            frac_b = [frac_b[2], frac_b[0], frac_b[1]]
        elif order == "<orderzyx>":
            frac_a = [frac_a[2], frac_a[1], frac_a[0]]
            frac_b = [frac_b[2], frac_b[1], frac_b[0]]
        else:
            raise ValueError(f"Unknown order {order}")
        if frac_a[0] > frac_b[0]:
            return 1
        elif frac_a[0] < frac_b[0]:
            return -1
        elif frac_a[1] > frac_b[1]:
            return 1
        elif frac_a[1] < frac_b[1]:
            return -1
        elif frac_a[2] > frac_b[2]:
            return 1
        elif frac_a[2] < frac_b[2]:
            return -1
        else:
            return 0

    return innfer_f


def sort_sites(sites, order=None):
    # sort the sites according to their distance to the start site
    sites_dict = OrderedDict()
    for site in sites:
        elem = site["element"]
        if elem not in sites_dict:
            sites_dict[elem] = []
        sites_dict[elem].append(site)
    for elem in sites_dict:
        sites_dict[elem] = sorted(
            sites_dict[elem],
            key=cmp_to_key(compare_by_coords(order)),
        )
    ret = []
    for elem in sites_dict:
        ret.extend(sites_dict[elem])
    return ret


class UnifiedUniGenXDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path: Union[str, list[str]] = None,
        args=None,
        shuffle: bool = True,
        mode=MODE.TRAIN,
        material_coeff=1,
        molecule_coeff=1,
    ):
        mat_args = args.copy()
        mat_args.target = "uni_mat"
        mol_args = args.copy()
        mol_args.target = "uni_mol"
        material_data_path, molecule_data_path = data_path[0], data_path[1]
        material_dataset = UniGenXDataset(
            tokenizer, material_data_path, mat_args, shuffle=shuffle, mode=mode
        )
        molecule_dataset = UniGenXDataset(
            tokenizer, molecule_data_path, mol_args, shuffle=shuffle, mode=mode
        )
        self.material_dataset = material_dataset
        self.molecule_dataset = molecule_dataset
        self.tokenizer = tokenizer
        self.mode = mode
        self.material_coeff = material_coeff
        self.molecule_coeff = molecule_coeff

    def __len__(self):
        return self.material_coeff * len(
            self.material_dataset
        ) + self.molecule_coeff * len(self.molecule_dataset)

    def __getitem__(self, idx):
        material_count = self.material_coeff * len(self.material_dataset)
        if idx < material_count:
            material_idx = idx % len(self.material_dataset)
            return self.material_dataset[material_idx]
        else:
            molecule_idx = (idx - material_count) % len(self.molecule_dataset)
            return self.molecule_dataset[molecule_idx]

    def collate(self, samples):
        return collate_fn(samples, self.tokenizer, self.mode)


class UniGenXDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path: Union[str, list[str]] = None,
        args=None,
        shuffle: bool = True,
        mode=MODE.TRAIN,
    ):
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode
        self.max_position_embeddings = args.max_position_embeddings

        self.data = []
        self.sizes = []
        self.env = None
        self.keys = None

        if data_path is not None:
            if data_path.count(",") > 0:
                data_path = data_path.split(",")
            if isinstance(data_path, str):
                data_path = [data_path]
            for path in data_path:
                self.load_data_from_file(path)

        assert (self.data == [] and self.keys is not None) or (
            self.data != [] and self.keys is None
        )

        if shuffle:
            if self.keys is not None:
                cb = list(zip(self.sizes, self.keys))
                random.shuffle(cb)
                self.sizes, self.keys = zip(*cb)
                self.sizes = list(self.sizes)
                self.keys = list(self.keys)
            else:
                random.shuffle(self.data)

        if self.args.scale_coords:
            logger.info(f"scale coords with scale {self.args.scale_coords}")

    def get_sequence_length(self, data_item):
        if self.args.target == "material" or self.args.target == "uni_mat":
            # <bos> [n * ] <coords> [3 lattice] [n coords] <eos>
            n = len(data_item["sites"])
            return 1 + n + 1 + 3 + n + 1
        elif self.args.target == "mol" or self.args.target == "uni_mol":
            # <bos> smiles <coords> [n coords] <eos>
            return len(data_item["smi"]) + 3 + data_item["num"]
        elif self.args.target == "prot":
            # <bos> seq <coord> [n Cα coords] <eos>
            seq = data_item.get("seq", data_item.get("aa"))
            n = len(seq)
            return 1 + n + 1 + n + 1
        elif self.args.target == "cond_mol":
            return 3
        elif self.args.target == "ecnum":
            # <bos> [<eci> Li]*<=3 <prot> seq <eos>
            seq = data_item.get("seq") or data_item.get("aa") or ""
            return 1 + 6 + 1 + len(seq) + 1
        else:
            return 0

    def load_dict(self, lines: List[dict]):
        skipped = 0
        for data_item in lines:
            size = self.get_sequence_length(data_item)
            if (
                self.args.target == "material"
                and self.args.max_sites is not None
                and len(data_item["sites"]) > self.args.max_sites
            ):
                skipped += 1
                continue
            if size > self.args.max_position_embeddings:
                skipped += 1
                continue

            if self.mode != MODE.INFER and self.args.target == "material":
                # normalize fractional coordinates
                for i in range(len(data_item["sites"])):
                    data_item["sites"][i][
                        "fractional_coordinates"
                    ] = normalize_frac_coordinates(
                        data_item["sites"][i]["fractional_coordinates"]
                    )
                sorted_sites = sort_sites(data_item["sites"])
                data_item["sites"] = sorted_sites

            self.data.append(data_item)  # type(data_item:) = dict
            self.sizes.append(size)
        logger.info(f"skipped {skipped} samples due to length constraints")

    def load_json(self, lines: List[str]):
        lines = [json.loads(line) for line in lines]
        self.load_dict(lines)

    def load_txt(self, lines: List[str]):
        skipped = 0
        for line in lines:
            data_item = line.strip()
            size = self.get_sequence_length(data_item)
            if size > self.args.max_position_embeddings:
                skipped += 1
                continue
            self.data.append(data_item)
            self.sizes.append(size)
        logger.info(f"skipped {skipped} samples due to length constraints")

    def infer_data_format(self, data_path, data_format):
        if data_path.endswith(".jsonl") or data_path.endswith(".json"):
            file_format = "json"
        elif data_path.endswith(".txt"):
            file_format = "txt"
        elif data_path.endswith(".lmdb"):
            file_format = "lmdb"
        elif data_path.endswith("pickle") or data_path.endswith("pkl"):
            file_format = "pickle"
        elif os.path.isdir(data_path):
            # docking (dock) / misato read a directory: LMDB sub-dbs for dock,
            # test_mols.pkl + MD_pockets.hdf5 for misato.
            file_format = "dir"
        else:
            raise ValueError(f"Unknown data format {data_path}")
        if data_format is not None:
            if data_format == file_format:
                return file_format
            else:
                return data_format
        return file_format

    def load_data_from_file(self, data_path, data_format=None):
        data_path = data_path.strip()
        data_format = self.infer_data_format(data_path, data_format)

        if data_format == "json":
            with open(data_path, "r") as f:
                lines = f.readlines()
            self.load_json(lines)
        elif data_format == "txt":
            with open(data_path, "r") as f:
                lines = f.readlines()
            self.load_txt(lines)
        elif data_format == "lmdb":
            self.env = lmdb.open(data_path, readonly=True, lock=False, readahead=False)
            self.txn = self.env.begin(write=False)
            try:
                metadata = pickle.loads(
                    zlib.decompress(self.txn.get("__metadata__".encode()))
                )
                self.sizes, self.keys = metadata["sizes"], metadata["keys"]
            except:
                cursor = self.txn.cursor()
                for key, _ in cursor:
                    self.keys.append(key)
        elif data_format == "pickle":
            with open(data_path, "rb") as f:
                self.data = pickle.load(f)
        elif data_format == "dir":
            self._load_docking_dir(data_path)
        else:
            raise ValueError(f"Unknown data format {data_format}")

    def __len__(self):
        if len(self.data) != 0:
            return len(self.data)
        elif self.keys is not None:
            return len(self.keys)
        else:
            raise ValueError("Dataset is empty")

    def get_infer_item_mat(self, index):
        item = dict()
        data_item = self.data[index]

        sorted_sites = data_item["sites"]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # get all sites
        sites_ids.extend(
            [self.tokenizer.get_idx(site["element"]) for site in sorted_sites]
        )
        coordinates_mask.extend([0 for _ in range(len(sorted_sites))])

        if self.args.space_group:
            # add special token
            sites_ids.append(self.tokenizer.coord_idx)
            coordinates_mask.append(0)

            # add order if needed
            if self.args.reorder:
                sites_ids.append(self.tokenizer.get_idx(self.tokenizer.order_tokens[0]))
                coordinates_mask.append(0)
        else:
            """
            # mask for space group
            coordinates_mask.append(0)
            # mask for special token
            coordinates_mask.append(0)
            """
            sites_ids.append(self.tokenizer.coord_idx)
            coordinates_mask.append(0)
            # mask for order
            if self.args.reorder:
                coordinates_mask.append(0)

        # add mask for lattice
        coordinates_mask.extend([1 for _ in range(3)])

        # add mask for coordinates
        coordinates_mask.extend([1 for _ in range(len(sorted_sites))])

        # add mask for eos
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_infer_item_mol(self, index):
        item = dict()
        if self.env:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.data[index])
                    data_item = pickle.loads(datapoint_pickled)
        else:
            data_item = self.data[index]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        smiles = data_item["smi"]
        atom_num = data_item["num"]

        # tokenize smiles
        smiles_id = []
        for i in range(len(smiles)):
            if smiles[i].islower():
                smiles_id[-1] = self.tokenizer.get_idx(smiles[i - 1] + smiles[i])
            else:
                smiles_id.append(self.tokenizer.get_idx(smiles[i]))

        sites_ids.extend(smiles_id)
        coordinates_mask.extend([0 for _ in range(len(smiles_id))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add coordinates
        coordinates_mask.extend([1 for _ in range(atom_num)])

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_infer_item_prot(self, index):
        if self.env is not None:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.keys[index].encode())
                    data_item = pickle.loads(zlib.decompress(datapoint_pickled))
        else:
            data_item = self.data[index]

        item = dict()
        seq = data_item.get("seq", data_item.get("aa"))

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]
        sites_ids.extend([self.tokenizer.get_idx(res) for res in seq])
        coordinates_mask.extend([0 for _ in range(len(seq))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)
        # add coordinates
        coordinates_mask.extend([1 for _ in range(len(seq))])

        # eos
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = index
        item["tokens"] = sites_ids
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_infer_record_prot(self, index):
        # Raw record dict for prot inference so the caller can attach a
        # prediction and serialize it. For LMDB input self.data is empty and
        # records live in self.env keyed by self.keys[index] (same load path as
        # get_infer_item_prot); for JSON input the record is self.data[index].
        if self.env is not None:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.keys[index].encode())
                    data_item = pickle.loads(zlib.decompress(datapoint_pickled))
        else:
            data_item = self.data[index]
        return data_item

    def get_infer_cond_mat(self, index):
        from math import exp, log

        item = dict()
        data_item = self.data[index]

        # Layout mirrors the (single-property) conditional-material training
        # item: ``<prop> propval <bos>`` -- the property marker token, then a
        # coordinate slot (mask=1) carrying the continuous property value via
        # the coordinate-encoder MLP, then <bos>. Must match get_train_cond_mat
        # so the checkpoint sees the same prefix layout at inference time.
        sites_ids = []
        coordinates_mask = []
        prop = data_item["prop"]
        sites_ids.extend(
            [
                self.tokenizer.get_idx(f"<{prop}>"),
                self.tokenizer.mask_idx,
            ]
        )
        coordinates_mask.extend([0, 1])

        # begin with bos
        sites_ids.append(self.tokenizer.bos_idx)
        coordinates_mask.append(0)

        # standardize the conditioning value exactly as during training
        prop_val = data_item["prop_val"]
        if prop == "bulk":
            if prop_val > 0:
                prop_val = log(prop_val + 1)
            else:
                prop_val = -log(-prop_val + 1)
        elif prop == "band":
            prop_val = log(prop_val + 1 / exp(1))
        else:  # mag
            if prop_val > 0:
                prop_val = -log(prop_val) / 10
            elif prop_val < 0:
                prop_val = log(-prop_val) / 10

        coordinates = np.array([prop_val, prop_val, prop_val]).reshape(1, 3)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = index
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_infer_cond_mol(self, index):
        item = dict()
        if self.env:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.data[index])
                    data_item = pickle.loads(datapoint_pickled)
        else:
            data_item = self.data[index]

        # Single- or multi-property conditioning: "prop"/"prop_val" are lists
        # (one entry per constraint). A single-property run is the length-1 case.
        prop_val_list = data_item["prop_val"]
        prop_list = data_item["prop"]

        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # one (<prop> propval) pair per property constraint
        for prop in prop_list:
            sites_ids.append(self.tokenizer.get_idx(f"<{prop[0]}>"))
            coordinates_mask.append(0)
            sites_ids.append(self.tokenizer.mask_idx)
            coordinates_mask.append(1)

        # begin of words
        sites_ids.append(self.tokenizer.get_idx("<w>"))
        coordinates_mask.append(0)

        # each property value is fed as a continuous coordinate row [v, v, v];
        # no lattice slot (molecular domain)
        coordinates = np.array(
            [[prop_val, prop_val, prop_val] for prop_val in prop_val_list]
        ).reshape(len(prop_val_list), 3)
        coordinates_mask = np.array(coordinates_mask)
        sites_ids = np.array(sites_ids)

        item["id"] = index
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        item["num_cond"] = len(prop_val_list)
        return item

    def get_infer_uni_mat(self, index):
        item = dict()
        data_item = self.data[index]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # add flag token
        flag_token = self.tokenizer.get_idx("<material>")
        sites_ids.append(flag_token)
        coordinates_mask.append(0)

        # get all sites
        sorted_sites = data_item["sites"]
        sites_ids.extend(
            [self.tokenizer.get_idx(f'<m>{site["element"]}') for site in sorted_sites]
        )
        coordinates_mask.extend([0 for _ in range(len(sorted_sites))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)
        # mask for order
        if self.args.reorder:
            coordinates_mask.append(0)

        # add mask for lattice
        coordinates_mask.extend([1 for _ in range(3)])

        # add mask for coordinates
        coordinates_mask.extend([1 for _ in range(len(sorted_sites))])

        # add mask for eos
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_infer_uni_mol(self, index):
        item = dict()
        if self.env:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.data[index])
                    data_item = pickle.loads(datapoint_pickled)
        else:
            data_item = self.data[index]
        # data_item = self.data[index]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # add flag token
        flag_token = self.tokenizer.get_idx("<molecule>")
        sites_ids.append(flag_token)
        coordinates_mask.append(0)

        smiles = data_item["smi"]
        atom_num = data_item["num"]
        # tokenize smiles
        smiles_id = []
        for i in range(len(smiles)):
            if smiles[i].islower():
                smiles_id[-1] = self.tokenizer.get_idx(
                    f"<s>{smiles[i - 1] + smiles[i]}"
                )
            else:
                smiles_id.append(self.tokenizer.get_idx(f"<s>{smiles[i]}"))
        # print(smiles)
        sites_ids.extend(smiles_id)
        coordinates_mask.extend([0 for _ in range(len(smiles_id))])
        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)
        # add coordinates
        coordinates_mask.extend([1 for _ in range(atom_num)])

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = data_item.get("id", index)
        item["tokens"] = sites_ids
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_train_item_mat(self, index):
        item = dict()
        data_item = self.data[index]

        # sort sites if reorder
        if self.args.reorder:
            order = random.choice(self.tokenizer.order_tokens)
            sites = sort_sites(data_item["sites"], order)
        else:
            sites = data_item["sites"]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # get all sites
        sites_ids.extend([self.tokenizer.get_idx(site["element"]) for site in sites])
        coordinates_mask.extend([0 for _ in range(len(sites))])

        # add space group
        # By zgb: no space group test
        """
        sites_ids.append(self.tokenizer.sg_idx)
        coordinates_mask.append(0)
        space_group_no = str(data_item["space_group"]["no"])
        space_group_tok = f"<sgn>{space_group_no}"
        sites_ids.append(self.tokenizer.get_idx(space_group_tok))
        coordinates_mask.append(0)
        """

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add order if needed
        if self.args.reorder:
            sites_ids.append(self.tokenizer.get_idx(order))
            coordinates_mask.append(0)

        # add lattice
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(3)])
        coordinates_mask.extend([1 for _ in range(3)])

        if self.args.rotation_augmentation:
            # add rotation augmentation
            lattice = self._random_rotation(lattice)

        if self.args.translation_augmentation:
            translation_vector = np.random.uniform(0, 1, size=3)
            sites = sites.copy()
            for site in sites:
                site["fractional_coordinates"] = list(
                    (np.asarray(site["fractional_coordinates"]) + translation_vector)
                    % 1
                )
            sites = sort_sites(sites)

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(sites))])
        coordinates_mask.extend([1 for _ in range(len(sites))])
        coordinates = np.array(
            [site["fractional_coordinates"] for site in sites]
        ).astype(np.float32)
        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['formula']}"

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates = np.concatenate([lattice, coordinates], axis=0)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_train_item_mol(self, index):
        item = dict()
        if self.env:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.data[index])
                    data_item = pickle.loads(datapoint_pickled)
        else:
            data_item = self.data[index]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        smiles = data_item["smi"]
        coords = data_item["pos"]

        # tokenize smiles
        smiles_id = []
        for i in range(len(smiles)):
            if smiles[i].islower():
                smiles_id[-1] = self.tokenizer.get_idx(smiles[i - 1] + smiles[i])
            else:
                smiles_id.append(self.tokenizer.get_idx(smiles[i]))
        # print(smiles)
        # sites_ids.extend([self.tokenizer.get_idx(char) for char in smiles])
        sites_ids.extend(smiles_id)
        # coordinates_mask.extend([0 for _ in range(len(smiles))])
        coordinates_mask.extend([0 for _ in range(len(smiles_id))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(coords))])
        coordinates_mask.extend([1 for _ in range(len(coords))])
        coordinates = np.array(coords).astype(np.float32)

        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        if self.args.rotation_augmentation:
            # add rotation augmentation
            coordinates = self._random_rotation(coordinates)

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['smi']}"

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_train_item_prot(self, index):
        item = dict()
        key = self.data[index]
        if isinstance(key, str):
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(key.encode())
                    data_item = pickle.loads(zlib.decompress(datapoint_pickled))
        else:
            data_item = key
        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # tokenize sequence
        seq = data_item["aa"]
        sites_ids.extend([self.tokenizer.get_idx(char) for char in seq])
        coordinates_mask.extend([0 for _ in range(len(seq))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add coordinates
        coordinates = data_item["pos"]

        coord_mean = np.mean(coordinates, axis=0)
        coordinates = coordinates - coord_mean  # 取中心

        if self.args.rotation_augmentation:
            # add rotation augmentation
            coordinates = self._random_rotation(coordinates)

        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(seq))])
        coordinates_mask.extend([1 for _ in range(len(seq))])

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        item["id"] = index
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_train_cond_mol(self, index):
        item = dict()
        if self.env:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.data[index])
                    data_item = pickle.loads(datapoint_pickled)
        else:
            data_item = self.data[index]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        smiles = data_item["smi"]
        coords = data_item["pos"]
        # single- or multi-property conditioning: "prop"/"prop_val" are lists
        # (one entry per constraint); a single-property run is the length-1 case.
        prop_val_list = data_item["prop_val"]
        prop_list = data_item["prop"]

        # one (<prop> propval) pair per property constraint
        for prop in prop_list:
            sites_ids.append(self.tokenizer.get_idx(f"<{prop[0]}>"))
            coordinates_mask.append(0)
            sites_ids.append(self.tokenizer.mask_idx)
            coordinates_mask.append(1)

        # begin of words
        sites_ids.append(self.tokenizer.get_idx("<w>"))
        coordinates_mask.append(0)

        # tokenize smiles
        smiles_id = []
        for i in range(len(smiles)):
            if smiles[i].islower():
                smiles_id[-1] = self.tokenizer.get_idx(smiles[i - 1] + smiles[i])
            else:
                smiles_id.append(self.tokenizer.get_idx(smiles[i]))
        # print(smiles)
        # sites_ids.extend([self.tokenizer.get_idx(char) for char in smiles])
        sites_ids.extend(smiles_id)
        # coordinates_mask.extend([0 for _ in range(len(smiles))])
        coordinates_mask.extend([0 for _ in range(len(smiles_id))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(coords))])
        coordinates_mask.extend([1 for _ in range(len(coords))])
        coordinates = np.array(coords).astype(np.float32)

        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        if self.args.rotation_augmentation:
            # add rotation augmentation
            coordinates = self._random_rotation(coordinates)

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['smi']}"

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        # prepend one property-value coordinate row [v, v, v] per constraint
        # (no lattice slot -- molecular domain)
        coordinates = np.insert(
            coordinates,
            0,
            np.array([[prop_val, prop_val, prop_val] for prop_val in prop_val_list]),
            axis=0,
        )

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_train_cond_mat(self, index):
        from math import exp, log

        item = dict()
        tags = []
        data_item = self.data[index]

        # sort sites if reorder
        if self.args.reorder:
            order = random.choice(self.tokenizer.order_tokens)
            sites = sort_sites(data_item["sites"], order)
        else:
            sites = data_item["sites"]

        sites_ids = []
        coordinates_mask = []

        # property
        for key in data_item["property"]:
            prop = re.search(r"dft_(.*?)_", key)
            if prop:
                prop_tok = prop.group(1)
                sites_ids.extend(
                    [
                        self.tokenizer.get_idx(f"<{prop_tok}>"),
                        self.tokenizer.mask_idx,
                    ]
                )
                coordinates_mask.extend([0, 1])
                tags.append(0)  # diffloss on property
                prop_val = data_item["property"][key]
                if prop_tok == "bulk":
                    if prop_val > 0:
                        prop_val = log(prop_val + 1)
                    else:
                        prop_val = -log(-prop_val + 1)
                elif prop_tok == "band":
                    prop_val = log(prop_val + 1 / exp(1))
                else:  # mag
                    if prop_val > 0:
                        prop_val = -log(prop_val) / 10
                    elif prop_val < 0:
                        prop_val = log(-prop_val) / 10
                break

        # begin with bos
        sites_ids.append(self.tokenizer.bos_idx)
        coordinates_mask.append(0)

        # get all sites
        sites_ids.extend([self.tokenizer.get_idx(site["element"]) for site in sites])
        coordinates_mask.extend([0 for _ in range(len(sites))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add order if needed
        if self.args.reorder:
            sites_ids.append(self.tokenizer.get_idx(order))
            coordinates_mask.append(0)

        # add lattice
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(3)])
        coordinates_mask.extend([1 for _ in range(3)])
        tags.extend([0 for _ in range(3)])

        if self.args.rotation_augmentation:
            # add rotation augmentation
            lattice = self._random_rotation(lattice)

        if self.args.translation_augmentation:
            translation_vector = np.random.uniform(0, 1, size=3)
            sites = sites.copy()
            for site in sites:
                site["fractional_coordinates"] = list(
                    (np.asarray(site["fractional_coordinates"]) + translation_vector)
                    % 1
                )
            sites = sort_sites(sites)

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(sites))])
        coordinates_mask.extend([1 for _ in range(len(sites))])
        tags.extend([0 for _ in range(len(sites))])
        coordinates = np.array(
            [site["fractional_coordinates"] for site in sites]
        ).astype(np.float32)
        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['formula']}"

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates = np.concatenate([lattice, coordinates], axis=0)
        tags = np.array(tags)

        coordinates = np.insert(
            coordinates, 0, np.array([prop_val, prop_val, prop_val]), axis=0
        )
        assert len(tags) == len(coordinates), f"{len(tags)}, {len(coordinates)}"
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        item["id"] = data_item["id"]
        # toks = []
        # for i in sites_ids:
        #     toks.append(self.tokenizer.get_tok(i))
        # print(toks)
        # print(coordinates)
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        item["tags"] = tags
        return item

    def get_train_uni_mat(self, index):
        item = dict()
        data_item = self.data[index]

        # sort sites if reorder
        if self.args.reorder:
            order = random.choice(self.tokenizer.order_tokens)
            sites = sort_sites(data_item["sites"], order)
        else:
            sites = data_item["sites"]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # add flag token
        flag_token = self.tokenizer.get_idx("<material>")
        sites_ids.append(flag_token)
        coordinates_mask.append(0)

        # get all sites
        sites_ids.extend(
            [self.tokenizer.get_idx(f'<m>{site["element"]}') for site in sites]
        )
        coordinates_mask.extend([0 for _ in range(len(sites))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add order if needed
        if self.args.reorder:
            sites_ids.append(self.tokenizer.get_idx(order))
            coordinates_mask.append(0)

        # add lattice
        lattice = np.array(data_item["lattice"]).astype(np.float32)
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(3)])
        coordinates_mask.extend([1 for _ in range(3)])

        if self.args.rotation_augmentation:
            # add rotation augmentation
            lattice = self._random_rotation(lattice)

        if self.args.translation_augmentation:
            translation_vector = np.random.uniform(0, 1, size=3)
            sites = sites.copy()
            for site in sites:
                site["fractional_coordinates"] = list(
                    (np.asarray(site["fractional_coordinates"]) + translation_vector)
                    % 1
                )
            sites = sort_sites(sites)

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(sites))])
        coordinates_mask.extend([1 for _ in range(len(sites))])
        coordinates = np.array(
            [site["fractional_coordinates"] for site in sites]
        ).astype(np.float32)
        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['formula']}"

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates = np.concatenate([lattice, coordinates], axis=0)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_train_uni_mol(self, index):
        item = dict()
        if self.env:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.data[index])
                    data_item = pickle.loads(datapoint_pickled)
        else:
            data_item = self.data[index]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]

        # add flag token
        flag_token = self.tokenizer.get_idx("<molecule>")
        sites_ids.append(flag_token)
        coordinates_mask.append(0)

        smiles = data_item["smi"]
        coords = data_item["pos"]

        # tokenize smiles
        smiles_id = []
        for i in range(len(smiles)):
            if smiles[i].islower():
                smiles_id[-1] = self.tokenizer.get_idx(
                    f"<s>{smiles[i - 1] + smiles[i]}"
                )
            else:
                smiles_id.append(self.tokenizer.get_idx(f"<s>{smiles[i]}"))
        # print(smiles)
        # sites_ids.extend([self.tokenizer.get_idx(char) for char in smiles])
        sites_ids.extend(smiles_id)
        # coordinates_mask.extend([0 for _ in range(len(smiles))])
        coordinates_mask.extend([0 for _ in range(len(smiles_id))])

        # add special token
        sites_ids.append(self.tokenizer.coord_idx)
        coordinates_mask.append(0)

        # add coordinates
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(coords))])
        coordinates_mask.extend([1 for _ in range(len(coords))])
        coordinates = np.array(coords).astype(np.float32)

        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        if self.args.rotation_augmentation:
            # add rotation augmentation
            coordinates = self._random_rotation(coordinates)

        assert len(coordinates) > 0, f"{data_item['id']}, {data_item['smi']}"

        # eos
        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        assert len(sites_ids) == len(
            coordinates_mask
        ), f"{len(sites_ids)}, {len(coordinates_mask)})"

        item["id"] = data_item["id"]
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    # ------------------------------------------------------------------ #
    # protein-ligand docking (targets: dock / misato)
    # ------------------------------------------------------------------ #
    # SMILES atom/bond token regex (shared by dock + misato ligand tokenization)
    _DOCK_SMILES_PATTERN = (
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|"
        r"\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )

    def _load_docking_dir(self, data_path):
        """Load a docking (dock) / misato directory into self.data.

        dock: LMDB sub-dbs (pockets/ligands/crossdock/unimol). misato:
        test_mols.pkl (RDKit molecules) + MD_pockets.hdf5. Only the sub-sources
        present on disk are used; inference reads the ``unimol`` pairs (dock) or
        the h5 pockets (misato).
        """
        self.lig_regex = re.compile(self._DOCK_SMILES_PATTERN)
        self.coords_max, self.coords_min = 20, -20
        mode_dict = {MODE.TRAIN: "train", MODE.VAL: "valid", MODE.INFER: "test"}
        split = mode_dict.get(self.mode, "test")

        if self.args.target == "dock":
            self.txns = dict()
            self.data = list()
            # Protein-only pockets (pretraining source)
            protein_lmdb = os.path.join(data_path, f"pockets/{split}.lmdb")
            if os.path.exists(protein_lmdb):
                env = lmdb.open(
                    protein_lmdb,
                    subdir=False,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=256,
                )
                txn = env.begin()
                self.txns["protein"] = txn
                for rep in range(5):
                    for key in txn.cursor().iternext(values=False):
                        self.data.append({"txn": "protein", "key": key, "idx": rep})
            # Ligand-only conformers (pretraining source)
            ligand_lmdb = os.path.join(data_path, f"ligands/{split}.lmdb")
            if os.path.exists(ligand_lmdb):
                env = lmdb.open(
                    ligand_lmdb,
                    subdir=False,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=256,
                )
                txn = env.begin()
                self.txns["ligand"] = txn
                for rep in range(10):
                    for key in txn.cursor().iternext(values=False):
                        self.data.append({"txn": "ligand", "key": key, "idx": rep})
            # CrossDock protein-ligand pairs
            pair_lmdb = os.path.join(
                data_path, "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
            )
            if os.path.exists(pair_lmdb):
                env = lmdb.open(
                    pair_lmdb,
                    subdir=False,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=256,
                )
                txn = env.begin()
                self.txns["crossdock"] = txn
                for rep in range(20):
                    for key in txn.cursor().iternext(values=False):
                        self.data.append({"txn": "crossdock", "key": key, "idx": rep})
            # Uni-Mol binding-pose pairs (the docking inference source)
            pair_lmdb = os.path.join(
                data_path,
                f"protein_ligand_binding_pose_prediction/{split}.lmdb",
            )
            if os.path.exists(pair_lmdb):
                env = lmdb.open(
                    pair_lmdb,
                    subdir=False,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                    max_readers=256,
                )
                txn = env.begin()
                self.txns["unimol"] = txn
                for rep in range(1):
                    for key in txn.cursor().iternext(values=False):
                        self.data.append({"txn": "unimol", "key": key, "idx": rep})

        elif self.args.target == "misato":
            import h5py

            self.data = list()
            pkl_path = os.path.join(data_path, f"{split}_mols.pkl")
            with open(pkl_path, "rb") as f:
                self.mols = pickle.load(f)  # dict: key -> RDKit molecule
            for k in self.mols.keys():
                for frame in range(200):  # 200 sampled frames per target
                    self.data.append({"pdb_id": k, "frame": frame % 100})
            self.h5 = h5py.File(os.path.join(data_path, "MD_pockets.hdf5"), "r")
        else:
            raise ValueError(
                f"dir format only supports dock/misato, not {self.args.target}"
            )

    def get_train_item_docking(self, index):
        from unigenx.data.docking_utils import (
            Mol2SmilesCoords,
            get_mol,
            normalize_coords,
        )

        item = dict()

        data_idx = self.data[index]
        txn, idx, key = data_idx["txn"], data_idx["idx"], data_idx["key"]
        datapoint_pickled = self.txns[txn].get(key)
        data = pickle.loads(datapoint_pickled)

        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]
        coordinates = []
        tags = list()  # Tag for which should be added diffloss.

        if txn == "protein":
            # Process protein, especially pockets
            sites_ids.append(self.tokenizer.get_idx("<PROT_START>"))
            coordinates_mask.append(0)

            coords_max, coords_min = 20, -20
            prot_coords = normalize_coords(data["coordinates"][0])
            if np.max(prot_coords) > coords_max or np.min(prot_coords) < coords_min:
                return None
            protein = data["atoms"]
            non_hydrogen_indices = [
                i for i, atom in enumerate(protein) if not atom.startswith("H")
            ]
            alpha_carbon_indices = [i for i, atom in enumerate(protein) if atom == "CA"]
            filtered_atoms = [
                protein[i][0] if i not in alpha_carbon_indices else "CA"
                for i in non_hydrogen_indices
            ]
            filtered_coords = prot_coords[non_hydrogen_indices, :]

            sites_ids.extend(
                [
                    self.tokenizer.get_idx(filtered_atom)
                    for filtered_atom in filtered_atoms
                ]
            )
            coordinates_mask.extend([0 for _ in range(len(filtered_atoms))])

            sites_ids.append(self.tokenizer.get_idx("<PROT_END>"))
            coordinates_mask.append(0)
            sites_ids.append(self.tokenizer.get_idx("<PROT_COORDS_START>"))
            coordinates_mask.append(0)

            sites_ids.extend(
                [self.tokenizer.mask_idx for _ in range(len(filtered_coords))]
            )
            coordinates_mask.extend([1 for _ in range(len(filtered_coords))])
            tags.extend([1 for _ in range(len(filtered_coords))])
            sites_ids.append(self.tokenizer.get_idx("<PROT_COORDS_END>"))
            coordinates_mask.append(0)
            coordinates = filtered_coords

        elif txn == "ligand":
            sites_ids.append(self.tokenizer.get_idx("<LIG_START>"))
            coordinates_mask.append(0)

            lig_coords = normalize_coords(data["coordinates"][idx])
            if (
                np.max(lig_coords) > self.coords_max
                or np.min(lig_coords) < self.coords_min
            ):
                return None
            ligand_SMILES = data["smi"]
            ligand_atoms = data["atoms"]
            tokens_smi = self.lig_regex.findall(ligand_SMILES)
            assert len(ligand_atoms) == lig_coords.shape[0]
            non_hydrogen_indices = [
                i for i, atom in enumerate(ligand_atoms) if atom != "H"
            ]
            filtered_coords = lig_coords[non_hydrogen_indices, :]

            sites_ids.extend(
                [self.tokenizer.get_idx(token_smi) for token_smi in tokens_smi]
            )
            coordinates_mask.extend([0 for _ in range(len(tokens_smi))])

            sites_ids.append(self.tokenizer.get_idx("<LIG_END>"))
            coordinates_mask.append(0)
            sites_ids.append(self.tokenizer.get_idx("<LIG_COORDS_START>"))
            coordinates_mask.append(0)

            sites_ids.extend(
                [self.tokenizer.mask_idx for _ in range(len(filtered_coords))]
            )
            coordinates_mask.extend([1 for _ in range(len(filtered_coords))])
            tags.extend([1 for _ in range(len(filtered_coords))])

            sites_ids.append(self.tokenizer.get_idx("<LIG_COORDS_END>"))
            coordinates_mask.append(0)
            coordinates = filtered_coords

        elif txn == "crossdock":
            prot_coords = (
                data["protein_pos"].numpy() - data["ligand_center_of_mass"].numpy()
            )
            lig_mol = get_mol(
                data["ligand_element"].tolist(),
                data["ligand_bond_index"][0].tolist(),
                data["ligand_bond_index"][1].tolist(),
                data["ligand_bond_type"].tolist(),
                data["ligand_pos"].tolist(),
            )
            if lig_mol is None:
                return None
            smiles, lig_atoms, lig_coords = Mol2SmilesCoords(lig_mol, canonical=False)
            lig_coords = lig_coords - data["ligand_center_of_mass"].numpy()
            protein = data["protein_atom_name"]
            ligand_SMILES = smiles
            ligand_atoms = lig_atoms

            non_hydrogen_indices = [
                i for i, atom in enumerate(protein) if not atom.startswith("H")
            ]
            alpha_carbon_indices = [i for i, atom in enumerate(protein) if atom == "CA"]
            filtered_atoms = [
                protein[i][0] if i not in alpha_carbon_indices else "CA"
                for i in non_hydrogen_indices
            ]
            filtered_coords = prot_coords[non_hydrogen_indices, :]

            sites_ids.append(self.tokenizer.get_idx("<PROT_START>"))
            coordinates_mask.append(0)
            sites_ids.extend(
                [
                    self.tokenizer.get_idx(filtered_atom)
                    for filtered_atom in filtered_atoms
                ]
            )
            coordinates_mask.extend([0 for _ in range(len(filtered_atoms))])
            sites_ids.append(self.tokenizer.get_idx("<PROT_END>"))
            coordinates_mask.append(0)
            sites_ids.append(self.tokenizer.get_idx("<PROT_COORDS_START>"))
            coordinates_mask.append(0)
            sites_ids.extend(
                [self.tokenizer.mask_idx for _ in range(len(filtered_coords))]
            )
            coordinates_mask.extend([2 for _ in range(len(filtered_coords))])
            tags.extend([0 for _ in range(len(filtered_coords))])
            sites_ids.append(self.tokenizer.get_idx("<PROT_COORDS_END>"))
            coordinates_mask.append(0)
            coordinates.extend(filtered_coords)

            tokens_smi = self.lig_regex.findall(ligand_SMILES)
            assert len(ligand_atoms) == lig_coords.shape[0]
            non_hydrogen_indices = [
                i for i, atom in enumerate(ligand_atoms) if atom != "H"
            ]
            filtered_coords = lig_coords[non_hydrogen_indices, :]

            sites_ids.append(self.tokenizer.get_idx("<LIG_START>"))
            coordinates_mask.append(0)
            sites_ids.extend(
                [self.tokenizer.get_idx(token_smi) for token_smi in tokens_smi]
            )
            coordinates_mask.extend([0 for _ in range(len(tokens_smi))])
            sites_ids.append(self.tokenizer.get_idx("<LIG_END>"))
            coordinates_mask.append(0)
            sites_ids.append(self.tokenizer.get_idx("<LIG_COORDS_START>"))
            coordinates_mask.append(0)
            sites_ids.extend(
                [self.tokenizer.mask_idx for _ in range(len(filtered_coords))]
            )
            coordinates_mask.extend([1 for _ in range(len(filtered_coords))])
            tags.extend([1 for _ in range(len(filtered_coords))])
            sites_ids.append(self.tokenizer.get_idx("<LIG_COORDS_END>"))
            coordinates_mask.append(0)
            coordinates.extend(filtered_coords)

        elif txn == "unimol":
            center = [data["config"]["cx"], data["config"]["cy"], data["config"]["cz"]]
            prot_coords = data["holo_pocket_coordinates"][0] - center
            lig = data["holo_mol"]

            p = np.random.uniform()
            smiles, lig_atoms, lig_coords = Mol2SmilesCoords(
                lig, canonical=(p < self.args.smi_rand_aug)
            )
            lig_coords = lig_coords - center
            protein = data["pocket_atoms"]
            ligand_SMILES = smiles
            ligand_atoms = lig_atoms

            non_hydrogen_indices = [
                i for i, atom in enumerate(protein) if not atom.startswith("H")
            ]
            alpha_carbon_indices = [i for i, atom in enumerate(protein) if atom == "CA"]
            filtered_atoms = [
                protein[i][0] if i not in alpha_carbon_indices else "CA"
                for i in non_hydrogen_indices
            ]
            filtered_coords = prot_coords[non_hydrogen_indices, :]

            sites_ids.append(self.tokenizer.get_idx("<PROT_START>"))
            coordinates_mask.append(0)
            sites_ids.extend(
                [
                    self.tokenizer.get_idx(filtered_atom)
                    for filtered_atom in filtered_atoms
                ]
            )
            coordinates_mask.extend([0 for _ in range(len(filtered_atoms))])
            sites_ids.append(self.tokenizer.get_idx("<PROT_END>"))
            coordinates_mask.append(0)
            sites_ids.append(self.tokenizer.get_idx("<PROT_COORDS_START>"))
            coordinates_mask.append(0)
            sites_ids.extend(
                [self.tokenizer.mask_idx for _ in range(len(filtered_coords))]
            )
            coordinates_mask.extend([2 for _ in range(len(filtered_coords))])
            tags.extend([0 for _ in range(len(filtered_coords))])
            sites_ids.append(self.tokenizer.get_idx("<PROT_COORDS_END>"))
            coordinates_mask.append(0)
            coordinates.extend(filtered_coords)

            tokens_smi = self.lig_regex.findall(ligand_SMILES)
            assert len(ligand_atoms) == lig_coords.shape[0]
            non_hydrogen_indices = [
                i for i, atom in enumerate(ligand_atoms) if atom != "H"
            ]
            filtered_coords = lig_coords[non_hydrogen_indices, :]

            sites_ids.append(self.tokenizer.get_idx("<LIG_START>"))
            coordinates_mask.append(0)
            sites_ids.extend(
                [self.tokenizer.get_idx(token_smi) for token_smi in tokens_smi]
            )
            coordinates_mask.extend([0 for _ in range(len(tokens_smi))])
            sites_ids.append(self.tokenizer.get_idx("<LIG_END>"))
            coordinates_mask.append(0)
            sites_ids.append(self.tokenizer.get_idx("<LIG_COORDS_START>"))
            coordinates_mask.append(0)
            sites_ids.extend(
                [self.tokenizer.mask_idx for _ in range(len(filtered_coords))]
            )
            coordinates_mask.extend([1 for _ in range(len(filtered_coords))])
            tags.extend([1 for _ in range(len(filtered_coords))])
            sites_ids.append(self.tokenizer.get_idx("<LIG_COORDS_END>"))
            coordinates_mask.append(0)
            coordinates.extend(filtered_coords)
        else:
            raise ValueError(f"Unknown txn {txn}")

        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        if (
            self.tokenizer.unk_idx in sites_ids
            or len(sites_ids) > self.args.max_position_embeddings
        ):
            return None

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)
        assert len(sites_ids) == len(coordinates_mask)

        coordinates = np.array(coordinates).astype(np.float32)
        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords
        if self.args.rotation_augmentation:
            coordinates = self._random_rotation(coordinates)
        assert len(coordinates) == len(tags)
        tags = np.array(tags)

        item["id"] = index
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        item["tags"] = tags
        return item

    def get_infer_item_docking(self, index):
        from unigenx.data.docking_utils import Mol2SmilesCoords

        item = dict()

        data_idx = self.data[index]
        txn, key = data_idx["txn"], data_idx["key"]
        datapoint_pickled = self.txns[txn].get(key)
        data = pickle.loads(datapoint_pickled)

        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]
        coordinates = []
        gt_coords = None

        if txn == "unimol":
            center = [data["config"]["cx"], data["config"]["cy"], data["config"]["cz"]]
            prot_coords = data["holo_pocket_coordinates"][0] - center
            lig = data["holo_mol"]

            smiles, lig_atoms, lig_coords = Mol2SmilesCoords(lig, canonical=True)
            lig_coords = lig_coords - center
            protein = data["pocket_atoms"]
            ligand_SMILES = smiles
            ligand_atoms = lig_atoms

            # ---- pocket (given coordinates: mask == 2) ----
            non_hydrogen_indices = [
                i for i, atom in enumerate(protein) if not atom.startswith("H")
            ]
            alpha_carbon_indices = [i for i, atom in enumerate(protein) if atom == "CA"]
            filtered_atoms = [
                protein[i][0] if i not in alpha_carbon_indices else "CA"
                for i in non_hydrogen_indices
            ]
            filtered_coords = prot_coords[non_hydrogen_indices, :]

            sites_ids.append(self.tokenizer.get_idx("<PROT_START>"))
            coordinates_mask.append(0)
            sites_ids.extend(
                [
                    self.tokenizer.get_idx(filtered_atom)
                    for filtered_atom in filtered_atoms
                ]
            )
            coordinates_mask.extend([0 for _ in range(len(filtered_atoms))])
            sites_ids.append(self.tokenizer.get_idx("<PROT_END>"))
            coordinates_mask.append(0)
            sites_ids.append(self.tokenizer.get_idx("<PROT_COORDS_START>"))
            coordinates_mask.append(0)
            sites_ids.extend(
                [self.tokenizer.mask_idx for _ in range(len(filtered_coords))]
            )
            coordinates_mask.extend([2 for _ in range(len(filtered_coords))])
            sites_ids.append(self.tokenizer.get_idx("<PROT_COORDS_END>"))
            coordinates_mask.append(0)
            coordinates.extend(filtered_coords)

            # ---- ligand (coordinates generated: mask == 1) ----
            tokens_smi = self.lig_regex.findall(ligand_SMILES)
            assert len(ligand_atoms) == lig_coords.shape[0]
            non_hydrogen_indices = [
                i for i, atom in enumerate(ligand_atoms) if atom != "H"
            ]
            filtered_coords = lig_coords[non_hydrogen_indices, :]

            sites_ids.append(self.tokenizer.get_idx("<LIG_START>"))
            coordinates_mask.append(0)
            sites_ids.extend(
                [self.tokenizer.get_idx(token_smi) for token_smi in tokens_smi]
            )
            coordinates_mask.extend([0 for _ in range(len(tokens_smi))])
            sites_ids.append(self.tokenizer.get_idx("<LIG_END>"))
            coordinates_mask.append(0)
            sites_ids.append(self.tokenizer.get_idx("<LIG_COORDS_START>"))
            coordinates_mask.append(0)
            # only the mask extends here (generate-the-rest): the ligand
            # coordinate slots + the trailing separator are produced by generate.
            coordinates_mask.extend([1 for _ in range(len(filtered_coords))])
            coordinates_mask.append(0)
            gt_coords = filtered_coords
        else:
            raise ValueError(f"Unknown txn {txn}")

        if self.tokenizer.unk_idx in sites_ids:
            return None

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)
        coordinates = np.array(coordinates).astype(np.float32)
        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords

        item["id"] = index
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        item["gt_coords"] = gt_coords
        return item

    def get_simple_item_docking(self, index):
        """Return the raw (untokenized) ligand / pocket for building the output
        jsonl (smiles, ligand + pocket coordinates, center)."""
        from unigenx.data.docking_utils import Mol2SmilesCoords

        item = dict()

        data_idx = self.data[index]
        txn, key = data_idx["txn"], data_idx["key"]
        datapoint_pickled = self.txns[txn].get(key)
        data = pickle.loads(datapoint_pickled)

        center = [data["config"]["cx"], data["config"]["cy"], data["config"]["cz"]]
        prot_coords = data["holo_pocket_coordinates"][0] - center
        lig = data["holo_mol"]
        smiles, lig_atoms, lig_coords = Mol2SmilesCoords(lig, canonical=True)
        lig_coords = lig_coords - center
        protein = data["pocket_atoms"]

        non_hydrogen_indices = [
            i for i, atom in enumerate(protein) if not atom.startswith("H")
        ]
        alpha_carbon_indices = [i for i, atom in enumerate(protein) if atom == "CA"]
        filtered_atoms = [
            protein[i][0] if i not in alpha_carbon_indices else "CA"
            for i in non_hydrogen_indices
        ]
        filtered_prot_coords = prot_coords[non_hydrogen_indices, :]

        tokens_smi = self.lig_regex.findall(smiles)
        non_hydrogen_indices = [i for i, atom in enumerate(lig_atoms) if atom != "H"]
        filtered_lig_coords = lig_coords[non_hydrogen_indices, :]

        item["smiles"] = smiles
        item["lig_atoms"] = tokens_smi
        item["lig_coords"] = filtered_lig_coords
        item["center"] = center
        item["pocket"] = filtered_atoms
        item["prot_coords"] = filtered_prot_coords
        return item

    def get_train_item_misato(self, index):
        import periodictable

        item = dict()

        data_idx = self.data[index]
        key, frame = data_idx["pdb_id"], data_idx["frame"]
        mol_dict = self.mols[key]
        lig_begin_atoms_index = self.h5[key]["molecules_begin_atom_index"][-1]
        pocket_atoms = self.h5[key]["atoms_number"][:lig_begin_atoms_index]
        pocket_atoms = [periodictable.elements[Z].symbol for Z in pocket_atoms]

        holo_coords = self.h5[key]["trajectory_coordinates"][
            frame, :lig_begin_atoms_index
        ]

        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]
        coordinates = []
        tags = list()

        if isinstance(mol_dict, dict):
            lig_smiles = mol_dict["smi"]
            mol = mol_dict["mol"]
        else:
            # test_mols.pkl may store the RDKit Mol object directly
            from rdkit import Chem

            mol = mol_dict
            lig_smiles = Chem.MolToSmiles(mol)
        lig_coords = mol.GetConformer(frame).GetPositions()
        lig_coords0 = mol.GetConformer(0).GetPositions()

        # Tokenize Pocket
        sites_ids.append(self.tokenizer.get_idx("<PROT_START>"))
        coordinates_mask.append(0)
        for atom in pocket_atoms:
            sites_ids.append(self.tokenizer.get_idx(atom))
            coordinates_mask.append(0)
        sites_ids.append(self.tokenizer.get_idx("<PROT_END>"))
        coordinates_mask.append(0)

        # Apo pocket coordinates (given: mask == 2)
        apo_coords = list(self.h5[key]["apo_pocket_coordinates"])
        coordinates.extend(apo_coords)
        sites_ids.append(self.tokenizer.get_idx("<PROT_APO_COORDS_START>"))
        coordinates_mask.append(0)
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(pocket_atoms))])
        coordinates_mask.extend([2 for _ in range(len(pocket_atoms))])
        tags.extend([0 for _ in range(len(pocket_atoms))])
        sites_ids.append(self.tokenizer.get_idx("<PROT_APO_COORDS_END>"))
        coordinates_mask.append(0)

        # Ligand SMILES
        tokens_smi = self.lig_regex.findall(lig_smiles)
        sites_ids.append(self.tokenizer.get_idx("<LIG_START>"))
        coordinates_mask.append(0)
        sites_ids.extend(
            [self.tokenizer.get_idx(token_smi) for token_smi in tokens_smi]
        )
        coordinates_mask.extend([0 for _ in range(len(tokens_smi))])
        sites_ids.append(self.tokenizer.get_idx("<LIG_END>"))
        coordinates_mask.append(0)

        # Holo pocket coordinates (given: mask == 2)
        coordinates.extend(holo_coords)
        sites_ids.append(self.tokenizer.get_idx("<PROT_HOLO_COORDS_START>"))
        coordinates_mask.append(0)
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(pocket_atoms))])
        coordinates_mask.extend([2 for _ in range(len(pocket_atoms))])
        tags.extend([0 for _ in range(len(pocket_atoms))])
        sites_ids.append(self.tokenizer.get_idx("<PROT_HOLO_COORDS_END>"))
        coordinates_mask.append(0)

        # Docked ligand coordinates (predicted: mask == 1)
        coordinates.extend(lig_coords)
        sites_ids.append(self.tokenizer.get_idx("<LIG_COORDS_START>"))
        coordinates_mask.append(0)
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(lig_coords))])
        coordinates_mask.extend([1 for _ in range(len(lig_coords))])
        tags.extend([1 for _ in range(len(lig_coords))])
        sites_ids.append(self.tokenizer.get_idx("<LIG_COORDS_END>"))
        coordinates_mask.append(0)

        sites_ids.append(self.tokenizer.eos_idx)
        coordinates_mask.append(0)

        center = lig_coords0.mean(axis=0)
        coordinates = np.array(coordinates).astype(np.float32) - center

        assert len(coordinates) == len(tags)
        assert len(sites_ids) == len(coordinates_mask)
        if len(sites_ids) > self.args.max_position_embeddings:
            return None

        if self.args.scale_coords:
            coordinates = coordinates * self.args.scale_coords
        if self.args.rotation_augmentation:
            coordinates = self._random_rotation(coordinates)

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)
        tags = np.array(tags)

        item["id"] = index
        item["tokens"] = sites_ids
        item["coordinates"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        item["tags"] = tags
        return item

    def get_infer_item_misato(self, index):
        import periodictable

        item = dict()

        data_idx = self.data[index]
        key, frame = data_idx["pdb_id"], data_idx["frame"]
        mol_dict = self.mols[key]
        lig_begin_atoms_index = self.h5[key]["molecules_begin_atom_index"][-1]
        pocket_atoms = self.h5[key]["atoms_number"][:lig_begin_atoms_index]
        pocket_atoms = [periodictable.elements[Z].symbol for Z in pocket_atoms]

        holo_coords = self.h5[key]["trajectory_coordinates"][
            frame, :lig_begin_atoms_index
        ]

        sites_ids = [self.tokenizer.bos_idx]
        coordinates_mask = [0]
        coordinates = []

        if isinstance(mol_dict, dict):
            lig_smiles = mol_dict["smi"]
            mol = mol_dict["mol"]
        else:
            # test_mols.pkl may store the RDKit Mol object directly
            from rdkit import Chem

            mol = mol_dict
            lig_smiles = Chem.MolToSmiles(mol)
        lig_coords = mol.GetConformer(frame).GetPositions()

        # Tokenize Pocket
        sites_ids.append(self.tokenizer.get_idx("<PROT_START>"))
        coordinates_mask.append(0)
        for atom in pocket_atoms:
            sites_ids.append(self.tokenizer.get_idx(atom))
            coordinates_mask.append(0)
        sites_ids.append(self.tokenizer.get_idx("<PROT_END>"))
        coordinates_mask.append(0)

        # Apo pocket coordinates (given: mask == 2)
        apo_coords = list(self.h5[key]["apo_pocket_coordinates"])
        assert len(apo_coords) == len(pocket_atoms)
        coordinates.extend(apo_coords)
        sites_ids.append(self.tokenizer.get_idx("<PROT_APO_COORDS_START>"))
        coordinates_mask.append(0)
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(pocket_atoms))])
        coordinates_mask.extend([2 for _ in range(len(pocket_atoms))])
        sites_ids.append(self.tokenizer.get_idx("<PROT_APO_COORDS_END>"))
        coordinates_mask.append(0)

        # Ligand SMILES
        tokens_smi = self.lig_regex.findall(lig_smiles)
        sites_ids.append(self.tokenizer.get_idx("<LIG_START>"))
        coordinates_mask.append(0)
        sites_ids.extend(
            [self.tokenizer.get_idx(token_smi) for token_smi in tokens_smi]
        )
        coordinates_mask.extend([0 for _ in range(len(tokens_smi))])
        sites_ids.append(self.tokenizer.get_idx("<LIG_END>"))
        coordinates_mask.append(0)

        # Holo pocket coordinates. mask == 1 keeps holo inside the given prompt
        # coordinate stream (item["coordinates"] = apo ++ holo), so at inference
        # the holo pocket is teacher-forced like apo; the mask==1 (vs apo's
        # mask==2) only routes holo to atom_coordinates at decode time (so the
        # output splits into apo_coords / holo_coords / ligand_coords).
        coordinates.extend(holo_coords)
        sites_ids.append(self.tokenizer.get_idx("<PROT_HOLO_COORDS_START>"))
        coordinates_mask.append(0)
        sites_ids.extend([self.tokenizer.mask_idx for _ in range(len(pocket_atoms))])
        coordinates_mask.extend([1 for _ in range(len(pocket_atoms))])
        sites_ids.append(self.tokenizer.get_idx("<PROT_HOLO_COORDS_END>"))
        coordinates_mask.append(0)

        # Docked ligand coordinates (generated: mask == 3, generate-the-rest).
        # Exactly one trailing separator slot (mask 0) follows the ligand
        # coordinate slots: the release generate contract fills the first
        # non-coordinate slot beyond the prompt with <eos> and stops, so the
        # mask width must be prompt + n_lig + 1 for
        # input_coordinates[coordinates_mask] to line up. (The v4 source emitted
        # two closing tokens <LIG_COORDS_END>,<eos> via a bespoke token
        # injection in _greedy_search; the coordinates are identical either way
        # since these trailing mask==0 slots are discarded.)
        coordinates.extend(lig_coords)
        sites_ids.append(self.tokenizer.get_idx("<LIG_COORDS_START>"))
        coordinates_mask.append(0)
        coordinates_mask.extend([3 for _ in range(len(lig_coords))])
        coordinates_mask.append(0)

        lig_coords0 = mol.GetConformer(0).GetPositions()
        center = lig_coords0.mean(axis=0)
        coordinates = np.array(coordinates).astype(np.float32) - center
        apo_coords = np.array(apo_coords).astype(np.float32) - center
        holo_coords = np.array(holo_coords).astype(np.float32) - center

        if len(coordinates_mask) > self.args.max_position_embeddings:
            return None
        if self.args.scale_coords:
            apo_coords = apo_coords * self.args.scale_coords
            holo_coords = holo_coords * self.args.scale_coords

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.array(coordinates_mask)

        item["id"] = index
        item["tokens"] = sites_ids
        # given coordinate stream = apo (mask 2) ++ holo (mask 1) in sequence
        item["coordinates"] = np.concatenate([apo_coords, holo_coords], axis=0)
        item["gt_coords"] = coordinates
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_simple_item_misato(self, index):
        import periodictable

        item = dict()

        data_idx = self.data[index]
        key, frame = data_idx["pdb_id"], data_idx["frame"]
        mol_dict = self.mols[key]
        lig_begin_atoms_index = self.h5[key]["molecules_begin_atom_index"][-1]
        pocket_atoms = self.h5[key]["atoms_number"][:lig_begin_atoms_index]
        pocket_atoms = [periodictable.elements[Z].symbol for Z in pocket_atoms]

        holo_coords = self.h5[key]["trajectory_coordinates"][
            frame, :lig_begin_atoms_index
        ]

        if isinstance(mol_dict, dict):
            mol = mol_dict["mol"]
            lig_smiles = mol_dict["smi"]
        else:
            # test_mols.pkl may store the RDKit Mol object directly
            from rdkit import Chem

            mol = mol_dict
            lig_smiles = Chem.MolToSmiles(mol)
        lig_coords = mol.GetConformer(frame).GetPositions()
        lig_coords0 = mol.GetConformer(0).GetPositions()

        apo_coords = np.array((self.h5[key]["apo_pocket_coordinates"])).astype(
            np.float32
        )
        center = lig_coords0.mean(axis=0)

        item["smiles"] = lig_smiles
        item["pdb_id"] = key
        item["lig_coords"] = lig_coords - center
        item["holo_coords"] = holo_coords - center
        item["apo_coords"] = apo_coords - center
        item["center"] = center
        item["pocket"] = pocket_atoms
        return item

    def get_train_item_ecnum(self, index):
        # EC-number conditioned enzyme (protein-sequence) design. The EC number
        # is split on "." into (up to) its first three levels; each level token
        # is preceded by an <ec1>/<ec2>/<ec3> marker, then the amino-acid
        # sequence follows the <prot> separator:
        #   <bos> <ec1> L1 <ec2> L2 <ec3> L3 <prot> {residues} <eos>
        # Sequence-only target (no coordinate slots), so coordinates_mask is all
        # zeros -- the shared collate_fn can then batch it like any other target.
        data_item = self.data[index]

        item = dict()
        seq = data_item.get("seq", data_item.get("aa"))
        ecnums = data_item.get("EC_number")
        ecnums = ecnums.split(".")[:3]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        for i, ecnum in enumerate(ecnums):
            sites_ids.append(self.tokenizer.get_idx(f"<ec{i + 1}>"))
            sites_ids.append(self.tokenizer.get_idx(ecnum))
        # separator between the EC prefix and the amino-acid sequence
        sites_ids.append(self.tokenizer.get_idx("<prot>"))
        sites_ids.extend([self.tokenizer.get_idx(res) for res in seq])
        # eos
        sites_ids.append(self.tokenizer.eos_idx)

        # a residue / EC token collapsing to <unk> would silently corrupt the
        # conditioning; fail loudly instead (dict-misconfiguration guard).
        assert self.tokenizer.unk_idx not in sites_ids

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.zeros(len(sites_ids), dtype=np.int64)

        item["id"] = index
        item["tokens"] = sites_ids
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_infer_item_ecnum(self, index):
        # Inference prefix for EC-number conditioned enzyme design:
        #   <bos> <ec1> L1 <ec2> L2 <ec3> L3 <prot>
        # (the same prefix the standalone gen_threedimargendiff_ecnum.py builds
        # from its --ecnum argument, assembled here from the data item so the
        # path is dataset-driven like every other target). The model then samples
        # the amino-acid sequence and stops at <coord>; sequence-only, so
        # coordinates_mask is all zeros.
        if self.env is not None:
            with self.env.begin() as txn:
                with txn.cursor() as curs:
                    datapoint_pickled = curs.get(self.keys[index].encode())
                    data_item = pickle.loads(zlib.decompress(datapoint_pickled))
        else:
            data_item = self.data[index]

        item = dict()
        ecnums = data_item.get("EC_number")
        ecnums = ecnums.split(".")[:3]

        # begin with bos
        sites_ids = [self.tokenizer.bos_idx]
        for i, ecnum in enumerate(ecnums):
            sites_ids.append(self.tokenizer.get_idx(f"<ec{i + 1}>"))
            sites_ids.append(self.tokenizer.get_idx(ecnum))
        # separator; generation continues with the amino-acid sequence
        sites_ids.append(self.tokenizer.get_idx("<prot>"))

        sites_ids = np.array(sites_ids)
        coordinates_mask = np.zeros(len(sites_ids), dtype=np.int64)

        item["id"] = data_item.get("id", index)
        item["tokens"] = sites_ids
        item["coordinates_mask"] = coordinates_mask
        return item

    def get_train_item(self, index):
        if self.args.target == "material":
            return self.get_train_item_mat(index)
        elif self.args.target == "mol":
            return self.get_train_item_mol(index)
        elif self.args.target == "prot":
            return self.get_train_item_prot(index)
        elif self.args.target == "cond_mat":
            return self.get_train_cond_mat(index)
        elif self.args.target == "cond_mol":
            return self.get_train_cond_mol(index)
        elif self.args.target == "uni_mat":
            return self.get_train_uni_mat(index)
        elif self.args.target == "uni_mol":
            return self.get_train_uni_mol(index)
        elif self.args.target == "dock":
            return self.get_train_item_docking(index)
        elif self.args.target == "misato":
            return self.get_train_item_misato(index)
        elif self.args.target == "ecnum":
            return self.get_train_item_ecnum(index)
        else:
            raise ValueError(f"Unknown target {self.args.target}")

    def get_infer_item(self, index):
        if self.args.target == "material":
            return self.get_infer_item_mat(index)
        elif self.args.target == "mol":
            return self.get_infer_item_mol(index)
        elif self.args.target == "prot":
            return self.get_infer_item_prot(index)
        elif self.args.target == "cond_mat":
            return self.get_infer_cond_mat(index)
        elif self.args.target == "cond_mol":
            return self.get_infer_cond_mol(index)
        elif self.args.target == "uni_mat":
            return self.get_infer_uni_mat(index)
        elif self.args.target == "uni_mol":
            return self.get_infer_uni_mol(index)
        elif self.args.target == "dock":
            return self.get_infer_item_docking(index)
        elif self.args.target == "misato":
            return self.get_infer_item_misato(index)
        elif self.args.target == "ecnum":
            return self.get_infer_item_ecnum(index)
        else:
            raise ValueError(f"Unknown target {self.args.target }")

    def __getitem__(self, index):
        if self.mode in [MODE.TRAIN, MODE.VAL]:
            return self.get_train_item(index)
        elif self.mode == MODE.INFER:
            return self.get_infer_item(index)

    def _random_rotation(self, lattice):
        # Generate random rotation angles
        angles = np.random.uniform(0, 2 * np.pi, size=3)

        # Compute sine and cosine of angles
        sin_angles = np.sin(angles)
        cos_angles = np.cos(angles)

        # Construct rotation matrix
        rotation_matrix = np.eye(3)

        rotation_matrix[0, 0] = cos_angles[0] * cos_angles[1]
        rotation_matrix[0, 1] = (
            cos_angles[0] * sin_angles[1] * sin_angles[2]
            - sin_angles[0] * cos_angles[2]
        )
        rotation_matrix[0, 2] = (
            cos_angles[0] * sin_angles[1] * cos_angles[2]
            + sin_angles[0] * sin_angles[2]
        )
        rotation_matrix[1, 0] = sin_angles[0] * cos_angles[1]
        rotation_matrix[1, 1] = (
            sin_angles[0] * sin_angles[1] * sin_angles[2]
            + cos_angles[0] * cos_angles[2]
        )
        rotation_matrix[1, 2] = (
            sin_angles[0] * sin_angles[1] * cos_angles[2]
            - cos_angles[0] * sin_angles[2]
        )
        rotation_matrix[2, 0] = -sin_angles[1]
        rotation_matrix[2, 1] = cos_angles[1] * sin_angles[2]
        rotation_matrix[2, 2] = cos_angles[1] * cos_angles[2]

        # Rotate lattice
        lattice = np.dot(rotation_matrix, lattice.T).T

        return lattice

    def collate(self, samples):
        return collate_fn(samples, self.tokenizer, self.mode)
