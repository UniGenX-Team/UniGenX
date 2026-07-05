#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate UniGenX protein-structure predictions against the CASP14+15 / CAMEO
test set (per-domain TM-score / RMSD / LDDT, top-1 and top-5).

Pipeline (run as ``__main__``):

1. Read the prediction ``.jsonl`` (each record: ``keys``, ``seq``,
   ``prediction.coordinates`` -- a list of ``max_model_num`` predicted
   Cα-coordinate sets). Write each valid model out as ``<key>-<i>.pdb`` under a
   ``pred_pdbs`` directory next to the prediction file.
2. Load the benchmark metadata (native PDBs + per-target domain definitions)
   from an LMDB database (``--proteintest_lmdb``); the ``__metadata__`` entry
   holds ``keys`` / ``pdbs`` / ``domains`` / ``types``.
3. For every target domain, score the predicted model(s) against the native
   structure with TM-score and LDDT (per-domain residue selection), aggregate
   the top-1 / top-5 scores and average per CASP/CAMEO category.

External prerequisites (not shipped): the ``TMscore`` and ``lddt`` executables
(on ``$PATH``) and the benchmark LMDB database. ``lmdb`` is a declared Python
dependency.
"""
import argparse
import json
import logging
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Union

import lmdb
import pandas as pd
from lddt4SinglePair import lddt4SinglePair
from LGA4SinglePair import LGA4SinglePair
from TMscore4SinglePair import TMscore4SinglePair
from tqdm import tqdm
from utils import bstr2obj, write_pdb

logger = logging.getLogger(__name__)


def calculate_score(predlines: list, natilines: list, residx: set) -> dict:
    """Calculate score between predicted and native structure by TM-score"""

    def _select_residues_by_residx(atomlines: list):
        lines = []
        for line in atomlines:
            if line.startswith("ATOM"):
                resnum = int(line[22:26].strip())
                if resnum in residx:
                    lines.append(line)
        lines.append("TER\n")
        lines.append("END\n")
        return lines

    with (
        tempfile.NamedTemporaryFile() as predpdb,
        tempfile.NamedTemporaryFile() as natipdb,
    ):
        with open(predpdb.name, "w") as fp:
            fp.writelines(_select_residues_by_residx(predlines))
        with open(natipdb.name, "w") as fp:
            fp.writelines(_select_residues_by_residx(natilines))
        score = TMscore4SinglePair(predpdb.name, natipdb.name)
        print(score)
        score["LDDT"] = lddt4SinglePair(predpdb.name, natipdb.name)["LDDT"]
        return score


def evaluate_predicted_structure(
    metadata: Mapping[str, Union[list, str]],
    preddir: str,
    max_model_num: int = 1,
) -> pd.DataFrame:
    scores = []
    for target in tqdm(metadata["keys"]):
        taridx = metadata["keys"].index(target)
        # calculate score for each domain
        for domstr, domlen, domgroup in metadata["domains"][taridx]:
            try:
                residx = set()
                domseg = domstr.split(":")[1]
                for seg in domseg.split(","):
                    start, finish = [int(_) for _ in seg.split("-")]
                    residx.update(range(start, finish + 1))
                assert domlen == len(residx), f"domain length!={domlen}"
            except Exception as e:
                logger.error(f"Domain {domstr} parsing error, {e}")
                continue

            # process score for each predicted model
            for num in range(1, max_model_num + 1):
                score = {
                    "Target": domstr.split(":")[0],
                    "Length": domlen,
                    "Group": domgroup,
                    "Type": metadata["types"][taridx],
                    "ModelIndex": num,
                }
                try:
                    pdb_file = os.path.join(preddir, f"{target}-{num}.pdb")
                    if not os.path.exists(pdb_file):
                        available_nums = []
                        for filename in os.listdir(preddir):
                            if filename.startswith(f"{target}-") and filename.endswith(
                                ".pdb"
                            ):
                                try:
                                    available_nums.append(
                                        int(filename.split("-")[-1].split(".")[0])
                                    )
                                except ValueError:
                                    continue
                        if not available_nums:
                            raise FileNotFoundError(
                                f"No alternative PDB files found for target '{target}' in '{preddir}'."
                            )
                        new_num = random.choice(available_nums)
                        pdb_file = os.path.join(preddir, f"{target}-{new_num}.pdb")
                    with open(pdb_file, "r") as fp:
                        predlines = fp.readlines()
                    len_predlines = set()
                    for line in predlines:
                        len_predlines.add(len(line))
                    if len(len_predlines) != 1:
                        logger.error(f"Wrong predicted file {pdb_file}")
                        continue
                    assert predlines, f" wrong predicted file {pdb_file}"
                    natilines = metadata["pdbs"][taridx]
                    score.update(calculate_score(predlines, natilines, residx))
                except Exception as e:
                    logger.error(f"Failed to evaluate {domstr}, {e}.")
                    continue
                scores.append(score)
    df = pd.DataFrame(scores)
    # save to csv
    df.to_csv("Score4EachModel.csv")
    return df


def calculate_average_score(df: pd.DataFrame) -> pd.DataFrame:
    CATEGORY = {
        "CAMEO  Easy": ["Easy"],
        "CAMEO  Medi": ["Medium", "Hard"],
        "CASP14 Full": ["MultiDom"],
        "CASP15 Full": ["MultiDom"],
        "CASP14 Easy": ["TBM-easy", "TBM-hard"],
        "CASP14 Hard": ["FM/TBM", "FM"],
        "CASP15 Easy": ["TBM-easy", "TBM-hard"],
        "CASP15 Hard": ["FM/TBM", "FM"],
    }
    # group score by target
    records = []
    for target, gdf in df.groupby("Target"):
        record = {
            "Target": target,
            "Length": gdf["Length"].iloc[0],
            "Group": gdf["Group"].iloc[0],
            "Type": gdf["Type"].iloc[0],
        }
        max_model_num = gdf["ModelIndex"].max()
        for col in ["TMscore", "RMSD", "GDT_TS", "LDDT"]:
            maxscore = float("-inf")
            for num in range(1, max_model_num + 1):
                try:
                    score = gdf[gdf["ModelIndex"] == num][col].iloc[0]
                except:
                    score = None
                record[f"Model{num}_{col}"] = score
                if score is not None:
                    maxscore = max(maxscore, score)
            record[f"ModelMax_{col}"] = maxscore
        records.append(record)
    newdf = pd.DataFrame(records)
    # calculate average score for each category
    scores = []
    for key, groups in CATEGORY.items():
        _type = key.split()[0]
        subdf = newdf[(newdf["Type"] == _type) & newdf["Group"].isin(groups)]
        scores.append(
            {
                "CatAndGroup": key,
                "Number": len(subdf),
                "Top1TMscore": subdf["Model1_TMscore"].mean() * 100,
                "Top5TMscore": subdf["ModelMax_TMscore"].mean() * 100,
                "Top1LDDT": subdf["Model1_LDDT"].mean() * 100,
                "Top5LDDT": subdf["ModelMax_LDDT"].mean() * 100,
            }
        )
    meandf = pd.DataFrame(scores).set_index("CatAndGroup")
    return newdf, meandf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process protein LMDB and prediction files."
    )
    parser.add_argument(
        "--proteintest_lmdb",
        type=str,
        required=True,
        help="Path to the benchmark metadata LMDB database.",
    )
    parser.add_argument(
        "--prediction_file",
        type=str,
        required=True,
        help="Path to the prediction .jsonl file.",
    )
    parser.add_argument(
        "--max_model_num",
        type=int,
        default=5,
        help="Maximum number of models (default: 5).",
    )

    args = parser.parse_args()

    inplmdb = args.proteintest_lmdb
    predfile = args.prediction_file
    max_model_num = args.max_model_num
    preddir = os.path.join(os.path.dirname(predfile), "pred_pdbs")
    if not os.path.exists(preddir):
        os.makedirs(preddir, exist_ok=True)
        # extract predicted pdbs
    with open(predfile, "r") as f:
        pred_lines = f.readlines()
        pred_lines = [json.loads(line) for line in pred_lines]
    for pred in tqdm(pred_lines):
        invalid = []
        for i in range(max_model_num):
            fname = pred["keys"] + f"-{i+1}.pdb"
            for line in pred["prediction"]["coordinates"][i]:
                for c in line:
                    if math.isnan(c):
                        invalid.append(i)
            if i not in invalid:
                write_pdb(
                    pred["seq"],
                    pred["prediction"]["coordinates"][i],
                    os.path.join(preddir, fname),
                    scale=10,
                )
        if len(invalid) == max_model_num:
            logger.warning(f"All models for {pred['keys']} are invalid, skip.")

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    logger.info(f"Loading metadata from {inplmdb}.")
    with lmdb.open(
        inplmdb, subdir=True, readonly=True, lock=False, readahead=False
    ).begin(write=False) as txn:
        metadata = bstr2obj(txn.get("__metadata__".encode()))
    logger.info(f"Metadata contains {len(metadata['keys'])} keys, pdbs, ...")

    logger.info(f"TMscore between predicted.pdb and native.pdb {preddir}. ")
    df = evaluate_predicted_structure(metadata, preddir, max_model_num)
    print(df)

    logger.info("Average TMscore for different categories.")
    df.to_csv(Path(preddir) / "Score4EachModel.csv")
    df = pd.read_csv(Path(preddir) / "Score4EachModel.csv")
    newdf, meandf = calculate_average_score(df)
    newdf.to_csv(Path(preddir) / "Score4Target.csv")
    print(newdf)
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(meandf)
