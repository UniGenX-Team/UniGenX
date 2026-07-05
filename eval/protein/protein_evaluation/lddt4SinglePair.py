# -*- coding: utf-8 -*-
"""LDDT a predicted pdb against a native pdb.

Wraps the external ``lddt`` program (OpenStructure). The executable is looked up
on ``$PATH`` (override with the ``binary`` argument / ``--lddt_bin`` on the
command line); when it is missing the initialised score dict is returned
unchanged so callers degrade gracefully instead of crashing.
"""
import os
import sys
from typing import Any, Mapping

from utils import check_output_lines


def check_lddt(name: str = "lddt") -> bool:
    """Check whether `name` is on PATH and marked as executable."""
    from shutil import which

    return which(name) is not None


def lddt4SinglePair(
    predicted_pdb: str, native_pdb: str, binary: str = "lddt"
) -> Mapping[str, Any]:
    """Calculate model score by using the lddt program"""
    # intialization
    score = {
        "PredictedPDB": os.path.basename(predicted_pdb),
        "NativePDB": os.path.basename(native_pdb),
        "PredictedLen": 0,
        "NativeLen": 0,
        "AlignLen": 0,
        "Radius": 0.0,
        "Coverage": 0.0,
        "LDDT": 0.0,
        "LocalLDDT": [],
    }

    if not check_lddt(binary):
        print(f"'{binary}' does not exist in $PATH", file=sys.stderr)
        return score

    if not os.path.exists(predicted_pdb):
        print(f"cannot found predicted model {predicted_pdb}", file=sys.stderr)
        return score

    if not os.path.exists(native_pdb):
        print(f"cannot found native structure {native_pdb}", file=sys.stderr)
        return score

    # execuate command and get output
    cmds = [binary, "-c", predicted_pdb, native_pdb]
    lines = check_output_lines(cmds)

    # parse model score
    start_local = False
    local_lddts = []
    for i, l in enumerate(lines):
        cols = l.split()
        if l.startswith("Inclusion") and len(cols) > 2:
            score["Radius"] = float(cols[2])
        elif l.startswith("Coverage") and len(cols) > 6:
            score["Coverage"] = float(cols[1])
            score["NativeLen"] = int(cols[5])
        elif l.startswith("Global") and len(cols) > 3:
            score["LDDT"] = float(cols[3])
        elif l.startswith("Local"):
            continue
        elif l.startswith("Chain"):
            start_local = True
        elif start_local and len(cols) > 5:
            local_lddts.append(cols)
        else:
            continue
    score["PredictedLen"] = len(local_lddts)
    score["AlignLen"] = sum([_[4] != "-" for _ in local_lddts])
    score["LocalLDDT"] = [
        float("nan") if _[4] == "-" else float(_[4]) for _ in local_lddts
    ]

    # check data format and lddt output
    if len(lines) == i - 1 or lines[-1] != "\n":
        print("ERROR: wrong lddt results", file=sys.stderr)

    return score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LDDT a predicted pdb against a native pdb."
    )
    parser.add_argument("predicted_pdb", help="Predicted model PDB file.")
    parser.add_argument("native_pdb", help="Native structure PDB file.")
    parser.add_argument(
        "--lddt_bin",
        default="lddt",
        help="lddt executable name or path (looked up on $PATH by default).",
    )
    args = parser.parse_args()

    # lddt between predicted and native pdb
    score = lddt4SinglePair(args.predicted_pdb, args.native_pdb, binary=args.lddt_bin)
    keys = [
        "PredictedPDB",
        "NativePDB",
        "PredictedLen",
        "NativeLen",
        "AlignLen",
        "Radius",
        "Coverage",
        "LDDT",
        "LocalLDDT",
    ]
    print(" ".join(["%s" % score.get(_, "XXXXX") for _ in keys[:2]]), end=" ")
    print(" ".join(["%5d" % score.get(_, 0) for _ in keys[2:5]]), end=" ")
    print(" ".join(["%10.3f" % score.get(_, 0.0) for _ in keys[5:8]]))

    print(sys.argv[0], "Done.")
