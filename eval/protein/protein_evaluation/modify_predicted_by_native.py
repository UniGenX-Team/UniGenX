#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections
import dataclasses
import sys
from typing import Any, Tuple

from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.Data.PDBData import protein_letters_3to1_extended as aa3to1


@dataclasses.dataclass(frozen=True)
class Residue:
    name: str
    seqres: str
    is_missing: bool
    resid: str
    atoms: list


def pdb2residues(pdbfile: str) -> Tuple[str, str, str, list]:
    protein = collections.defaultdict(dict)
    with open(pdbfile, "r") as fp:
        for line in fp:
            if line.startswith("ENDMDL"):
                break
            if len(line) < 55:
                continue
            #         1         2         3         4         5         6         7         8
            # 12345678901234567890123456789012345678901234567890123456789012345678901234567890
            # ATOM     32  N  AARG A  -3      11.281  86.699  94.383  0.50 35.88           N
            # ATOM     33  N  BARG A  -3      11.296  86.721  94.521  0.50 35.60           N
            record, altloc, resname = line[:6], line[16], line[17:20]
            if altloc not in (" ", "A"):
                continue
            if record == "ATOM  " or (record == "HETATM" and resname == "MSE"):
                chainid, resnumb = line[21], int(line[22:26].strip())
                current = protein[chainid].get(resnumb, (resname, []))
                current[1].append(line)
                protein[chainid][resnumb] = current
    # fix missing residues
    for chainid, chaindata in protein.items():
        _min, _max = min(chaindata.keys()), max(chaindata.keys())
        for i in range(_min, _max + 1):
            if i not in chaindata:
                chaindata[i] = ("XAA", [])
    # convert to list
    residues = []
    for chainid, chaindata in protein.items():
        for resnumb in sorted(chaindata.keys()):
            resname, lines = chaindata[resnumb]
            residues.append(
                Residue(
                    name=resname,
                    seqres=aa3to1.get(resname, "X"),
                    is_missing=resname == "XAA",
                    resid=f"{chainid}{resnumb:>4d}",
                    atoms=lines,
                )
            )
    return residues


def make_alignmets_by_biopython(seq: str, pdbseq: str) -> Any:
    alignments = PairwiseAligner(scoring="blastp").align(seq, pdbseq)
    if len(alignments) > 1:
        # parameters copy from hh-suite/scripts/renumberpdb.pl
        # https://github.com/soedinglab/hh-suite/blob/master/scripts/renumberpdb.pl
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.open_gap_score = -3
        aligner.target_open_gap_score = -20
        aligner.extend_gap_score = -0.1
        aligner.end_gap_score = -0.09
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        alignments = aligner.align(seq, pdbseq)
    return alignments


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} <native_pdb> <predicted_pdb> <new_pdb>")
    refpdb, rawpdb, newpdb = sys.argv[1:4]
    # print(refpdb, rawpdb, newpdb)

    ref_residues = pdb2residues(refpdb)
    refseq = "".join(_.seqres for _ in ref_residues)
    # print(refseq)

    pdb_residues = pdb2residues(rawpdb)
    pdbseq = "".join(_.seqres for _ in pdb_residues)
    # print(pdbseq)

    alignments = make_alignmets_by_biopython(refseq, pdbseq)
    if len(alignments) == 1:
        ali = alignments[0]
    elif "T1119" in refpdb:
        ali = alignments[2]
    else:
        raise ValueError(f"Multiple alignments between {refpdb} and {rawpdb}")
    # print(ali)
    lines = []
    for i, j in zip(ali.indices[0], ali.indices[1]):
        if i == -1 or j == -1 or ref_residues[i].is_missing:
            continue
        for l in pdb_residues[j].atoms:
            lines.append(l[:21] + ref_residues[i].resid + l[26:])

    with open(newpdb, "w") as fp:
        fp.writelines(lines)
