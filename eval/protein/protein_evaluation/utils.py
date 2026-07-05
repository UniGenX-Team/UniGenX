# -*- coding: utf-8 -*-
"""
utils.py
Copyright
Author: zhujianwei@ict.ac.cn (Jianwei Zhu)

This module provides utility functions that are used within program
that are also useful for external consumption.
"""
from __future__ import absolute_import, division, print_function

import operator
import os
import pickle
import subprocess
import sys
import tempfile
import zlib


def check_output_file(command, filename):
    """Exculate a command and return output to a file."""

    print(" ".join(command))

    # open file and write the output to this file
    with open(filename, "w") as tmp:
        proc = subprocess.Popen(command, stdout=tmp, stderr=tmp)

        return proc.wait()


def check_output_stdout(command):
    """Exculate a command and return output to stdout."""

    print(" ".join(command))

    # write the output to stdout
    proc = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stdout)

    return proc.wait()


def check_output_lines(command):
    """Exculate a command and return output to a list."""

    # open a temporary file and write the output to this file
    lines = None
    with tempfile.TemporaryFile() as tmp:
        proc = subprocess.Popen(command, stdout=tmp, stderr=tmp)
        proc.wait()

        tmp.seek(0)
        lines = [_.decode("utf-8") for _ in tmp.readlines()]
    return lines


def parse_listfile(listfile, col_list=None):
    """Parse list file from columns list."""

    lines = []
    try:
        with open(listfile, "r") as fin:
            if col_list:
                for line in fin:
                    cols = line.split()
                    lines.append(tuple(cols[i - 1] for i in col_list))
            else:
                for line in fin:
                    lines.append(tuple(line.split()))

    except Exception as e:
        print('ERROR: wrong list file "%s"\n      ' % listfile, e, file=sys.stderr)

    return lines


def parse_fastafile(fastafile):
    """Parse fasta file."""

    seqs = []
    try:
        with open(fastafile, "r") as fin:
            header, seq = "", []
            for line in fin:
                if line[0] == ">":
                    seqs.append((header, "".join(seq)))
                    header, seq = line.strip(), []
                else:
                    seq.append(line.strip())
            seqs.append((header, "".join(seq)))
            del seqs[0]

    except Exception as e:
        print('ERROR: wrong fasta file "%s"\n      ' % fastafile, e, file=sys.stderr)

    return seqs


def parse_protein_id(filename):
    """
    Parse protein name from path name
    filename = "/tmp/d1a3aa_" --> d1a3aa_
    filename = "/tmp/d1a3aa_.fasta" --> d1a3aa_
    """

    base = os.path.basename(filename)
    protein_id = os.path.splitext(base)[0]
    # name = filename.split('/')[-1]
    # protein_id = '.'.join(name.split('.')[:-1]) or name

    return protein_id


def check_outdir(outdir):
    """Check output directory. If it is not exist, create it"""

    if not os.path.exists(outdir):
        print("Output directory create %s" % outdir)
        os.makedirs(outdir)
    else:
        print("Output directory exists %s" % outdir)


def accumulate(iterable, func=operator.add):
    """
    Return running totals
    accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    """

    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = func(total, element)
        yield total


def obj2bstr(obj):
    return zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def bstr2obj(bstr: bytes):
    return pickle.loads(zlib.decompress(bstr))


def write_pdb(sequence, coordinates, output_file="output.pdb", scale: float = 1.0):
    """
    Write an amino acid sequence and corresponding Cα coordinates to a PDB format file.

    :param sequence: Amino acid sequence (str)
    :param coordinates: Corresponding Cα coordinates (list of tuples, e.g., [(x1, y1, z1), (x2, y2, z2), ...])
    :param output_file: Name of the output PDB file (str)
    """
    # Ensure the sequence and coordinates have matching lengths
    if len(sequence) != len(coordinates):
        raise ValueError("The sequence and coordinates lengths do not match.")

    # Standard mapping from one-letter to three-letter amino acid codes
    aa_map = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    }

    # Open the output file for writing
    with open(output_file, "w") as pdb_file:
        atom_index = 1  # Atom index counter
        res_index = 1  # Residue index counter

        # Iterate over residues and their coordinates
        for res, coord in zip(sequence, coordinates):
            # Get the three-letter code for the current amino acid
            res_name = aa_map.get(
                res.upper(), "UNK"
            )  # Default to "UNK" for unknown amino acids
            if res_name == "UNK":
                print(f"Warning: Unknown amino acid {res} encountered.")

            # Unpack the coordinates (x, y, z)
            x, y, z = coord
            x, y, z = scale * x, scale * y, scale * z  # Convert to Angstroms

            # Construct an ATOM line in PDB format (without chain ID)
            pdb_line = (
                f"ATOM  {atom_index:5d}  CA  {res_name}  {res_index:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            pdb_file.write(pdb_line)

            # Increment counters
            atom_index += 1
            res_index += 1
