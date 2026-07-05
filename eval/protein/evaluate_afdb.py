# -*- coding: utf-8 -*-
"""Protein structure-prediction evaluation against native structures (AFDB test).

Reads a UniGenX prediction ``.jsonl`` (one record per protein, each carrying the
amino-acid sequence, the ground-truth Cα coordinates ``pos`` and the model
``prediction.coordinates``), writes both the ground-truth and the predicted
structures out as Cα-only PDB files, and scores every prediction against its
native structure with **TM-score** (via the external ``USalign`` program) and
**RMSD** (via PyMOL rigid Cα alignment). It reports the average RMSD / TM-score
and, optionally, saves aligned cartoon images for a handful of rank positions.

External dependencies (lazy-imported so this module stays importable without
them, and skipped/erroring only when actually used):

* ``USalign`` -- external binary, looked up on ``$PATH`` (override with
  ``--usalign_bin``). Produces the TM-score.
* ``pymol``   -- Python package providing ``pymol.cmd``; used for the RMSD
  alignment and the optional aligned-structure images.

Neither is required to import this file; both are needed to run the ``__main__``
scoring pipeline.
"""
import argparse
import json
import os
import re
import shutil
import subprocess

import numpy as np
from tqdm import tqdm


def calculate_tmscore(pdb_file1, pdb_file2, usalign_bin="USalign"):
    """Compute the TM-score between two protein structures using US-align.

    ``usalign_bin`` is looked up on ``$PATH`` (or given as an absolute path);
    returns ``None`` when the binary is missing or the score cannot be parsed.
    """
    if shutil.which(usalign_bin) is None:
        print(
            f"'{usalign_bin}' executable not found. Please ensure US-align is "
            "installed and on your PATH (or pass --usalign_bin)."
        )
        return None

    # build the US-align command
    command = [usalign_bin, pdb_file1, pdb_file2]

    try:
        # run the command and capture its output
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # check whether it ran successfully
        if result.returncode != 0:
            print(f"Error running US-align: {result.stderr}")
            return None

        # parse the TM-score out of the output
        output = result.stdout
        tm_score_search = re.search(r"TM-score=\s*(\d+\.\d+)", output)
        if tm_score_search:
            tm_score = float(tm_score_search.group(1))
            return round(tm_score, 4)
        else:
            print("TM-score not found in US-align output.")
            return None

    except FileNotFoundError:
        print(
            f"'{usalign_bin}' executable not found. Please ensure US-align is "
            "installed and on your PATH (or pass --usalign_bin)."
        )
        return None


def calculate_rmsd(
    pdb_file1, pdb_file2, selection="name CA", mobile_obj="mol1", target_obj="mol2"
):
    """Compute the RMSD between two protein structures with PyMOL.

    :param pdb_file1: first PDB file (mobile structure)
    :param pdb_file2: second PDB file (target structure)
    :param selection: atom selection used for alignment / RMSD (default: Cα only)
    :param mobile_obj: object name for the first structure
    :param target_obj: object name for the second structure
    """
    from pymol import cmd  # lazy: PyMOL is an optional external dependency

    # load the PDB files
    cmd.load(pdb_file1, mobile_obj)
    cmd.load(pdb_file2, target_obj)

    # rigid alignment
    alignment_rms = cmd.align(
        f"{mobile_obj} and {selection}", f"{target_obj} and {selection}"
    )

    # RMSD after alignment
    rmsd_value = alignment_rms[0]

    # clear the loaded objects
    cmd.delete("all")

    return round(rmsd_value, 4)


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


def align_and_save_image(pdb1, pdb2, output_image):
    """
    Align two proteins from PDB files and save the aligned image.

    :param pdb1: Path to the first PDB file
    :param pdb2: Path to the second PDB file
    :param output_image: Path to save the output image
    """
    from pymol import cmd  # lazy: PyMOL is an optional external dependency

    # Load the PDB files
    cmd.load(pdb1, "protein1")
    cmd.load(pdb2, "protein2")

    # Align the second protein to the first protein
    alignment_score = cmd.align("protein2", "protein1")
    print(f"Alignment score: {alignment_score}")

    # Set visualization style
    cmd.hide("everything")
    cmd.show("cartoon", "protein1")
    cmd.show("cartoon", "protein2")
    cmd.color("blue", "protein1")
    cmd.color("red", "protein2")

    # Zoom and save the image
    cmd.zoom()
    cmd.png(output_image, width=1920, height=1080, dpi=300)
    cmd.delete("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score UniGenX protein predictions vs. native structures "
        "(TM-score via US-align, RMSD via PyMOL)."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Prediction .jsonl (each record has 'aa', 'pos' and "
        "'prediction.coordinates').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write gt/ and pred/ PDB files and aligned images.",
    )
    parser.add_argument(
        "--usalign_bin",
        type=str,
        default="USalign",
        help="US-align executable name or path (looked up on $PATH by default).",
    )
    parser.add_argument(
        "--pred_scale",
        type=float,
        default=10.0,
        help="Scale applied to predicted coordinates when writing PDBs.",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Also save aligned cartoon images for a set of rank positions "
        "(requires PyMOL).",
    )
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]

    all_rmsd = []
    all_tmscore = []
    data_path = args.output_dir
    os.makedirs(os.path.join(data_path, "gt"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "pred"), exist_ok=True)
    for i, x in enumerate(tqdm(lines)):
        x["coordinates"] = np.array(x["pos"])
        x["seq"] = "".join(x["aa"])
        gt_pdb = os.path.join(data_path, "gt", f"gt_{i}.pdb")
        pred_pdb = os.path.join(data_path, "pred", f"pred_{i}.pdb")
        write_pdb(x["seq"], x["coordinates"], gt_pdb)
        write_pdb(
            x["seq"], x["prediction"]["coordinates"], pred_pdb, scale=args.pred_scale
        )
        rmsd = calculate_rmsd(gt_pdb, pred_pdb)
        tmscore = calculate_tmscore(gt_pdb, pred_pdb, usalign_bin=args.usalign_bin)
        all_rmsd.append(rmsd)
        all_tmscore.append(tmscore)

    print(f"Average RMSD: {sum(all_rmsd)/len(all_rmsd)}")
    print(f"Average TM-score: {sum(all_tmscore)/len(all_tmscore)}")
    sorted_indices = np.argsort(all_tmscore)[::-1]
    sorted_indices = sorted_indices.tolist()

    if args.save_images:
        png_rank = [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        # keep only ranks that exist for this set
        png_rank = [rk for rk in png_rank if rk <= len(sorted_indices)]
        output_image_path = os.path.join(data_path, "align_images")
        if os.path.exists(output_image_path):
            shutil.rmtree(output_image_path)
        os.makedirs(output_image_path, exist_ok=True)
        for rk in png_rank:
            idx = sorted_indices[rk - 1]
            gt = os.path.join(data_path, "gt", f"gt_{idx}.pdb")
            pred = os.path.join(data_path, "pred", f"pred_{idx}.pdb")
            output_path = os.path.join(output_image_path, f"rank_{rk}_pdb{idx}.png")
            align_and_save_image(gt, pred, output_path)
