# -*- coding: utf-8 -*-
"""Relax generated crystal structures with CHGNet.

Reads a generation ``.jsonl`` (each record carries ``sites`` and a
``prediction`` with ``lattice`` + fractional ``coordinates``), relaxes each
structure with a CHGNet ``StructOptimizer`` and writes the relaxed lattice /
coordinates back. CHGNet is imported lazily so this module can be imported
(and ``get_pred_structure_from_coords`` used) without CHGNet installed.
"""
import json


def get_pred_structure_from_coords(data):
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.structure import Structure

    return Structure(
        lattice=Lattice(data["prediction"]["lattice"]),
        species=[site["element"] for site in data["sites"]],
        coords=data["prediction"]["coordinates"],
    )


def get_relaxed_structure(relaxer, structure, steps=500):
    return relaxer.relax(structure, steps=steps)


def convert(o):
    import numpy as np

    if isinstance(o, np.float32):
        return float(o)
    raise TypeError


def main(input_path, output_path, steps=500):
    from chgnet.model import StructOptimizer
    from tqdm import tqdm

    relaxer = StructOptimizer()
    with open(input_path, "r") as f:
        lines = f.readlines()
    with open(output_path, "w") as fw:
        for line in tqdm(lines):
            data = json.loads(line)
            try:
                structure = get_pred_structure_from_coords(data)
                relaxed_structure = get_relaxed_structure(relaxer, structure, steps)
                data["prediction"]["lattice"] = relaxed_structure[
                    "final_structure"
                ].lattice.matrix.tolist()
                data["prediction"]["coordinates"] = relaxed_structure[
                    "final_structure"
                ].frac_coords.tolist()
            except Exception:
                print("Failed to relax, reserving original structure")
            fw.write(json.dumps(data, default=convert) + "\n")
    print("FINISHED RELAXATION!")


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument("--input", type=str, help="input jsonl file")
    arg_parser.add_argument("--output", type=str, help="output jsonl file")
    arg_parser.add_argument("--steps", type=int, default=500)
    args = arg_parser.parse_args()
    main(args.input, args.output, args.steps)
