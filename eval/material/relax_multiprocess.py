# -*- coding: utf-8 -*-
"""Multiprocess variant of :mod:`relax` (one CHGNet relaxer per worker).

CHGNet is imported lazily inside the worker so importing this module does not
require CHGNet to be installed.
"""
import json
import multiprocessing as mp


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


def process_line(line):
    from chgnet.model import StructOptimizer

    relaxer = StructOptimizer()
    data = json.loads(line)
    try:
        structure = get_pred_structure_from_coords(data)
        relaxed_structure = get_relaxed_structure(relaxer, structure)
        data["prediction"]["lattice"] = relaxed_structure[
            "final_structure"
        ].lattice.matrix.tolist()
        data["prediction"]["coordinates"] = relaxed_structure[
            "final_structure"
        ].frac_coords.tolist()
    except Exception:
        print("Failed to relax, reserving original structure")
    return json.dumps(data, default=convert)


def main(input_path, output_path, num_workers=40):
    from tqdm import tqdm

    with open(input_path, "r") as f:
        lines = f.readlines()

    pool = mp.Pool(processes=num_workers)
    results = list(tqdm(pool.imap(process_line, lines), total=len(lines)))
    with open(output_path, "w") as fw:
        for result in results:
            fw.write(result + "\n")
    pool.close()
    pool.join()
    print("FINISHED RELAXATION!")


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument("input", type=str, help="input jsonl file")
    arg_parser.add_argument("--output", type=str, help="output jsonl file")
    arg_parser.add_argument(
        "--num_workers", type=int, default=40, help="number of processes to use"
    )
    args = arg_parser.parse_args()
    main(args.input, args.output, args.num_workers)
