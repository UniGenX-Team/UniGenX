# -*- coding: utf-8 -*-
"""Property-conditional molecule generation: 3D-structure front-end.

Rebuilds a 3D RDKit molecule for every generated ``cond_mol`` record (SMILES +
predicted coordinates), MMFF-optimizes it, and groups the molecules by the
conditioning property. The grouped ``(mol3d, condition)`` structures are what a
downstream property calculator (e.g. ``evaluate_mol_prop.py``) consumes to score
the property MAE / constraint satisfaction of the generated ensembles.

Input is the ``.jsonl`` produced by ``unigenx_infer.py --target cond_mol``: one
record per molecule with ``"smi"``, ``"coordinates"``, ``"prop"`` and
``"prop_val"`` fields.

Note: this is the single-property front-end (one conditioning property per
molecule). ``prop``/``prop_val`` may be written by inference as per-constraint
lists; for the single-property case the sole entry is used as the group key.
"""
import argparse
import json

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from tqdm import tqdm


def generate_mol_struct(gen_seqs_file):
    with open(gen_seqs_file, "r") as f:
        mol3d = dict()
        condition = dict()

        lines = f.readlines()
        data = [json.loads(line) for line in lines]

        for i in tqdm(range(len(data))):
            data_item = data[i]
            positions = data_item["coordinates"]
            smiles = data_item["smi"]
            prop = data_item["prop"]
            prop_val = data_item["prop_val"]
            # cond_mol inference stores prop/prop_val as per-constraint lists;
            # this single-property front-end groups by the (sole) property.
            if isinstance(prop, (list, tuple)):
                prop = prop[0]
            if isinstance(prop_val, (list, tuple)):
                prop_val = prop_val[0]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # invalid SMILES -> not a valid generation, skip it
                continue
            positions = np.array(positions)
            mol.Compute2DCoords()
            conf = mol.GetConformer()
            if mol.GetNumAtoms() != positions.shape[0]:
                positions = positions[: mol.GetNumAtoms(), :]
            assert mol.GetNumAtoms() == positions.shape[0]
            for jdx in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(
                    jdx,
                    Point3D(
                        positions[jdx, 0],
                        positions[jdx, 1],
                        positions[jdx, 2],
                    ),
                )
            mol = Chem.AddHs(mol)
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                # print("failed")
                continue
            if prop not in condition:
                condition[prop] = [prop_val]
                mol3d[prop] = [mol]
            else:
                condition[prop].append(prop_val)
                mol3d[prop].append(mol)

    return mol3d, condition


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build 3D RDKit molecules from cond_mol generation output and group "
            "them by conditioning property."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="generated .jsonl from unigenx_infer.py --target cond_mol",
    )
    args = parser.parse_args()

    mol3d, condition = generate_mol_struct(args.input)
    for prop in sorted(condition):
        print(f"property {prop}: {len(mol3d[prop])} valid 3D molecules")
