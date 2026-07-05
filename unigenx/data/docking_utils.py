# -*- coding: utf-8 -*-
"""Protein-ligand docking helpers (shared by the dataset layer and the docking
evaluation).

Ported verbatim (behaviour-preserving) from the docking source. ``calc_rmsd`` is
the naive coordinate RMSD used to reproduce the paper docking numbers -- it does
*not* align or symmetry-correct the two structures, so the two coordinate sets
must already be in the same atom order and frame. ``Mol2SmilesCoords`` /
``get_mol`` / ``normalize_coords`` build the RDKit ligand and its
SMILES-order coordinates for the dataset collation.

Depends only on numpy + RDKit (no absolute paths, no heavy deps).
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.error")
ptable = rdchem.GetPeriodicTable()


def normalize_coords(coordinates):
    return coordinates - coordinates.mean(axis=0)


def Mol2SmilesCoords(mol, conf_id=None, canonical=False, return_map=False):
    mol = Chem.RemoveHs(mol)
    if conf_id is not None:
        conf = mol.GetConformer(conf_id)
    else:
        conf = mol.GetConformer()
    smiles = Chem.MolToSmiles(mol, canonical=canonical, doRandom=(not canonical))
    mol_smiles = Chem.MolFromSmiles(smiles)
    atom_map = {
        atom.GetIdx(): mol.GetSubstructMatch(mol_smiles)[atom.GetIdx()]
        for atom in mol_smiles.GetAtoms()
    }
    atom_seq = [atom.GetSymbol() for atom in mol_smiles.GetAtoms()]
    coordinates = np.array(
        [
            conf.GetAtomPosition(atom_map[atom.GetIdx()])
            for atom in mol_smiles.GetAtoms()
        ]
    )
    if return_map:
        return smiles, atom_seq, coordinates, atom_map
    return smiles, atom_seq, coordinates


def get_mol(atom_ids, bond_starts, bond_ends, bond_types, coords):
    mol = Chem.RWMol()
    atoms = []
    for atom_id in atom_ids:
        atom = Chem.Atom(atom_id)
        mol.AddAtom(atom)
        atoms.append(ptable.GetElementSymbol(atom_id))

    for start, end, bond_type in zip(bond_starts, bond_ends, bond_types):
        if start > end:
            continue
        if bond_type == 1:
            mol.AddBond(start, end, rdchem.BondType.SINGLE)
        elif bond_type == 2:
            mol.AddBond(start, end, rdchem.BondType.DOUBLE)
        elif bond_type == 3:
            mol.AddBond(start, end, rdchem.BondType.TRIPLE)
        elif bond_type == 4:
            mol.AddBond(start, end, rdchem.BondType.AROMATIC)

    conformer = Chem.Conformer(len(atom_ids))
    for i in range(len(atom_ids)):
        conformer.SetAtomPosition(i, Point3D(coords[i][0], coords[i][1], coords[i][2]))
    mol.AddConformer(conformer)
    try:
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None


def calc_rmsd(matrix1, matrix2):
    """Naive RMSD between two (N, 3) coordinate matrices (no alignment).

    rmsd = sqrt(mean_i(sum_j (m1[i, j] - m2[i, j])**2)).
    """
    matrix1 = np.asarray(matrix1)
    matrix2 = np.asarray(matrix2)
    if matrix1.shape != matrix2.shape or matrix1.shape[1] != 3:
        raise ValueError("Both matrices must have the shape (N, 3).")

    diff_squared = np.square(matrix1 - matrix2)
    rmsd = np.sqrt(np.mean(np.sum(diff_squared, axis=1)))

    return rmsd
