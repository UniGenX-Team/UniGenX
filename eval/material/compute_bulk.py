# -*- coding: utf-8 -*-
"""Bulk modulus estimation for generated crystals.

For each structure a set of isotropically strained copies is built, their
energies are predicted with a machine-learning interatomic potential, and an
equation-of-state (Murnaghan) is fit to the energy-volume curve. The curvature
of that fit gives the bulk modulus (reported in GPa).

The energy predictor is an *injected* dependency: pass an ``energy_predictor``
callable ``list[ase.Atoms] -> sequence[float]``. This keeps the reproducible
core (strain generation + ASE EOS fitting) free of any heavy/internal package,
so it can be exercised with a toy energy function. When no predictor is given,
:func:`compute_bulk_modulus` tries to build a MatterSim/forcefields potential
lazily; that path requires the corresponding package to be installed.
"""
import os

import numpy as np
from ase.eos import EquationOfState
from ase.units import GPa
from pymatgen.io.ase import AseAtomsAdaptor


def is_valid_atoms(atoms, min_distance=0.5):
    """Return True if the minimum interatomic distance is above ``min_distance``."""
    from ase.geometry import get_distances

    positions = atoms.get_positions()
    _, distances = get_distances(positions, positions, cell=atoms.get_cell(), pbc=True)
    np.fill_diagonal(distances, np.inf)
    return np.min(distances) > min_distance


def fit_bulk_modulus(volumes, energies, eos="murnaghan"):
    """Fit an equation of state to (volumes, energies) and return B in GPa.

    Returns ``(v0, e0, bulk_modulus_GPa)``. Raises if the fit fails.
    """
    equation = EquationOfState(list(volumes), list(energies), eos=eos)
    v0, e0, B = equation.fit()
    return v0, e0, B / GPa


def _default_energy_predictor(atoms_list, batch_size):
    """Lazily build a MatterSim/forcefields potential and predict energies.

    Imported lazily so the module (and the reproducible EOS fit) does not depend
    on the internal potential package being installed.
    """
    from forcefields.datasets.utils.build import build_dataloader
    from forcefields.potential import Potential
    from materials.relaxation.relaxation import download_model_weights

    dataloader = build_dataloader(
        atoms_list, batch_size=batch_size, only_inference=True
    )
    potential = Potential.load(load_path=download_model_weights(), device="cuda:0")
    energies, _, __ = potential.predict_properties(dataloader)
    return energies


def compute_bulk_modulus(
    structures,
    energy_predictor=None,
    npoints: int = 5,
    eps_max: float = 0.04,
    max_natoms_per_batch: int = 4000,
    min_distance: float = 0.5,
    failed_dir: str = "failed_structures",
):
    """Estimate the bulk modulus (GPa) of each pymatgen ``Structure``.

    :param structures: list of pymatgen Structure objects.
    :param energy_predictor: callable ``(atoms_list, batch_size) -> energies``.
        If None, a MatterSim/forcefields potential is built lazily.
    :returns: ``(bulk_moduli, failed_indices)`` where ``bulk_moduli`` is a numpy
        array (NaN for structures whose EOS fit failed).
    """
    os.makedirs(failed_dir, exist_ok=True)
    adaptor = AseAtomsAdaptor()
    atoms_list = [adaptor.get_atoms(s) for s in structures]
    max_natoms = max(len(a) for a in atoms_list)
    eps = np.linspace(-eps_max, eps_max, npoints)
    batch_size = max_natoms_per_batch // max_natoms
    if batch_size == 0:
        raise ValueError("max_natoms_per_batch too small for the largest structure")

    atoms_list_distorted = []
    volumes_list = []
    structure_ids = []

    for i, atoms in enumerate(atoms_list):
        for e in eps:
            try:
                atoms_copy = atoms.copy()
                atoms_copy.set_cell(atoms.get_cell() * (1 + e), scale_atoms=True)
                if not is_valid_atoms(atoms_copy, min_distance):
                    raise ValueError("Too-close atoms after distortion")
                atoms_list_distorted.append(atoms_copy)
                volumes_list.append(atoms_copy.get_volume())
                structure_ids.append(i)
            except Exception as err:
                print(f"[Distort Skipped] Structure {i}, strain {e:.3f}: {err}")
                continue

    if energy_predictor is None:
        energy_predictor = _default_energy_predictor
    try:
        energies = energy_predictor(atoms_list_distorted, batch_size)
    except Exception as err:
        raise RuntimeError(f"[ERROR] Inference failed: {err}")

    bulk_moduli = []
    failed_indices = []

    for i in range(len(atoms_list)):
        try:
            idxs = [j for j, sid in enumerate(structure_ids) if sid == i]
            if len(idxs) < npoints:
                raise ValueError(f"Only {len(idxs)} valid distortions (need {npoints})")
            energies_ = [energies[j] for j in idxs]
            volumes_ = [volumes_list[j] for j in idxs]
            _, _, B = fit_bulk_modulus(volumes_, energies_, eos="murnaghan")
            bulk_moduli.append(B)
        except Exception as err:
            print(f"[Fit Failed] Structure {i}: {err}")
            bulk_moduli.append(np.nan)
            failed_indices.append(i)

    print(f"Done. {len(failed_indices)} structures failed.")
    return np.array(bulk_moduli), failed_indices


def load_structures_from_cif(path):
    """Load all ``*.cif`` files under ``path`` as pymatgen Structures."""
    from pathlib import Path

    from pymatgen.io.cif import CifParser

    structures = []
    for cif_file in Path(path).glob("*.cif"):
        parser = CifParser(cif_file)
        structures.append(parser.parse_structures(primitive=True)[0])
    return structures


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimate bulk moduli (GPa) of generated crystals via EOS fit."
    )
    parser.add_argument("cif_dir", type=str, help="directory of generated .cif files")
    parser.add_argument("--hist_out", type=str, default="bulk.pdf")
    cli = parser.parse_args()

    structures = load_structures_from_cif(cli.cif_dir)
    bulk_moduli, _ = compute_bulk_modulus(structures)

    values = bulk_moduli[~np.isnan(bulk_moduli)]
    values = values[(values > 0) & (values < 600)]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=30, color="blue", alpha=0.7, edgecolor="black")
    plt.xlabel("Bulk Modulus (GPa)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(cli.hist_out)
    plt.close()
