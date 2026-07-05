# -*- coding: utf-8 -*-
"""TICA free-energy-surface evaluation for protein conformational dynamics (MD).

Reference implementation for the paper's MD analysis: a TICA (time-lagged
independent component analysis) model is fitted on Cα pairwise-distance features
of a *reference* MD trajectory, then generated / test trajectories are projected
onto the leading two independent components (TIC1/TIC2). The resulting 2D density
is the free-energy surface whose agreement with the reference is summarised by
the paper's average free-energy deviation (~0.91 kcal/mol; 1FME 0.64 -> A3D 1.20).

``internal_coordinates_ca_backbone`` and ``convert_to_nparray`` depend only on
numpy + lmdb and are importable/testable without ``deeptime``. The TICA fit and
plotting live in :func:`run_tica_free_energy`, which imports ``deeptime`` (and
matplotlib) lazily so this module can be imported without those heavy,
optional dependencies installed.
"""
import argparse
import itertools
import pickle
import zlib

import lmdb
import numpy as np


def convert_to_nparray(a):
    if isinstance(a, (list, tuple)):  # handle both list and tuple
        return np.asarray(a)
    return a


def internal_coordinates_ca_backbone(lmdb_path: str) -> np.ndarray:
    """Read Cα coordinates from an LMDB dataset and compute all Cα-pair distances.

    Args:
        lmdb_path: path to the processed LMDB dataset (Cα coordinates only).

    Returns:
        np.ndarray: an ``(n_frames, n_pairs)`` distance matrix.
    """
    # open the LMDB environment
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, subdir=False)
    all_distances = []

    with env.begin() as txn:
        # iterate over all entries (skip metadata)
        for key_bytes, value_bytes in txn.cursor():
            if key_bytes == b"__metadata__":
                continue

            # decompress the entry
            data_item = pickle.loads(zlib.decompress(value_bytes))
            ca_coords = data_item["coords"]  # shape: (n_ca_atoms, 3)

            # compute the distance for every Cα pair
            ca_coords = convert_to_nparray(ca_coords)
            n_ca = ca_coords.shape[0]
            pair_idx = np.array(list(itertools.combinations(range(n_ca), 2)))

            # euclidean distance computed manually (avoids an mdtraj dependency)
            diff = ca_coords[pair_idx[:, 0]] - ca_coords[pair_idx[:, 1]]
            dist = np.sqrt(np.sum(diff**2, axis=1)).reshape(1, -1)  # (1, n_pairs)

            all_distances.append(dist)

    # stack all frames
    return np.vstack(all_distances)


def run_tica_free_energy(
    train_lmdb: str,
    test_lmdb: str,
    lagtime: int = 10,
    out_png: str = None,
):
    """Fit TICA on the reference trajectory and project the test trajectory.

    Fits TICA on the reference (``train_lmdb``) Cα-distance features, projects
    the test/generated trajectory (``test_lmdb``) onto the leading two ICs, and
    scatters TIC1/TIC2 (the free-energy-surface projection). ``deeptime`` and
    matplotlib are imported here so the module stays importable without them.
    """
    from deeptime.decomposition import TICA

    distances = internal_coordinates_ca_backbone(train_lmdb)
    distances_test = internal_coordinates_ca_backbone(test_lmdb)

    # dim=2 keeps exactly the first two independent components (TIC1/TIC2), which
    # the free-energy-surface projection below indexes; without it a low-rank /
    # degenerate feature set can yield a single component and crash the TIC2 index.
    tica_estimator = TICA(lagtime=lagtime, dim=2)
    tica = tica_estimator.fit_fetch(distances)

    projected_data = [tica.transform(dist) for dist in distances_test]

    import matplotlib

    if out_png is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # take the first two TICA components
    projected_data = np.vstack(projected_data)
    tica_components = projected_data[:, :2]

    plt.figure(figsize=(10, 8))
    plt.scatter(
        tica_components[:, 0],
        tica_components[:, 1],
        s=5,
        alpha=0.5,
        edgecolors="none",
    )
    plt.title("Distribution of TICA Components")
    plt.xlabel("TIC 1")
    plt.ylabel("TIC 2")

    if out_png is not None:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Saved TICA free-energy-surface projection to {out_png}")
    else:
        plt.show()

    return tica_components


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "TICA free-energy-surface evaluation: fit TICA on a reference MD "
            "trajectory and project a test/generated trajectory onto TIC1/TIC2."
        )
    )
    parser.add_argument(
        "--train_lmdb",
        required=True,
        help="LMDB of the reference (fit) trajectory (Cα coordinates)",
    )
    parser.add_argument(
        "--test_lmdb",
        required=True,
        help="LMDB of the test/generated trajectory to project",
    )
    parser.add_argument("--lagtime", type=int, default=10, help="TICA lag time")
    parser.add_argument(
        "--out_png",
        default=None,
        help="if set, save the TIC1/TIC2 scatter here instead of showing it",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tica_free_energy(
        train_lmdb=args.train_lmdb,
        test_lmdb=args.test_lmdb,
        lagtime=args.lagtime,
        out_png=args.out_png,
    )
