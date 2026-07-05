# -*- coding: utf-8 -*-
"""Protein-ligand docking evaluation (best-of-N coordinate RMSD).

Reproduces the paper docking metric (the ``RMSD`` notebooks): the generation
jsonl is grouped into consecutive windows of ``--samples_per_target`` records,
one window per target. For every target the metric is the *best* (minimum) RMSD
over all ground-truth x prediction sample pairs in the window (best-of-N). RMSD
is the naive coordinate RMSD (``unigenx.data.docking_utils.calc_rmsd``); the two
coordinate sets are already in the same atom order and frame.

Three coordinate sets are scored when present in the records:

  * ``ligand``   -- ground-truth ligand vs ``prediction.ligand_coords``. The GT
                    ligand key is ``ligand_gt`` (``--target dock``) or
                    ``lig_coords`` (``--target misato``); auto-detected.
  * ``holo``     -- ``holo_coords`` vs ``prediction.holo_coords`` (misato).
  * ``combined`` -- ligand + holo concatenated (misato).

For each set it reports, over all targets, the fraction with best RMSD below
each threshold in ``[2, 4, 6, 8, 10]`` Angstrom, plus the median and mean.

``--symmetry`` switches the ligand RMSD to a symmetry-corrected RMSD via
``spyrmsd`` (optional dependency; the protein/holo sets stay on ``calc_rmsd``).
By default the naive ``calc_rmsd`` is used to match the published numbers.
"""
import argparse
import json
import os
import sys

import numpy as np

# make ``unigenx`` importable when this file is run as a standalone script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from unigenx.data.docking_utils import calc_rmsd  # noqa: E402

THRESHOLDS = [2, 4, 6, 8, 10]


def _as_array(x):
    return None if x is None else np.asarray(x, dtype=np.float64)


def _get_ligand_gt(rec):
    # ``dock`` stores the ground-truth ligand under ``ligand_gt``; ``misato``
    # (and the simple-item records) under ``lig_coords``.
    for key in ("ligand_gt", "lig_coords"):
        if rec.get(key) is not None:
            return _as_array(rec[key])
    return None


def _get_ligand_pred(rec):
    pred = rec.get("prediction") or {}
    return _as_array(pred.get("ligand_coords"))


def _get_holo_gt(rec):
    return _as_array(rec.get("holo_coords"))


def _get_holo_pred(rec):
    pred = rec.get("prediction") or {}
    return _as_array(pred.get("holo_coords"))


def _make_symmetric_rmsd():
    """Return a symmetry-corrected RMSD(coords_gt, coords_pred) closure.

    Uses ``spyrmsd`` if available. spyrmsd needs the molecular graph; without a
    per-record connectivity we fall back to the hungarian (atom-matching) RMSD
    on element-agnostic point sets, which still corrects index permutations.
    """
    from spyrmsd import rmsd as spy_rmsd  # noqa: F401  (import-time availability)

    def _sym(gt, pred):
        gt = np.asarray(gt, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        # element-agnostic symmetric RMSD over positions (permutation corrected)
        anum = np.ones(gt.shape[0], dtype=int)
        adj = np.zeros((gt.shape[0], gt.shape[0]), dtype=int)
        return spy_rmsd.symmrmsd(gt, pred, anum, anum, adj, adj)

    return _sym


def best_of_n_rmsd(gt_list, pred_list, rmsd_fn=calc_rmsd):
    """Minimum RMSD over all ground-truth x prediction pairs for one target."""
    best = np.inf
    for gt in gt_list:
        for pred in pred_list:
            if gt is None or pred is None:
                continue
            if gt.shape != pred.shape:
                continue
            best = min(best, float(rmsd_fn(gt, pred)))
    return best


def _collect(records, samples_per_target):
    """Group records into per-target windows and pull out each coordinate set."""
    n = len(records)
    n_targets = n // samples_per_target
    variants = {
        "ligand": ([], []),  # (gt_lists, pred_lists) per target
        "holo": ([], []),
        "combined": ([], []),
    }
    for k in range(n_targets):
        window = records[k * samples_per_target : (k + 1) * samples_per_target]
        lig_gt = [_get_ligand_gt(r) for r in window]
        lig_pred = [_get_ligand_pred(r) for r in window]
        holo_gt = [_get_holo_gt(r) for r in window]
        holo_pred = [_get_holo_pred(r) for r in window]

        if any(g is not None for g in lig_gt) and any(p is not None for p in lig_pred):
            variants["ligand"][0].append(lig_gt)
            variants["ligand"][1].append(lig_pred)
        if any(g is not None for g in holo_gt) and any(
            p is not None for p in holo_pred
        ):
            variants["holo"][0].append(holo_gt)
            variants["holo"][1].append(holo_pred)
            # combined = ligand ++ holo (both must be present)
            comb_gt = [
                np.concatenate([lg, hg], axis=0)
                if lg is not None and hg is not None
                else None
                for lg, hg in zip(lig_gt, holo_gt)
            ]
            comb_pred = [
                np.concatenate([lp, hp], axis=0)
                if lp is not None and hp is not None
                else None
                for lp, hp in zip(lig_pred, holo_pred)
            ]
            if any(g is not None for g in comb_gt) and any(
                p is not None for p in comb_pred
            ):
                variants["combined"][0].append(comb_gt)
                variants["combined"][1].append(comb_pred)
    return n_targets, variants


def evaluate(records, samples_per_target=100, rmsd_fn=calc_rmsd):
    """Return {variant: {"min_rmsd": [...], "ratios": {t: r}, "median", "mean"}}."""
    n_targets, variants = _collect(records, samples_per_target)
    results = {}
    for name, (gt_lists, pred_lists) in variants.items():
        if not gt_lists:
            continue
        min_rmsd_list = [
            best_of_n_rmsd(gt_lists[k], pred_lists[k], rmsd_fn=rmsd_fn)
            for k in range(len(gt_lists))
        ]
        arr = np.asarray(min_rmsd_list, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        results[name] = {
            "n_targets": len(min_rmsd_list),
            "min_rmsd": min_rmsd_list,
            "ratios": {t: float(np.mean(arr < t)) for t in THRESHOLDS},
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
        }
    return results


def _format(results):
    lines = []
    for name in ("ligand", "holo", "combined"):
        if name not in results:
            continue
        r = results[name]
        lines.append(f"== {name} (n_targets={r['n_targets']}) ==")
        for t in THRESHOLDS:
            lines.append(f"  ratio < {t:>2}A: {r['ratios'][t]:.4f}")
        lines.append(f"  median: {r['median']:.4f}")
        lines.append(f"  mean:   {r['mean']:.4f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="generation jsonl to score")
    parser.add_argument("--output", default=None, help="optional metrics text file")
    parser.add_argument(
        "--samples_per_target",
        type=int,
        default=100,
        help="consecutive records per target (best-of-N window; 100 or 200)",
    )
    parser.add_argument(
        "--symmetry",
        action="store_true",
        help="symmetry-corrected ligand RMSD via spyrmsd (default: naive calc_rmsd)",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    rmsd_fn = calc_rmsd
    if args.symmetry:
        rmsd_fn = _make_symmetric_rmsd()

    results = evaluate(records, args.samples_per_target, rmsd_fn=rmsd_fn)
    report = _format(results)
    print(report)
    if args.output:
        with open(args.output, "w") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
