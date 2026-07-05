#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate predicted structures for a target list against CASP14+15 / CAMEO
reference servers (TM-score / GDT_TS / GDT_HA / LDDT, top-1 and top-5).

Given a list of CASP/CAMEO targets and a directory of official-format
predictions, this collects, for each target, the model files of a set of
reference servers (AlphaFold2, RoseTTAFold, ESMFold, SFM, ...), scores each one
against the native structure with the requested criterion, and reports the
per-category (CAMEO Easy/Medium, CASP14/15 Easy/Hard/Full) averages.

External prerequisites (not shipped):

* The scoring executables on ``$PATH``: ``TMscore`` (criterion ``TMscore``),
  ``lddt`` (criterion ``LDDT``) and the LGA wrapper ``runlga.mol_mol.pl``
  (criteria ``GDT_TS`` / ``GDT_HA``).
* ``metadata4target`` -- the CASP/CAMEO domain-definition + classification
  metadata, an external benchmark table (see :func:`_load_metadata4target`).
* The official prediction directory layout expected by :func:`_collect_models`.

``click``, ``pandas``, ``joblib`` and ``tqdm`` are Python dependencies.
"""
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import click
import pandas as pd
from joblib import Parallel, delayed
from lddt4SinglePair import lddt4SinglePair
from LGA4SinglePair import LGA4SinglePair
from TMscore4SinglePair import TMscore4SinglePair
from tqdm import tqdm

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_metadata4target() -> Mapping[str, Any]:
    """Lazily load the CASP/CAMEO ``metadata4target`` mapping.

    ``metadata4target`` maps each target id to its domain definitions, sequence
    length and benchmark type (CAMEO / CASP14 / CASP15). It is built from the
    official CASP domain-definition and CAMEO chain-definition tables, which are
    an external benchmark prerequisite and are not shipped in this repository.
    Provide a ``metadata`` module (importable as ``metadata`` or
    ``protein_data_process.metadata``) that exposes ``metadata4target`` -- e.g.
    alongside the CASP/CAMEO definition CSVs -- to run the target-list
    evaluation. Imported lazily so this module stays importable without it.
    """
    try:
        from metadata import metadata4target
    except ImportError:
        try:
            from protein_data_process.metadata import metadata4target
        except ImportError as e:
            raise ImportError(
                "metadata4target (CASP/CAMEO domain-definition metadata) is "
                "required for the target-list evaluation but was not found. "
                "Provide a `metadata` module exposing `metadata4target` (built "
                "from the official CASP/CAMEO definition tables)."
            ) from e
    return metadata4target


@click.group()
def cli():
    pass


def _get_servers(target: str) -> Mapping[str, str]:
    servers = OrderedDict()
    if len(target) >= 5 and target[0] == "T" and 1104 <= int(target[1:5]) <= 1197:
        # casp14 targets
        servers.update(
            {
                "229": "Yang-Server",
                "185": "BAKER",
                "270": "NBIS-AF2-std",
            }
        )
    elif len(target) >= 5 and target[0] == "T" and 1024 <= int(target[1:5]) <= 1101:
        # casp15 targets
        servers.update(
            {
                "427": "AlphaFold2",
                "473": "BAKER",
                "324": "Zhang-Server",
            }
        )
    elif len(target) >= 6 and target[4] == "_":
        # cameo targets
        servers.update(
            {
                "999": "BestSingleT",
                "19": "RoseTTAFold",
                "20": "SWISS-MODEL",
            }
        )
    else:
        logger.warning(f"{target} not in CASP15, CASP14 and CAMEO.")
    servers.update(
        {
            "886": "AF2NoMSA",
            "885": "AF2WithMSA",
            "887": "ESMFoldGitHub",
            "888": "SFM",
        }
    )
    return servers


def _collect_models(
    targets: Sequence[str],
    rootdir: str,
    top5: bool,
) -> Sequence[Tuple[str, str, int, str, str]]:
    # check prediction directory
    cameodir = Path(rootdir) / "cameo-official-targets.prediction"
    assert cameodir.exists(), f"{cameodir} does not exist."
    caspdir = Path(rootdir) / "casp-official-targets.prediction"
    assert caspdir.exists(), f"{caspdir} does not exist."
    caspdomdir = Path(rootdir) / "casp-official-trimmed-to-domains.prediction"
    assert caspdomdir.exists(), f"{caspdomdir} does not exist."
    # collect models
    models = []
    max_model_num = 5 if top5 else 1
    for t in tqdm(targets, desc="Collecting models"):
        for server in _get_servers(t):
            for idx in range(1, max_model_num + 1):
                if len(t) >= 8 and t[0] == "T" and t[-3:-1] == "-D":
                    # e.g. T1024-D1
                    native = caspdomdir / f"{t}.pdb"
                    model = caspdomdir / t / f"{t[:-3]}TS{server}_{idx}{t[-3:]}"
                elif len(t) >= 5 and t[0] == "T":
                    # e.g. T1024
                    native = caspdir / f"{t}.pdb"
                    model = caspdir / t / f"{t}TS{server}_{idx}"
                elif len(t) >= 6 and t[4] == "_":
                    # e.g. 1ctf_A
                    native = cameodir / t / "target.pdb"
                    prefix = cameodir / t / "servers" / f"server{server}"
                    model = prefix / f"model-{idx}" / f"model-{idx}.pdb"
                else:
                    logger.warning(f"{t} not in CASP15, CASP14 and CAMEO.")
                models.append((t, server, idx, str(model), str(native)))
    return models


@cli.command()
@click.option(
    "--target-list",
    type=click.Path(exists=True),
    help="Input list for targets.",
    required=True,
)
@click.option(
    "--prediction-root",
    type=click.Path(exists=True),
    help="Input directory for prediction results.",
    required=True,
)
@click.option(
    "--result-directory",
    type=click.Path(exists=True),
    help="Output directory for evaluation results.",
    required=True,
)
@click.option(
    "--num-workers", type=int, default=-1, help="Number of workers.", show_default=True
)
@click.option(
    "--criterion",
    type=str,
    default="TMscore",
    help=(
        "Evaluation metric 'TMscore', 'GDT_TS', 'GDT_HA' or 'LDDT'. "
        "If set to 'TMscore', the program TMscore will be used. "
        "If set to 'GDT_TS' or 'GDT_HA', program LGA will be used. "
        "If set to 'LDDT', the program lddt will be used."
    ),
    show_default=True,
)
@click.option(
    "--top5",
    is_flag=True,
    default=False,
    help="Whether to output top5 score.",
    show_default=True,
)
def evaluate(
    target_list: Path,
    prediction_root: Path,
    result_directory: Path,
    num_workers: int,
    criterion: str,
    top5: bool,
) -> None:
    # CASP/CAMEO benchmark metadata (external, lazily loaded)
    metadata4target = _load_metadata4target()

    # parse target list
    targets = []
    with open(target_list, "r") as fp:
        for line in fp:
            assert 5 < len(line) < 12, f"Invalid target name {line}."
            targets.append(line.rstrip("\n"))
    logger.info(f"Number of targets: {len(targets)}")

    # check criterion
    assert criterion in ("TMscore", "GDT_TS", "GDT_HA", "LDDT"), (
        f"Invalid --criterion parameter: {criterion}. Please use 'TMscore', "
        f"'GDT_TS', 'GDT_HA' or 'LDDT'."
    )

    # convert metadata information to dictionary
    groupdict, lengthdict, typedict = {}, {}, {}
    for target in targets:
        key = target
        if len(target) >= 8 and target[-3:-1] == "-D":
            # e.g. T1024-D1
            key = target[:-3]
        logger.info(f"{metadata4target[key]}")
        for domstr, domlen, domgroup in metadata4target[key]["domain"]:
            dom = domstr.split(":")[0]
            if dom == target or dom == f"{target}-D0":
                groupdict[target] = domgroup
                lengthdict[target] = domlen
                typedict[target] = metadata4target[key]["type"]
                break
        else:
            logger.error(f"{target} metadata information not found.")
    logger.info(f"Metadata information for targets: {len(groupdict)}")

    # collect models
    models = _collect_models(targets, prediction_root, top5)
    logger.info(f"{len(models)} models in {prediction_root}")
    for target, server, idx, model, native in models:
        logger.debug(f"{target} {server} {idx} {model} {native}")

    # collect scores
    def _score4pair(model_info: Tuple[str, str, int, str, str]):
        target, server, idx, model, native = model_info
        if criterion == "TMscore":
            s = TMscore4SinglePair(model, native)
        elif criterion == "GDT_TS" or criterion == "GDT_HA":
            s = LGA4SinglePair(model, native)
        elif criterion == "LDDT":
            s = lddt4SinglePair(model, native)
        else:
            raise ValueError(
                f"Invalid criterion {criterion}. Use 'TMscore' "
                f"'GDT_TS', 'GDT_HA' or 'LDDT'."
            )
        s["Target"], s["Server"], s["ModelIndex"] = target, server, idx
        return s

    scores = Parallel(n_jobs=num_workers)(delayed(_score4pair)(_) for _ in tqdm(models))
    df = pd.DataFrame(scores)
    print(df)

    # analysis scores
    newscores = {}
    for target, gdf in df.groupby("Target", sort=True):
        servers = _get_servers(target)
        score_dict = OrderedDict()
        score_dict["Group"] = groupdict.get(target, "NA")
        score_dict["Length"] = lengthdict.get(target, -1)
        score_dict["Type"] = typedict.get(target, "NA")
        for server, server_name in servers.items():
            subdf = gdf[gdf["Server"] == server]
            k1 = f"{server_name}_{criterion}_Top1"
            score_dict[k1] = subdf[subdf["ModelIndex"] == 1][criterion].max()
            if top5:
                k5 = f"{server_name}_{criterion}_Top5"
                score_dict[k5] = subdf[criterion].max()
        newscores[target] = score_dict
    df = pd.DataFrame.from_dict(newscores, orient="index")
    print(df)

    # output results
    result_directory = Path(result_directory)
    outfile = result_directory / f"{target_list}_{criterion}.csv"
    df.to_csv(outfile)
    logger.info(f"Write evaluation results to {outfile}")

    # simple analysis for results
    CATEGORY = {
        "CAMEO  Easy": ["Easy"],
        "CAMEO  Medi": ["Medium", "Hard"],
        "CASP14 Full": ["MultiDom"],
        "CASP14 Easy": ["TBM-easy", "TBM-hard"],
        "CASP14 Hard": ["FM/TBM", "FM"],
        "CASP15 Full": ["MultiDom"],
        "CASP15 Easy": ["TBM-easy", "TBM-hard"],
        "CASP15 Hard": ["FM/TBM", "FM"],
    }
    dfs = []
    for catandgroup, groups in CATEGORY.items():
        _subdf = df[df["Type"] == catandgroup.split()[0]]
        dfsub = pd.concat(
            [_subdf[_subdf["Group"] == g] for g in groups],
            ignore_index=True,
        )
        dfsub.drop(columns=["Type", "Group", "Length"], axis=1, inplace=True)
        tmpdf = dfsub.mean()
        tmpdf["Number"] = len(dfsub)
        tmpdf["CatAndGroup"] = catandgroup
        dfs.append(tmpdf)
    meandf = pd.concat(dfs, axis=1).T.set_index("CatAndGroup")
    print(f"{'CatAndGroup':<11} {'Number':>6}", end=" ")
    print(" ".join([f"{_[:7]:7s}" for _ in meandf.columns[:-1]]))
    for category, row in meandf.iterrows():
        print(f"{category:<11s} {int(row['Number']):>6d}", end=" "),
        print(" ".join([f"{_*100:>7.2f}" for _ in row.values[:-1]]))


if __name__ == "__main__":
    cli()
