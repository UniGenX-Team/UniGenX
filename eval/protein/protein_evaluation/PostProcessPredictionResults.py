#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-process raw predictions into the official CASP/CAMEO submission layout.

Copies predicted PDB files into the per-target directory structure expected by
the CASP/CAMEO evaluation servers (renaming to ``<target>TS<server>_<idx>`` for
CASP, ``server<server>/model-<idx>/model-<idx>.pdb`` for CAMEO), rewrites the
chain id, and -- for CASP targets -- trims each model to its per-domain residue
segments.

External prerequisite (not shipped): ``metadata4target`` -- the CASP/CAMEO
domain-definition metadata (an external benchmark table), imported lazily inside
``__main__``.
"""
import logging
import os
import sys
from pathlib import Path

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_name_chain_from_target(target: str):
    # parse target name and chain
    if len(target) == 6 and target[4] == "_":
        # e.g. 1ctf_A
        name, chain = target[:4], target[5]
    elif (len(target) == 5 or len(target) == 7) and target[0] == "T":
        # e.g. T1024 or T1106s2
        name, chain = target, " "
    else:
        # test or other names
        logger.warning(f"{target} may be a wrong name.")
        name, chain = target, " "
    return name, chain


def find_source_pdb(target: str, srcdir: str, model_num: int):
    srcdir = Path(srcdir)
    # alphafold2 results
    srcpdb = srcdir / target / f"ranked_{model_num-1}.pdb"
    if srcpdb.exists():
        return srcpdb
    # other custom results
    srcpdb = srcdir / f"{target}-{model_num}.pdb"
    if srcpdb.exists():
        return srcpdb
    return None


def get_destnation_pdb(target: str, dstdir: str, server: int, num: int):
    dstdir = Path(dstdir)
    if target[0] == "T":
        # casp target
        outdir = dstdir / target
        dstpdb = outdir / f"{target}TS{server}_{num}"
    else:
        # cameo target
        outdir = dstdir / target / "servers" / f"server{server}"
        dstpdb = outdir / f"model-{num}" / f"model-{num}.pdb"
    return dstpdb


def copy_and_modify(srcpdb: str, dstpdb: str, chain: str):
    lines = []
    with open(srcpdb, "r") as fp:
        lines = fp.readlines()

    for i, line in enumerate(lines):
        if len(line) != 81:
            continue
        if line.startswith("ATOM  ") or line.startswith("HETATM"):
            lines[i] = line[:21] + chain + line[22:]

    os.makedirs(dstpdb.parent, exist_ok=True)
    with open(dstpdb, "w") as fp:
        fp.writelines(lines)


def recover_domseg_from_residx(residx: set):
    domseg = " "
    for i in sorted(residx):
        if i - 1 in residx and i + 1 in residx:
            continue
        elif i - 1 in residx:
            domseg += f"{i},"
        elif i + 1 in residx:
            domseg += f"{i}-"
        else:
            return ""
    return domseg.rstrip(",")


def select_residues_by_index(atomlines: list, residx: set) -> list:
    lines = []
    for line in atomlines:
        if line.startswith("ATOM"):
            resnum = int(line[22:26].strip())
            if resnum in residx:
                lines.append(line)
    lines.append("TER\n")
    lines.append("END\n")
    return lines


if __name__ == "__main__":
    # CASP/CAMEO benchmark metadata (external; imported lazily). Provide a
    # `metadata` module exposing `metadata4target` (built from the official
    # CASP/CAMEO definition tables) to run the post-processing.
    from metadata import metadata4target

    if len(sys.argv) != 4 and len(sys.argv) != 5:
        sys.exit(
            f"Usage: {sys.argv[0]} <prediction_root_directory> <results_directory> <server_id> [--top5]"
        )
    rootdir, resdir, serverid = sys.argv[1:4]
    max_model_num = 5 if len(sys.argv) == 5 and sys.argv[4] == "--top5" else 1

    # check prediction directory
    cameodir = Path(rootdir) / "cameo-official-targets.prediction"
    assert cameodir.exists(), f"{cameodir} does not exist."
    caspdir = Path(rootdir) / "casp-official-targets.prediction"
    assert caspdir.exists(), f"{caspdir} does not exist."
    caspdomdir = Path(rootdir) / "casp-official-trimmed-to-domains.prediction"
    assert caspdomdir.exists(), f"{caspdomdir} does not exist."

    # get target names
    targets = [_.name for _ in caspdir.iterdir() if _.is_dir()]
    targets += [_.name for _ in cameodir.iterdir() if _.is_dir()]
    logger.info(f"Number of targets: {len(targets)}")

    # parse metadatarmation
    logger.info(f"Loading metadata for targets: {len(metadata4target)}")
    for tarname, metadata in metadata4target.items():
        logger.debug(tarname, metadata)

    # processing target prediction one by one
    for target in sorted(targets):
        tarname, chain = parse_name_chain_from_target(target)
        logger.info(f"The {target} name is {tarname} and chain is '{chain}'.")
        if target not in metadata4target:
            logger.error(f"{target} does not defined in metadata.")
            continue

        # process predicted model one by one
        for model_num in range(1, max_model_num + 1):
            # copy prediction result and modify chain id
            srcdir = Path(resdir)
            srcpdb = find_source_pdb(target, srcdir, model_num)
            if not srcpdb:
                logging.error(f"{target}-{model_num} not in {srcdir}.")
                continue
            logger.debug(f"Predicted result {srcpdb}")
            dstdir = caspdir if target[0] == "T" else cameodir
            dstpdb = get_destnation_pdb(target, dstdir, serverid, model_num)
            logger.debug(f"Destination pdb {dstpdb}")
            logging.debug(f"cp {srcpdb} {dstpdb}")
            copy_and_modify(srcpdb, dstpdb, chain)

            # cameo target does not have domain definition
            if target[0] != "T":
                continue

            # process domain for model one by one
            for domstr, domlen, domgroup in metadata4target[target]["domain"]:
                # select residue index by domain definition
                try:
                    residx = set()
                    cols = domstr.split(":")
                    assert 2 == len(cols), f"Wrong domain format {domstr}"
                    domname, domseg = cols[0], cols[1]
                    for seg in domseg.split(","):
                        start, finish = [int(_) for _ in seg.split("-")]
                        residx.update(range(start, finish + 1))
                    assert domlen == len(residx), f"domain length!={domlen}"
                    # check residue index by useless
                    tmps = recover_domseg_from_residx(residx)
                    assert tmps == domseg, f"Wrong domseg '{tmps}' '{domseg}'"
                except Exception as e:
                    logging.error(f"Domain {domstr} parsing error, {e}")
                    continue
                logger.debug(f"{domname:10} {len(residx):4} {domseg}")
                # process native pdb
                pdbfile = caspdir / f"{tarname}.pdb"
                dompdbfile = caspdomdir / f"{domname}.pdb"
                if pdbfile.exists() and not dompdbfile.exists():
                    try:
                        with open(pdbfile, "r") as fp:
                            lines = fp.readlines()
                        domlines = select_residues_by_index(lines, residx)
                        with open(dompdbfile, "w") as fp:
                            fp.writelines(domlines)
                    except Exception as e:
                        logging.error(f"Wrong native pdb {dompdbfile}, {e}")
                # process prediction pdb for domain
                domdir = caspdomdir / domname
                if not domdir.exists():
                    continue
                dstdom = domdir / f"{dstpdb.name}{domname[-3:]}"
                logger.debug(f"{dstpdb} {dstdom}")
                with open(dstpdb, "r") as fp:
                    lines = fp.readlines()
                dstlines = select_residues_by_index(lines, residx)
                with open(dstdom, "w") as fp:
                    fp.writelines(dstlines)
