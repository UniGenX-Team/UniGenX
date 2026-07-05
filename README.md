# UniGenX

**Unified autoregressive generation of 3D atomic structures** — crystals,
molecules, proteins, and protein-ligand complexes — with a single Llama decoder
that jointly models discrete tokens and continuous 3D coordinates.

UniGenX represents a structure as one sequence that interleaves *discrete*
tokens (element / residue symbols and special tokens) with *continuous*
coordinate slots. At each step the standard LM head predicts the next token,
while a **diffusion head** samples the (x, y, z) coordinate at every position
flagged by a coordinate mask. One forward pass therefore produces both the
sequence and its geometry.

This repository ships the **inference, evaluation, and single-node training**
code. `unigenx_infer.py` is the generation entry point and `unigenx_train.py`
is its training mirror; the released checkpoints reproduce the paper numbers.

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Pretrained checkpoints](#pretrained-checkpoints)
- [Generation](#generation)
- [Evaluation](#evaluation)
- [Training](#training)
- [Data formats](#data-formats)
- [Reproducing paper results](#reproducing-paper-results)
- [Development](#development)
- [Citation](#citation)
- [License and notes](#license-and-notes)

## Overview

The central mechanism is a **joint discrete-token + diffusion-coordinate**
model. A `coordinates_mask` (0/1 per position) marks which slots are
coordinates; at those positions a diffusion module (`DiffLoss`,
`target_channels=3`) replaces the token head and samples the point in space,
conditioned on the transformer hidden state. Because the whole structure lives
in one autoregressive sequence, the same architecture covers every domain — only
the tokenizer dictionary and the per-target sequence layout change.

**Supported domains** (selected with `--target`):

| Domain | `--target` | What is generated |
|---|---|---|
| Crystals | `material`, `uni_mat` | Lattice (first 3 coord slots) + fractional atom coordinates |
| Molecules | `mol`, `uni_mol` | 3D conformers |
| Conditional crystals | `cond_mat` | Property-conditioned crystals |
| Conditional molecules | `cond_mol` | Property-conditioned molecules (SMILES then coordinates) |
| Proteins | `prot` | Cα backbone conformations |
| Protein-ligand docking | `dock`, `misato` | Ligand poses / holo pocket + pose |
| Enzyme / EC design | `ecnum` | EC-number-conditioned enzyme sequences (no coordinates) |

**What is included:** the generation entry point, the domain evaluators under
`eval/`, and single-node training (`unigenx_train.py`, DeepSpeed / DDP).

**What is not included:** multi-node, pipeline-parallel, and `nnscaler`
back-ends; the dynamic token-bucket loader; the data-preparation pipelines that
build the on-disk formats; and turnkey full-paper reproduction (the released
checkpoints are downloaded separately and the public datasets are prepared by
the user — see [`REPRODUCE.md`](REPRODUCE.md)).

## Installation

```bash
conda env create -f environment.yaml   # creates the `unigenx` env (Python 3.11.9, PyTorch 2.3.0, CUDA 12.1)
conda activate unigenx
pip install -r requirements.txt         # PyG C++ wheels + h5py + periodictable
```

`environment.yaml` pins Python 3.11.9, PyTorch 2.3.0, CUDA 12.1, RDKit
2024.03.1, transformers 4.40.1. `requirements.txt` installs the PyG extension
wheels (`pyg-lib`, `torch-scatter`, `torch-sparse`, `torch-cluster`) from the
PyG wheel index.

> **The PyG wheel index URL must match your installed PyTorch + CUDA build.**
> The pinned one is `torch-2.3.0+cu121`. A mismatch here is the most common
> setup failure.

`pymatgen` and `smact` (used by inference and material evaluation) come from the
conda environment; if you use a different environment, install them separately.

Some evaluators pull in heavier third-party packages that are **imported
lazily** — install them only for the evaluation you actually run:

- **materials:** `smact`, `pymatgen`, `ase`, `chgnet` (relaxation), a
  force-field potential such as `mattersim` (bulk modulus).
- **molecules:** `psi4`, `psikit` (property MAE).
- **proteins:** `deeptime`, `mdtraj` (TICA / MD), `tmtools`, a `US-align` /
  `USalign` binary, `TMscore`, `LGA` (structure scoring), `PyMOL` (RMSD +
  images).
- **docking:** `spyrmsd` (only for the optional `--symmetry` RMSD); `rdkit` is
  already in the base environment.

Protein-ligand **docking / MISATO** additionally needs `h5py`, `periodictable`,
`lmdb`, and `rdkit` (the docking / MD I/O path); `h5py` and `periodictable` are
in `requirements.txt` but imported lazily.

## Quick start

An end-to-end small-molecule (GEOM-QM9) generation run:

1. Download the `mol_qm9` checkpoint and prepare a QM9 input `.jsonl` (see
   [Data formats](#data-formats)).
2. Edit the top of `scripts/gen_qm9.sh` and set the two blank variables:

   ```bash
   CKPT=/path/to/mol_qm9.pt      # the downloaded checkpoint
   INPUT=/path/to/qm9_input.jsonl
   ```

3. Run it:

   ```bash
   bash scripts/gen_qm9.sh
   ```

The script wraps `python unigenx_infer.py` (with `--target mol`,
`--dict_path unigenx/data/dict_qm9.txt`, and the DPM-Solver flags) and writes
`<ckpt>_<input>.jsonl` next to the checkpoint — one record per input with a
`"prediction"` field added, holding the sampled coordinates (and, where
relevant, the sampled sequence / lattice).

4. Score the output:

   ```bash
   python eval/molecule/evaluate_mol.py --input <ckpt>_<input>.jsonl --threshold 0.5 --output metrics.txt
   ```

## Pretrained checkpoints

Checkpoints are **not** shipped in this repository; download them separately. A
checkpoint and its dictionary are a **matched pair** — see
[Generation](#generation) and [`REPRODUCE.md`](REPRODUCE.md) for the exact
pairing. The families, by task:

| Task | Checkpoint family |
|---|---|
| Unified crystal + molecule | `unified_{carbon24,mp20,mpts52,qm9}`; `unified_pretrain` (backbone) |
| Crystal structure prediction | `csp_{carbon24,mp20,mpts52}`; `csp_pretrain` (backbone) |
| Material multi-property design | `mc_mat`, `bs_mat_1..6`, `ms_mat_1..6` |
| Molecular conformers | `mol_qm9`, `mol_drugs` |
| Conditional molecules | `c_mol` |
| Proteins (MD + structure prediction) | `1_m_p`..`12_m_p`, `b_p`, `e_bs`, `e_wo_bs` |
| Protein-ligand docking | `pld`, `pld_u` |
| Enzyme / EC design | `e`, `e_wo` |

**Pre-training backbones.** `csp_pretrain` and `unified_pretrain` are
fine-tuning starting points, not per-benchmark evaluation checkpoints, and
neither reproduces a specific paper table row on its own. `unified_pretrain`
carries saved `args` and loads through `unigenx_infer.py`; **`csp_pretrain` has
no saved `args` and cannot be driven through `unigenx_infer.py` as-is** — train
/ fine-tune from it first.

## Generation

Every task has a launcher under `scripts/`. Each is a thin wrapper around
`python unigenx_infer.py`; edit the top of the script to set `CKPT=` (path to
the `.pt`) and `INPUT=` (path to the input data), then run it. They ship with
`CKPT=`/`INPUT=` intentionally blank.

Output: each script writes `<ckpt>_<input>.jsonl` next to the checkpoint, one
record per input with a `"prediction"` field added (predicted coordinates and,
where relevant, the sampled sequence / lattice).

| Script | `--target` | Dictionary (vocab) | Checkpoints | Notes |
|---|---|---|---|---|
| `gen_uni.sh` | `uni_mat` (default) / `uni_mol` | `dict_uni.txt` (193) | `unified_{carbon24,mp20,mpts52,qm9}`, `unified_pretrain` | Unified crystal+molecule model. Switch to `--target uni_mol` (drop `--no_space_group`) for conformers. |
| `gen_mat.sh` | `material` | `dict_mat.txt` (355) | `csp_{carbon24,mp20,mpts52}`, `csp_pretrain` | Crystal structure prediction. First 3 coordinate slots are the lattice vectors. |
| `gen_cond_mat.sh` | `cond_mat` | `dict_cond_mat.txt` (355) | `mc_mat`, `bs_mat_{1..6}`, `ms_mat_{1..6}` | Multi-property material design (uses `--top_p 0.8 --temperature 1.0`). **Must** use `dict_cond_mat.txt`; `dict_mat.txt` maps the property markers to `<unk>` and silently breaks conditioning (both dicts are vocab 355). |
| `gen_qm9.sh` | `mol` | `dict_qm9.txt` (27) | `mol_qm9` | Small-molecule conformers. Uses DPM-Solver (`--is_solver --solver_order 2`). |
| `gen_drugs.sh` | `mol` | `dict_drugs.txt` (38) | `mol_drugs` | Drug-like conformers (reuses the `mol` path). Small batch size (16). |
| `gen_cond_mol.sh` | `cond_mol` | `dict_cond_mol.txt` (34) | `c_mol` | Property-conditional molecules (1-6 joint properties). Two-phase: sample SMILES conditioned on the property prefix (RDKit-validity filtered), then sample coordinates. |
| `gen_prot.sh` | `prot` | `dict_prot.txt` (28) | `1_m_p..12_m_p`, `b_p`, `e_bs`, `e_wo_bs` | Protein-backbone (Cα) conformations; `--infer_batch_size 1`; sliding window for long sequences. Input records carry `seq`/`aa`. |
| `gen_dock.sh` | `misato` / `dock` | `dict_dock.txt` (126) | `pld`, `pld_u` | Protein-ligand docking; set `TARGET=` in the script. `misato` = apo pocket + ligand SMILES -> holo pocket + ligand pose (paper docking numbers); `dock` = pocket coords -> ligand pose. "Pocket Given" vs "Not Given" are two input datasets, not a flag. |
| `gen_enzyme.sh` | `ecnum` | `dict_ecnum.txt` (64) | `e`, `e_wo` | EC-number-conditioned enzyme sequence design (uses `--top_p 0.95 --temperature 1.0`). Input records carry `EC_number` (e.g. `"1.1.1.1"`). |

### Sampler options (all scripts)

- `--diff_steps N` sets the DDPM coordinate-sampler step count (scripts use 200).
- `--is_solver --solver_order 2` switches to the faster DPM-Solver (see
  `gen_qm9.sh`).
- `--top_p` / `--temperature` control the discrete-token sampling (used by the
  conditional scripts).
- Model architecture is rebuilt from the checkpoint's saved `args`; only
  `diff_steps` / `target` / `is_solver` / `solver_order` / `solver_type` are
  overridable from the command line — changing arch flags there has no effect.

## Evaluation

Run the matching evaluator on the generated `.jsonl`. Metric conventions (so the
numbers line up with the paper) are documented in [`REPRODUCE.md`](REPRODUCE.md).

**Materials**
```bash
# CSP match rate + validity (pymatgen StructureMatcher + smact). Positional input.
python eval/material/evaluate_csp.py <gen.jsonl> [--multiple True] [--output metrics.txt]
# Bulk modulus via ASE equation-of-state (Murnaghan) over strained cells.
python eval/material/compute_bulk.py <cif_dir> [--hist_out bulk.pdf]
# Relax generated crystals with CHGNet (single- or multi-process).
python eval/material/relax.py --input <gen.jsonl> --output <relaxed.jsonl> [--steps 500]
python eval/material/relax_multiprocess.py <gen.jsonl> --output <relaxed.jsonl> [--num_workers 40]
```

**Molecules**
```bash
# Conformer COV/MAT (RDKit GetBestRMS). Use --threshold 0.5 (QM9) or 1.25 (Drugs).
python eval/molecule/evaluate_mol.py --input <gen.jsonl> --threshold 0.5 --output metrics.txt
# Property-conditional 3D front-end: rebuilds + MMFF-optimizes molecules, groups
# by conditioning property, then feeds the property-MAE calculator.
python eval/molecule/evaluate_cond.py --input <cond_mol_gen.jsonl>
# evaluate_mol_prop.py is a library module (property MAE via psi4/psikit),
# consumed by evaluate_cond.py -- not a standalone CLI.
```

**Proteins**
```bash
# Structure prediction vs native: TM-score (US-align binary) + RMSD (PyMOL).
python eval/protein/evaluate_afdb.py --input_file <gen.jsonl> --output_dir <out/> [--usalign_bin USalign]
# TICA free-energy-surface deviation for MD conformational dynamics.
python eval/protein/tica_eval.py --train_lmdb <ref> --test_lmdb <gen> --out_png fes.png [--lagtime 10]
```
`eval/protein/protein_evaluation/` is a TM-score / LDDT / LGA / GDT toolkit for
the CASP14+15 / CAMEO benchmark (`cameo-subset-casp14-and-casp15-combine.list`,
474 targets); its scripts use flat sibling imports, so run them from inside that
directory.

**Docking**
```bash
# Best-of-N coordinate RMSD (paper metric). Groups records into per-target windows.
python eval/docking/evaluate_docking.py --input <gen.jsonl> --samples_per_target 100 [--symmetry] [--output metrics.txt]
```

## Training

`unigenx_train.py` is the single-node training entry point and a mirror of
`unigenx_infer.py`: it bootstraps the same `UniGenXConfig` from the CLI, so the
checkpoint it saves loads straight back into `unigenx_infer.py`. The saved
checkpoint stores its full `args`, from which inference rebuilds the exact
architecture before loading weights. Training uses **single-node DeepSpeed / DDP**.

### Entry point and key arguments

```bash
torchrun --nproc_per_node=<num_gpus> unigenx_train.py \
    --strategy Zero1 --target material \
    --dict_path unigenx/data/dict_mat.txt \
    --train_data_path <train.jsonl> [--valid_data_path <valid.jsonl>] \
    --save_dir <out_dir> \
    --max_lr 1e-4 --total_num_steps 100000 --warmup_num_steps 1000
```

| Argument | Meaning |
|---|---|
| `--target` | Domain / data-path selector: `material`, `mol`, `prot`, `cond_mat`, `cond_mol`, `uni_mat`, `uni_mol`, `dock`, `misato`, `ecnum`. Same set as inference. |
| `--train_data_path` / `--valid_data_path` | Training (and optional validation) data. For `uni_mat`/`uni_mol` this may be a comma-joined `"material_path,mol_path"` pair (see below). |
| `--dict_path` | Per-target tokenizer dictionary — **the same dict used at inference** for that target. The vocab iron rule (`non-empty lines + 7`) is asserted against the model embedding at startup. |
| `--save_dir` | Output directory; DeepSpeed writes `global_step<N>/mp_rank_00_model_states.pt` checkpoints, each carrying `["args"]`. |
| `--strategy` | `Single` / `DDP` / `Zero0`-`Zero3`. Default `Zero1`. |
| `--max_lr` / `--total_num_steps` / `--warmup_num_steps` | Peak LR, total optimizer steps, and warmup steps for the warmup+decay schedule. |

Other launcher knobs: `--train_batch_size`, `--gradient_accumulation_steps`,
`--save_batch_interval`, `--log_interval`, `--diff_steps`.

### Launcher

`scripts/train_diff.sh` is a single-node `torchrun` example (fill in
`TRAIN_DATA` / `SAVE_DIR`, pick `--target` + matching `--dict_path`). Scale
across GPUs on one node with `--nproc_per_node=<num_gpus>`.

### Loading a trained checkpoint back into inference

The saved checkpoint carries `["args"]` (the full config); `unigenx_infer.py`
rebuilds the architecture from it and loads the weights from the `"module"` key.

- **ZeRO stage 1/2:** the `mp_rank_00_model_states.pt` under `global_step<N>/`
  loads directly.
- **ZeRO stage 3:** the weights are sharded — run DeepSpeed's `zero_to_fp32.py`
  on the checkpoint to consolidate fp32 weights (into the `"module"` key)
  **before** pointing `unigenx_infer.py` at it.

### Unified training (`uni_mat` / `uni_mol`)

For the unified crystal+molecule model, `--train_data_path` accepts a
comma-joined `"material_path,mol_path"` pair; the two sub-datasets are built with
their own per-domain tokenization and interleaved for mixed training. A single
(non-comma) path trains on that one domain.

### Known limitations

- **Single-node only.** Multi-node, pipeline-parallel, and `nnscaler` back-ends
  are **not** shipped.
- **Data preparation is left to the user.** The preprocessing pipelines that
  build the on-disk formats below are not included.
- **The dynamic token-bucket loader is not shipped** (`--dynamic_loader` is
  intentionally omitted — it needs the Cython bucket loader that is not part of
  the released single-node training code).
- **`misato` / `dock` training** additionally needs `h5py`, `periodictable`,
  `lmdb`, and `rdkit` (the docking / MD I/O path).
- **`ecnum` is sequence-only** (no coordinates), so it trains with no coordinate
  diffusion loss.

## Data formats

Per-target on-disk schema for the **training** readers (`get_train_item_*`);
the inference readers accept the same records. Prepare these yourself — the
preprocessing pipelines are not shipped.

| `--target` | Input | Record fields |
|---|---|---|
| `material`, `uni_mat` | jsonl or `.pkl` list | `id`, `formula`, `lattice` (3×3), `sites` = [{`element`, `fractional_coordinates`}]. jsonl records are normalized + site-sorted; `.pkl` records are taken as-is; space group is not consumed. |
| `mol`, `uni_mol` | LMDB (pickle), or jsonl / `.pkl` | `id`, `smi`, `pos` = [[x, y, z] per atom]. `num` (atom count) is only used for jsonl length filtering. |
| `prot` | LMDB (zlib+pickle), or `.pkl` / jsonl | `aa` (sequence), `pos` = [[x, y, z] per residue Cα]. Coordinates are mean-centered; no ESM embeddings. |
| `cond_mat` | jsonl / `.pkl` | material fields + `property` = **dict** (keys matched by `dft_(.*?)_`, values log-transformed). **Inference instead uses flat `prop` (str) + `prop_val` (scalar)** — the keys differ, so prepare each for its own stage. |
| `cond_mol` | LMDB, or jsonl / `.pkl` | mol fields + `prop` = **list**[property-name str] and `prop_val` = **list**[scalar] (parallel lists, one entry per conditioned property). |
| `dock` | Directory of LMDB sub-databases | whichever of `pockets/{split}.lmdb`, `ligands/{split}.lmdb`, `crossdocked...final.lmdb`, `protein_ligand_binding_pose_prediction/{split}.lmdb` exist; split maps TRAIN→train / VAL→valid / INFER→test. |
| `misato` | Directory: `{split}_mols.pkl` + `MD_pockets.hdf5` | pkl: `pdb_id -> {smi, mol}`; hdf5 per `pdb_id`: `molecules_begin_atom_index`, `atoms_number`, `trajectory_coordinates` (holo), `apo_pocket_coordinates`. |
| `ecnum` | jsonl / `.pkl` (train; LMDB for infer) | `EC_number` (dot-separated, first 3 levels) and `seq` or `aa`. Sequence-only, no coordinates. |

## Reproducing paper results

[`REPRODUCE.md`](REPRODUCE.md) maps every paper result to the exact recipe:
generation script, checkpoint(s), the paired tokenizer dictionary, the public
dataset, the evaluation script, and the expected headline number. It also
documents the dictionary invariant and the metric conventions.

## Development

```bash
pre-commit install
pre-commit run --all-files   # black, isort (black profile), ruff, whitespace/encoding/JSON checks
python -m pytest tests/ -q   # import / config round-trip / dict-vocab / collation / dry-run smoke tests
```

The test suite runs without any checkpoint or private data present; tests that
need a real checkpoint or a GPU skip automatically. See [`CLAUDE.md`](CLAUDE.md)
for architecture / developer notes.

## Citation

If you use UniGenX, please cite the paper. **This BibTeX entry is a placeholder —
fill in the authors, venue/arXiv identifier, and year before use.**

```bibtex
@article{unigenx,
  title   = {UniGenX: <fill in the full paper title>},
  author  = {<fill in authors>},
  journal = {<fill in venue / arXiv id>},
  year    = {<fill in year>}
}
```

## License and notes

Released under the MIT License (Copyright (c) Microsoft Corporation); see
[`LICENSE`](LICENSE). Some evaluators wrap external programs that you install
separately — for example `eval/protein/protein_evaluation/` drives the external
LGA / TM-score / US-align binaries used for CASP / CAMEO scoring (see
[Installation](#installation) for the per-domain optional dependencies).

Additional notes:

- **Model architecture flags are read from the checkpoint**, not the CLI (see
  the sampler note under [Generation](#generation)).
- **Protein-MD checkpoints are the baseline sequence-token architecture** and do
  **not** consume ESM-2 embeddings.
- **Using the wrong dictionary can silently generate garbage** even when the
  vocab sizes coincide, so the generation scripts hard-code the correct
  `--dict_path`; do not change it.
</content>
</invoke>
