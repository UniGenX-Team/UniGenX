# Reproducing the UniGenX paper results

This table maps every paper result to the exact recipe that reproduces it: the
generation script, the checkpoint(s) it consumes, the tokenizer dictionary that
**must** be paired with those checkpoints, the (public) dataset, the evaluation
script, and the expected headline number.

The released checkpoints are **not** shipped in this repository. Download them
separately, then for each run edit the top of the generation script and set
`CKPT=` (path to the `.pt`) and `INPUT=` (path to the input data), and run it.
Each script writes a `<ckpt>_<input>.jsonl` next to the checkpoint with a
`"prediction"` field per record; feed that jsonl to the matching eval script.

## Dictionary invariant (read this first)

A checkpoint and its dictionary are a matched pair. The tokenizer builds its
vocabulary as **`vocab = non-empty dict lines + 7`** (the 7 special tokens
`<pad> <bos> <eos> <unk>` + `<mask> <coord> <sg>`). If you load a checkpoint
with the wrong dictionary the vocab sizes still *happen* to match for some pairs
(e.g. `dict_mat` and `dict_cond_mat` are both 355) but the token meanings differ
and generation **silently produces garbage**. Always use the dictionary listed
below for a given checkpoint. The scripts already hard-code the correct
`--dict_path`.

## Result-by-result recipe

| Paper result | `--target` | Generation script | Checkpoint(s) | Dictionary (vocab) | Dataset | Evaluation | Expected headline |
|---|---|---|---|---|---|---|---|
| Unified joint modeling (crystals + molecules, one model) | `uni_mat` / `uni_mol` | `scripts/gen_uni.sh` | `unified_carbon24`, `unified_mp20`, `unified_mpts52`, `unified_qm9` (`unified_pretrain` = backbone) | `dict_uni.txt` (193) | Carbon-24 / MP-20 / MPTS-52 / GEOM-QM9 | `eval/material/evaluate_csp.py`, `eval/molecule/evaluate_mol.py` | Single model matches the per-domain specialists (words->words, numbers->numbers) |
| Crystal structure prediction (CSP) | `material` | `scripts/gen_mat.sh` | `csp_carbon24`, `csp_mp20`, `csp_mpts52` (`csp_pretrain` = backbone, see below) | `dict_mat.txt` (355) | Carbon-24, MP-20, MPTS-52 | `eval/material/evaluate_csp.py` | **MP-20 match rate 67.01%**; Carbon-24 +28%; MPTS-52 +120%; validity > 95% |
| Material multi-property design | `cond_mat` | `scripts/gen_cond_mat.sh` | `mc_mat`, `bs_mat_1..6`, `ms_mat_1..6` | `dict_cond_mat.txt` (355) | Materials Project (<= 16 atoms) | `eval/material/compute_bulk.py`, `eval/material/relax.py` | Low density + high Cv + high E_Hill; **55% SUN**; compared against MatterGen |
| Molecular conformer ensemble (QM9) | `mol` | `scripts/gen_qm9.sh` | `mol_qm9` | `dict_qm9.txt` (27) | GEOM-QM9 (DMCG splits) | `eval/molecule/evaluate_mol.py` (`--threshold 0.5`) | COV/MAT (recall + precision) at **0.5 Angstrom** |
| Molecular conformer ensemble (Drugs) | `mol` | `scripts/gen_drugs.sh` | `mol_drugs` | `dict_drugs.txt` (38) | GEOM-Drugs (DMCG splits) | `eval/molecule/evaluate_mol.py` (`--threshold 1.25`) | COV/MAT (recall + precision) at **1.25 Angstrom** |
| Molecule property-conditional generation (6 properties) | `cond_mol` | `scripts/gen_cond_mol.sh` | `c_mol` | `dict_cond_mol.txt` (34) | QM9 properties (alpha/gap/homo/lumo/mu/Cv) | `eval/molecule/evaluate_cond.py` -> `eval/molecule/evaluate_mol_prop.py` | LDM conditioning; **alpha +260%**; six-constraint joint 46% / 8% / 5% |
| Protein conformational dynamics (MD) | `prot` | `scripts/gen_prot.sh` | `1_m_p..12_m_p`, `b_p`, `e_bs`, `e_wo_bs` | `dict_prot.txt` (28) | 12 fast-folding proteins (leave-one-out) | `eval/protein/tica_eval.py` | Average free-energy deviation **0.91 kcal/mol** (1FME 0.64 -> A3D 1.20) |
| Protein-ligand docking | `misato` / `dock` | `scripts/gen_dock.sh` | `pld`, `pld_u` | `dict_dock.txt` (126) | Protein-ligand complexes (Pocket Given / Pocket Not Given); MISATO for apo->holo | `eval/docking/evaluate_docking.py` (`--samples_per_target 100`) | **Pocket Given 24.84% < 2 Angstrom** (vs 3DMolFormer 18.47%); **Pocket Not Given 15.92%** (vs 0.64%) |
| Enzyme design (EC-number) | `ecnum` | `scripts/gen_enzyme.sh` | `e`, `e_wo` | `dict_ecnum.txt` (64) | AFDB | (sequence / structure analysis) | EC-number-guided enzyme (sequence + structure) design |
| Protein structure prediction / AlphaFold3 comparison | `prot` | `scripts/gen_prot.sh` | `1_m_p..12_m_p`, `b_p` | `dict_prot.txt` (28) | CASP14 + CASP15 + CAMEO (474 targets) | `eval/protein/evaluate_afdb.py`, `eval/protein/protein_evaluation/` | TM-score / LDDT / GDT_TS / RMSD vs native; generality / AF3 comparison |

## Training

Training uses `unigenx_train.py` (single-node DeepSpeed), a mirror of
`unigenx_infer.py`. The example launcher `scripts/train_diff.sh` runs one target;
switch domains by changing `--target` and the matching `--dict_path` (the dict is
the **same one used at inference** for that target). The saved checkpoint carries
its full `["args"]`, so it loads straight back into `unigenx_infer.py` (for ZeRO
stage 3, run DeepSpeed's `zero_to_fp32.py` first). The preprocessing that
produces the on-disk formats below is not shipped; prepare the data yourself.

One row per released target (data formats follow the training-side
`get_train_item_*` readers):

| `--target` | Train command | Checkpoint(s) | Dictionary (vocab) | Training-data on-disk format | Note |
|---|---|---|---|---|---|
| `material` | `train_diff.sh` with `--target material --dict_path unigenx/data/dict_mat.txt` | `csp_{carbon24,mp20,mpts52}`, `csp_pretrain` | `dict_mat.txt` (355) | jsonl (or `.pkl` list): `id`, `formula`, `lattice` (3x3), `sites`=[{`element`, `fractional_coordinates`}] | jsonl records are normalized + site-sorted; `.pkl` records are taken as-is. Space group is not consumed. |
| `mol` | `train_diff.sh` with `--target mol --dict_path unigenx/data/dict_qm9.txt` (or `dict_drugs.txt`) | `mol_qm9`, `mol_drugs` | `dict_qm9.txt` (27) / `dict_drugs.txt` (38) | LMDB (pickle), or jsonl/`.pkl`: `id`, `smi`, `pos`=[[x,y,z] per atom] | `num` (atom count) is only used for jsonl length filtering. |
| `prot` | `train_diff.sh` with `--target prot --dict_path unigenx/data/dict_prot.txt` | `1_m_p`..`12_m_p`, `b_p`, `e_bs`, `e_wo_bs` | `dict_prot.txt` (28) | LMDB (zlib+pickle), or `.pkl`/jsonl: `aa` (sequence), `pos`=[[x,y,z] per residue Ca] | Coordinates are mean-centered. No ESM embeddings. |
| `cond_mat` | `train_diff.sh` with `--target cond_mat --dict_path unigenx/data/dict_cond_mat.txt` | `mc_mat`, `bs_mat_1..6`, `ms_mat_1..6` | `dict_cond_mat.txt` (355) | jsonl/`.pkl`: material fields + `property` = **dict** (keys matched by `dft_(.*?)_`, values log-transformed) | Training uses the `property` dict; **inference uses flat `prop` (str) + `prop_val` (scalar)** -- the keys differ, so prepare each for its own stage. |
| `cond_mol` | `train_diff.sh` with `--target cond_mol --dict_path unigenx/data/dict_cond_mol.txt` | `c_mol` | `dict_cond_mol.txt` (34) | LMDB, or jsonl/`.pkl`: mol fields + `prop` = **list**[property-name str], `prop_val` = **list**[scalar] | Multiple properties per record are joined via the parallel `prop` / `prop_val` lists. |
| `uni_mat` | `train_diff.sh` with `--target uni_mat --dict_path unigenx/data/dict_uni.txt` | `unified_{carbon24,mp20,mpts52,qm9}`, `unified_pretrain` | `dict_uni.txt` (193) | jsonl/`.pkl`, same as `material` | For mixed crystal+molecule training pass `--train_data_path "material_path,mol_path"` (comma-joined). |
| `uni_mol` | `train_diff.sh` with `--target uni_mol --dict_path unigenx/data/dict_uni.txt` | `unified_{carbon24,mp20,mpts52,qm9}`, `unified_pretrain` | `dict_uni.txt` (193) | LMDB, or jsonl/`.pkl`, same as `mol` | Mixed training uses the same comma-joined `"material_path,mol_path"` path form. |
| `dock` | `train_diff.sh` with `--target dock --dict_path unigenx/data/dict_dock.txt` | `pld`, `pld_u` | `dict_dock.txt` (126) | Directory of LMDB sub-databases; whichever of `pockets/{split}.lmdb`, `ligands/{split}.lmdb`, `crossdocked...final.lmdb`, `protein_ligand_binding_pose_prediction/{split}.lmdb` exist | split maps TRAIN->train / VAL->valid / INFER->test. Needs `h5py`+`periodictable`+`lmdb`+`rdkit`. |
| `misato` | `train_diff.sh` with `--target misato --dict_path unigenx/data/dict_dock.txt` | `pld`, `pld_u` | `dict_dock.txt` (126) | Directory: `{split}_mols.pkl` + `MD_pockets.hdf5` | pkl: `pdb_id -> {smi, mol}`; hdf5/pdb_id: `molecules_begin_atom_index`, `atoms_number`, `trajectory_coordinates` (holo), `apo_pocket_coordinates`. Needs `h5py`+`periodictable`+`lmdb`+`rdkit`. |
| `ecnum` | `train_diff.sh` with `--target ecnum --dict_path unigenx/data/dict_ecnum.txt` | `e`, `e_wo` | `dict_ecnum.txt` (64) | jsonl/`.pkl` (train; LMDB for infer): `EC_number` (dot-separated, first 3 levels), `seq` or `aa` | Sequence-only, no coordinates -- trains with no coordinate diffusion loss. |

## Metric conventions (so numbers line up with the paper)

- **CSP match rate** (`eval/material/evaluate_csp.py`): pymatgen `StructureMatcher`
  + smact charge-neutrality/electronegativity validity. Pass `--multiple True`
  for the top-N match-rate reported in the paper.
- **Conformer COV/MAT** (`eval/molecule/evaluate_mol.py`): RDKit `GetBestRMS`
  recall + precision; the RMSD threshold is the domain value (`0.5` for QM9,
  `1.25` for Drugs) passed via `--threshold`.
- **Docking RMSD** (`eval/docking/evaluate_docking.py`): naive coordinate RMSD
  (no alignment; structures are already in the same frame/atom order), best-of-N
  over the `--samples_per_target` window, reported at thresholds `[2, 4, 6, 8, 10]`
  Angstrom. This matches the published numbers. `--symmetry` switches to a
  symmetry-corrected RMSD (`spyrmsd`, optional) but is not the paper default.
- **MD free-energy surface** (`eval/protein/tica_eval.py`): TICA on Cα
  pairwise-distance features; the 2D TIC1/TIC2 density is the free-energy
  surface, summarised by its deviation from the reference trajectory.

## Backbone / pre-training checkpoints (ship, but not directly runnable)

- **`csp_pretrain`** and **`unified_pretrain`** are pre-training *backbones*, not
  per-benchmark evaluation checkpoints. `unified_pretrain` carries saved `args`
  and can be loaded by `unigenx_infer.py`; **`csp_pretrain` has no saved `args`**
  and therefore cannot be driven through `unigenx_infer.py` as-is. Use these as
  starting points for your own training / fine-tuning; they do **not** reproduce
  a specific paper table row on their own.
