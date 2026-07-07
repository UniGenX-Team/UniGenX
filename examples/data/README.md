# Example inference inputs

Tiny JSONL inputs for smoke-testing the public inference pipeline:

| File | Domain | Script | Checkpoint family |
|---|---|---|---|
| `mp20_10.jsonl` | MP-20 crystal CSP, 10 records | `scripts/gen_mat.sh` | `csp_mp20` |
| `drugs_10.jsonl` | GEOM-Drugs conformers, 10 distinct SMILES | `scripts/gen_drugs.sh` | `mol_drugs` |
| `protein_md_2.jsonl` | DESRES fast-folding MD proteins, 2 records | `scripts/gen_prot.sh` | `1_m_p..12_m_p`, `b_p`, `e_bs`, `e_wo_bs` |

The generation scripts default to these files, so after downloading a matching
checkpoint you only need to set `CKPT=` at the top of the script.
