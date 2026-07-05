#Please fill in your data path in the vacant code line below
# Protein-ligand docking generation (targets: dock / misato).
# checkpoints: pld / pld_u (both vocab 126).
#
# DICT: use unigenx/data/dict_dock.txt (vocab 126 = 119 dict tokens + 7 special
# tokens: <pad><bos><eos><unk> + <mask><coord><sg>).
#
# TARGET:
#   --target misato : apo pocket + ligand SMILES given; the model generates the
#                     holo pocket coordinates and the docked ligand coordinates.
#                     prediction = {apo_coords, holo_coords, ligand_coords}; the
#                     ground truth (holo_coords / lig_coords / apo_coords) is
#                     carried over from the input record. This is the path the
#                     paper docking numbers are evaluated on (best-of-N RMSD).
#   --target dock   : protein pocket coordinates given; the model generates the
#                     ligand pose. prediction = {ligand_coords} (+ prot_coords);
#                     ground truth ligand under ligand_gt.
#
# "Pocket Given" vs "Pocket Not Given" are two different INPUT datasets fed to
# the same code path (not a flag) -- point INPUT at the corresponding directory.
# INPUT is a directory (LMDB sub-dbs for dock / test_mols.pkl + MD_pockets.hdf5
# for misato). Score the output with:
#   python eval/docking/evaluate_docking.py --input <OUTPUT> --samples_per_target 100
CKPT=

CKPT_FOLDER=$(dirname $CKPT)
CKPT_NAME=$(basename $CKPT)

INPUT=

# set to "dock" or "misato"
TARGET=misato

INPUT_FNAME=$(basename $INPUT)
OUTPUT=${CKPT_FOLDER}/${CKPT_NAME%.*}_${INPUT_FNAME%.*}.jsonl

if [ -f ${OUTPUT} ]; then
    rm ${OUTPUT}
fi
if [ -f ${OUTPUT} ]; then
    echo "Output file ${OUTPUT} already exists. Skipping."
else
    python unigenx_infer.py \
    --dict_path unigenx/data/dict_dock.txt \
    --loadcheck_path ${CKPT} \
    --tokenizer num \
    --infer --infer_batch_size 1 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT} \
    --verbose \
    --target ${TARGET} \
    --diff_steps 200
fi
