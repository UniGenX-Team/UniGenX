#Please fill in your data path in the vacant code line below
# Protein-backbone (Cα) conformation generation (target prot).
# checkpoints: 1_m_p..12_m_p / b_p / e_bs / e_wo_bs (all vocab 28).
#
# DICT: use unigenx/data/dict_prot.txt (vocab 28 = 21 residue tokens + 7
# special tokens). For each input sequence the model samples num_topk=5
# conformations (backbone Cα coordinates); short sequences (<=256 residues)
# are generated in one pass, longer ones with a sliding window over the
# sequence. This is the baseline path (sequence tokens + coordinate mask).
#
# Requires --infer_batch_size 1 (one protein per output record), matching the
# reference inference script. INPUT is a .jsonl/.lmdb with per-record protein
# sequences ("seq"/"aa"); the generated coordinates are written to
# prediction.coordinates (a list of the 5 sampled conformations).
CKPT=

CKPT_FOLDER=$(dirname $CKPT)
CKPT_NAME=$(basename $CKPT)

INPUT=

INPUT_FNAME=$(basename $INPUT)
OUTPUT=${CKPT_FOLDER}/${CKPT_NAME%.*}_${INPUT_FNAME%.*}.jsonl

if [ -f ${OUTPUT} ]; then
    rm ${OUTPUT}
fi
if [ -f ${OUTPUT} ]; then
    echo "Output file ${OUTPUT} already exists. Skipping."
else
    python unigenx_infer.py \
    --dict_path unigenx/data/dict_prot.txt \
    --loadcheck_path ${CKPT} \
    --tokenizer num \
    --infer --infer_batch_size 1 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT} \
    --verbose \
    --target prot \
    --diff_steps 200
fi
