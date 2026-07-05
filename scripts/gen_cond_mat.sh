#Please fill in your data path in the vacant code line below
# Conditional (multi-property) material generation (target cond_mat).
# checkpoints: mc_mat / bs_mat_{1..6} / ms_mat_{1..6} (vocab 355).
# The property value conditions generation directly (fed as a continuous value
# in the coordinate stream, after a property marker token).
#
# DICT: use unigenx/data/dict_cond_mat.txt (the property-token material dict,
# vocab 355) where ids 122-128 are <band>/<bulk>/<mag>/<k_avg_cleaned>/<E_hill>/
# <density>/<heat_capacity_300K>. Do NOT use dict_mat.txt here: it has <sgn>1..7
# at those ids, so the property marker maps to <unk> and silently breaks
# conditioning.
CKPT=

CKPT_NAME=$(basename $CKPT)
CKPT_FOLDER=$(dirname $CKPT)

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
    --dict_path unigenx/data/dict_cond_mat.txt \
    --loadcheck_path ${CKPT} \
    --tokenizer num \
    --infer --infer_batch_size 256 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT} \
    --verbose \
    --no_space_group \
    --target cond_mat \
    --top_p 0.8 \
    --temperature 1.0 \
    --diff_steps 200
fi
