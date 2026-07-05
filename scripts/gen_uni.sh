#Please fill in your data path in the vacant code line below
# Unified UniGenX model (joint crystal + molecule), dict_uni.txt (vocab 193).
# checkpoints: unified_carbon24 / unified_mp20 / unified_mpts52 / unified_qm9 / unified_pretrain
# Default target below is uni_mat (crystals). For molecular conformers, set:
#     --target uni_mol   (and drop --no_space_group)
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
    --dict_path unigenx/data/dict_uni.txt \
    --loadcheck_path ${CKPT} \
    --tokenizer num \
    --infer --infer_batch_size 256 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT} \
    --verbose \
    --no_space_group \
    --target uni_mat \
    --diff_steps 200
fi
