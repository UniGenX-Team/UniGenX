# Please fill in your data path in the vacant code line below
# EC-number conditioned enzyme (protein-sequence) design (target ecnum).
# checkpoints: e / e_wo  (vocab 64).
#
# The prompt is  <bos> <ec1> L1 <ec2> L2 <ec3> L3 <prot>  where L1/L2/L3 are the
# first three levels of the EC number (split on "."); the model then samples an
# amino-acid sequence conditioned on that EC number and stops at <coord>. INPUT
# is a .jsonl/.pkl whose records carry an "EC_number" (e.g. "1.1.1.1"); one
# enzyme sequence is sampled per record and written to prediction.seq (null when
# a non-standard-residue sequence is produced). To sample many candidates for a
# single EC number, repeat that EC record in the INPUT file.
#
# --target ecnum uses unigenx/data/dict_ecnum.txt (tokenizer vocab 64), which
# matches the e / e_wo checkpoint embeddings.
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
    --dict_path unigenx/data/dict_ecnum.txt \
    --loadcheck_path ${CKPT} \
    --tokenizer num \
    --infer --infer_batch_size 8 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT} \
    --verbose \
    --target ecnum \
    --top_p 0.95 \
    --temperature 1.0 \
    --diff_steps 200
fi
