#Please fill in your data path in the vacant code line below
# Property-conditional molecule generation (target cond_mol).
# checkpoint: c_mol (vocab 34).
#
# DICT: use unigenx/data/dict_cond_mol.txt (vocab 34) whose last 6 standard
# tokens are the property markers <a><g><h><l><m><c> (alpha/gap/homo/lumo/mu/Cv).
# Each property value conditions generation as a continuous coordinate row after
# its marker token; one or many (num_cond) joint property constraints per
# molecule are supported (single- and multi-property LDM conditioning). This is
# pure conditional generation -- no classifier-free guidance.
#
# Two-phase: phase 1 samples the SMILES conditioned on the property prefix
# (RDKit-validity filtered), phase 2 generates the atom coordinates.
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
    --dict_path unigenx/data/dict_cond_mol.txt \
    --loadcheck_path ${CKPT} \
    --tokenizer num \
    --infer --infer_batch_size 256 \
    --input_file ${INPUT} \
    --output_file ${OUTPUT} \
    --verbose \
    --target cond_mol \
    --diff_steps 200
fi
