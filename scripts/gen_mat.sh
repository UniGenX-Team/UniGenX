# Fill CKPT with the downloaded checkpoint path. INPUT defaults to a tiny
# MP-20 example set committed under examples/data/ so the pipeline can run
# immediately after the checkpoint is available.
CKPT=${CKPT:-}
INPUT=${INPUT:-examples/data/mp20_10.jsonl}
PYTHON_BIN=${PYTHON:-python}

if [ -z "${CKPT}" ]; then
    echo "Set CKPT=/path/to/csp_mp20.pt before running this script." >&2
    exit 1
fi
if [ ! -f "${CKPT}" ]; then
    echo "Checkpoint not found: ${CKPT}" >&2
    exit 1
fi
if [ ! -f "${INPUT}" ]; then
    echo "Input file not found: ${INPUT}" >&2
    exit 1
fi

CKPT_NAME=$(basename "${CKPT}")
CKPT_FOLDER=$(dirname "${CKPT}")
INPUT_FNAME=$(basename "${INPUT}")
OUTPUT=${CKPT_FOLDER}/${CKPT_NAME%.*}_${INPUT_FNAME%.*}.jsonl

if [ -f "${OUTPUT}" ]; then
    rm "${OUTPUT}"
fi
if [ -f "${OUTPUT}" ]; then
    echo "Output file ${OUTPUT} already exists. Skipping."
else
    "${PYTHON_BIN}" unigenx_infer.py \
    --dict_path unigenx/data/dict_mat.txt \
    --loadcheck_path "${CKPT}" \
    --tokenizer num \
    --infer --infer_batch_size 256 \
    --input_file "${INPUT}" \
    --output_file "${OUTPUT}" \
    --verbose \
    --no_space_group \
    --target material \
    --diff_steps 200
fi
