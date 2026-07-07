# Fill CKPT with the downloaded checkpoint path. INPUT defaults to two fast-
# folding MD examples committed under examples/data/ so the pipeline can run
# immediately after the checkpoint is available.
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
CKPT=${CKPT:-}
INPUT=${INPUT:-examples/data/protein_md_2.jsonl}
PYTHON_BIN=${PYTHON:-python}

if [ -z "${CKPT}" ]; then
    echo "Set CKPT=/path/to/1_m_p.pt before running this script." >&2
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

CKPT_FOLDER=$(dirname "${CKPT}")
CKPT_NAME=$(basename "${CKPT}")
INPUT_FNAME=$(basename "${INPUT}")
OUTPUT=${CKPT_FOLDER}/${CKPT_NAME%.*}_${INPUT_FNAME%.*}.jsonl

if [ -f "${OUTPUT}" ]; then
    rm "${OUTPUT}"
fi
if [ -f "${OUTPUT}" ]; then
    echo "Output file ${OUTPUT} already exists. Skipping."
else
    "${PYTHON_BIN}" unigenx_infer.py \
    --dict_path unigenx/data/dict_prot.txt \
    --loadcheck_path "${CKPT}" \
    --tokenizer num \
    --infer --infer_batch_size 1 \
    --input_file "${INPUT}" \
    --output_file "${OUTPUT}" \
    --verbose \
    --target prot \
    --diff_steps 200
fi
