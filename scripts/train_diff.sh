#!/bin/bash
# Single-node UniGenX training example (diffusion-coordinate model).
#
# Fill in TRAIN_DATA / SAVE_DIR below and pick --target + --dict_path for your
# domain. This launches one process on one GPU via torchrun; scale with
# --nproc_per_node=<num_gpus>.
#
# --strategy Zero1 (DeepSpeed ZeRO stage 1) is the default. Zero3 shards the
# model weights, so run DeepSpeed's zero_to_fp32.py on the saved checkpoint to
# consolidate fp32 weights (under the "module" key) BEFORE loading it with
# unigenx_infer.py.

# Please fill in your data / output paths in the vacant lines below.
TRAIN_DATA=
SAVE_DIR=

torchrun --nproc_per_node=1 unigenx_train.py \
    --strategy Zero1 \
    --target material \
    --dict_path unigenx/data/dict_mat.txt \
    --tokenizer num \
    --train_data_path ${TRAIN_DATA} \
    --save_dir ${SAVE_DIR} \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_lr 1e-4 \
    --total_num_steps 100000 \
    --warmup_num_steps 1000 \
    --save_batch_interval 5000 \
    --log_interval 100 \
    --diff_steps 100
    # --dynamic_loader is intentionally omitted (needs the Cython token-bucket
    # loader that is not part of the released single-node training code).
