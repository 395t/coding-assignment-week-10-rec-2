#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_finetune_detr_lr8e-5
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --epochs 20 \
    --resume /scratch1/08401/ywen/code/detr/pretrained_dir/detr-r50-e632da11.pth \
    ${PY_ARGS}
