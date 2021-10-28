#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_finetune_deformable_detr_lr8e-5
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --epochs 20 \
    --resume /scratch1/08401/ywen/code/Deformable-DETR/pretrained_dir/r50_deformable_detr-checkpoint.pth \
    ${PY_ARGS}
