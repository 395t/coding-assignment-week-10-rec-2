#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_finetune_detr_dc5_lr8e-5
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --epochs 20 \
    --dilation \
    --resume /scratch1/08401/ywen/code/detr/pretrained_dir/detr-r50-dc5-f0fb7ef5.pth \
    ${PY_ARGS}
