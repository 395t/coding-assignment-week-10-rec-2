#!/usr/bin/env bash

set -x

EXP_DIR=exps/r101_deformable_detr_lr8e-5
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --backbone resnet101 \
    ${PY_ARGS}
