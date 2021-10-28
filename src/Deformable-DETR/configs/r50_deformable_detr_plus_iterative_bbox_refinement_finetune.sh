#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_finetune_deformable_detr_plus_iterative_bbox_refinement
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --epochs 20 \
    --resume /scratch1/08401/ywen/code/Deformable-DETR/pretrained_dir/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
    ${PY_ARGS}
