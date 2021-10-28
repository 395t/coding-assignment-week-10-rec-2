# Copied from official DETR

## Data preparation

Follow the instruction in the parent directory to extrac PASCAL VOC 2012 dataset
We expect the directory structure to be the following:
```
voc_root/
    └── voc2012/
        ├── train2012/
        ├── val2012/
        └── annotations/
        	├── train2012.json
        	└── train2012.json
```

### Training

#### Training on single node

For example, the command for training Deformable DETR on 4 GPUs is as following:

```bash
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_detr_finetune.sh
```
#### Training on slurm cluster

If you are using slurm cluster, you can simply run the following command to train on 3 node with each 4 GPUs:

```bash
GPUS_PER_NODE=4 ./tools/run_dist_slurm.sh <partition> deformable_detr 12 configs/r50_detr_finetune.sh
```