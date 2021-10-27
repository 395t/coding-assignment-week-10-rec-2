# coding-template

## Summary

The summary can contain but is not limited to:

- Code structure.

- Commands to reproduce your experiments.

- Write-up of your findings and conclusions.

- Ipython notebooks can be organized in `notebooks`.



We covered the implementation and evaluation of these object detection models

- [Deformable Convolutional Networks, Dai, Qi, Xiong, Li, Zhang, Hu, Wei; 2017](https://arxiv.org/abs/1703.06211)

- [CornerNet: Detecting Objects as Paired Keypoints, Law, Deng; 2018](https://arxiv.org/abs/1808.01244)

- [Objects as Points , Zhou, Wang, Krähenbühl; 2019](https://arxiv.org/abs/1904.07850)

- [End-to-End Object Detection with Transformers,, Carion, Massa, Synnaeve, Usunier, Kirillov, Zagoruyko; 2020](https://arxiv.org/abs/2005.12872)

- [Deformable DETR: Deformable Transformers for End-to-End Object Detection , Zhu, Su, Lu, Li, Wang, Dai; 2020](https://arxiv.org/abs/2010.04159#)

## DATASETS

We evaluate our tasks on two datasets
### COCO 2017

This dataset contains about training/validation split of 118K/5K  and a total of 330k images including unannotated images that are labeled uniformly distributed across 80 classes.

- [Microsoft COCO: Common Objects in Context,2014](https://arxiv.org/pdf/1405.0312.pdf)

### PASCAL VOC 2012

This consists of 20 object categories with each image has pixel-level segmentation annotations, bounding box annotations, and object class annotations.training/validation split of 9717/1823 images.

- [PASCAL VOC (PASCAL Visual Object Classes Challenge)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#data)


## Deformable Convolutional Networks

OpenMMLab Github Repository: https://github.com/open-mmlab/

The following data are all pre-trained by OpenMMlab using the COCO 2017 dataset

### Loss during training
The following four models use the same backbone R-50-FPN
![](images/deconv_coco_methods.png)

![](images/deconv_coco_models.png)
### faster rcnn r50
| Evaluation Type | IoU | Area | MaxDets | Result |
| ----------- | --- | --- | --- | --- |
|Average Precision|  0.50:0.95 | all | 100|0.374
|Average Precision|  0.50 | all | 1000|0.581
|Average Precision|  0.75 | all | 1000|0.404
|Average Precision|  0.50:0.95 | small | 1000|0.212
|Average Precision|  0.50:0.95 | medium | 1000|0.410
|Average Precision|  0.50:0.95 | large | 1000|0.481
|Average Recall|  0.50:0.95 | all | 100|0.517
|Average Recall|  0.50:0.95 | all | 300|0.517
|Average Recall|  0.50:0.95 | all | 1000|0.516
|Average Recall|  0.50:0.95 | small | 1000|0.326
|Average Recall|  0.50:0.95 | medium | 1000|0.557
|Average Recall|  0.50:0.95 | large | 1000|0.648

### dconv faster rcnn r50
| Evaluation Type | IoU | Area | MaxDets | Result |
| ----------- | --- | --- | --- | --- |
|Average Precision|  0.50:0.95 | all | 100|0.413
|Average Precision|  0.50 | all | 1000|0.624
|Average Precision|  0.75 | all | 1000|0.450
|Average Precision|  0.50:0.95 | small | 1000|0.246
|Average Precision|  0.50:0.95 | medium | 1000|0.449
|Average Precision|  0.50:0.95 | large | 1000|0.554
|Average Recall|  0.50:0.95 | all | 100|0.549
|Average Recall|  0.50:0.95 | all | 300|0.549
|Average Recall|  0.50:0.95 | all | 1000|0.549
|Average Recall|  0.50:0.95 | small | 1000|0.353
|Average Recall|  0.50:0.95 | medium | 1000|0.590
|Average Recall|  0.50:0.95 | large | 1000|0.698

### mdconv faster rcnn r50

| Evaluation Type | IoU | Area | MaxDets | Result |
| ----------- | --- | --- | --- | --- |
|Average Precision|  0.50:0.95 | all | 100|0.414
|Average Precision|  0.50 | all | 1000|0.625
|Average Precision|  0.75 | all | 1000|0.456
|Average Precision|  0.50:0.95 | small | 1000|0.246
|Average Precision|  0.50:0.95 | medium | 1000|0.452
|Average Precision|  0.50:0.95 | large | 100|0.542
|Average Recall|  0.50:0.95 | all | 300|0.548
|Average Recall|  0.50:0.95 | all | 1000|0.548
|Average Recall|  0.50:0.95 | all | 1000|0.548
|Average Recall|  0.50:0.95 | small | 1000|0.359
|Average Recall|  0.50:0.95 | medium | 1000|0.587
|Average Recall|  0.50:0.95 | large | 1000|0.691

### dconv faster rcnn r50 dpool

| Evaluation Type | IoU | Area | MaxDets | Result |
| ----------- | --- | --- | --- | --- |
|Average Precision|  0.50:0.95 | all | 100|0.381
|Average Precision|  0.50 | all | 1000|0.597
|Average Precision|  0.75 | all | 1000|0.420
|Average Precision|  0.50:0.95 | small | 1000|0.224
|Average Precision|  0.50:0.95 | medium | 1000|0.415
|Average Precision|  0.50:0.95 | large | 100|0.494
|Average Recall|  0.50:0.95 | all | 300|0.522
|Average Recall|  0.50:0.95 | all | 1000|0.522
|Average Recall|  0.50:0.95 | all | 1000|0.522
|Average Recall|  0.50:0.95 | small | 1000|0.334
|Average Recall|  0.50:0.95 | medium | 1000|0.559
|Average Recall|  0.50:0.95 | large | 1000|0.655

### dconv faster rcnn r50 mdpool

| Evaluation Type | IoU | Area | MaxDets | Result |
| ----------- | --- | --- | --- | --- |
|Average Precision|  0.50:0.95 | all | 100|0.379
|Average Precision|  0.50 | all | 1000|0.594
|Average Precision|  0.75 | all | 1000|0.418
|Average Precision|  0.50:0.95 | small | 1000|0.224
|Average Precision|  0.50:0.95 | medium | 1000|0.414
|Average Precision|  0.50:0.95 | large | 100|0.493
|Average Recall|  0.50:0.95 | all | 300|0.519
|Average Recall|  0.50:0.95 | all | 1000|0.519
|Average Recall|  0.50:0.95 | all | 1000|0.519
|Average Recall|  0.50:0.95 | small | 1000|0.329
|Average Recall|  0.50:0.95 | medium | 1000|0.558
|Average Recall|  0.50:0.95 | large | 1000|0.652

## Deformable DETR

For evaluation we for three variants of the model Deformable DETR basic, the single-scale meaning only using res5 feature map (of stride 32) as input feature maps for Deformable Transformer Encoder and the single-scale DC5 means the stride in C5 stage of ResNet and add a dilation of 2 instead.
The epoch was varied from 50-500 across all three models there was no observable changes in both evaluation loss, average precision and average recall for all three models used.

## Evaluation Results

Training with 5 epochs on Pascal VOC datasets with 70 number of queries.

![voc-training](images/deformable-detr-voc-training.png)

Validation



Due to constraint in time to train evaluated on model on coco 2017 datasets comparing the single-scale,single-scale-dc5 and the base detr model.
### Deformable DETR Single-Scale DC5

Average Precision 

| Epoch | Loss | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>small</sub>| AP<sub>med</sub> | AP<sub>large</sub> |
| :--  | :--   | :--  | :-- |:--  |:--  |:-- |:-- |
| 50   | 6.2433 | 0.414 | 0.618 |0.449|0.237|0.453|0.560

Average Recall 

| Epoch | Loss | AR<sub>1</sub> | AR<sub>10</sub> | AP<sub>100</sub> | AR<sub>small</sub>| AR<sub>med</sub> | AR<sub>large</sub> |
| :--  | :-- | :--  | :--  |:-- |:-- |:--  |:-- |
| 50   | 6.2433| 0.340 | 0.556|0.571|0.373 |0.624|0.803


### Deformable DETR Single-Scale

Average Precision  

| Epoch | Loss | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>small</sub>| AP<sub>med</sub> | AP<sub>large</sub> |
| :--  | :--   | :--  | :-- |:--  |:--  |:-- |:-- |
| 50 | 6.2165 | 0.394 | 0.597|0.422|0.207|0.430 |0.559 |

Average Recall

| Epoch | Loss | AR<sub>1</sub> | AR<sub>10</sub> | AP<sub>100</sub> | AR<sub>small</sub>| AR<sub>med</sub> | AR<sub>large</sub> |
| :--  | :--   | :--  | :-- |:--  |:--  |:-- |:-- |
| 50  | 6.2165| 0.326 | 0.534|0.571|0.328 |0.624 |0.800|

### Deformable DETR

Average Precision

| Epoch | Loss | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>small</sub>| AP<sub>med</sub> | AP<sub>large</sub> |
| :--  | :--   | :--  | :-- |:--  |:--  |:-- |:-- |
| 50  | 5.8611 | 0.445 | 0.635 |0.487 |0.268 |0.477 |0.595 |

Average Recall

| Epoch | Loss | AR<sub>1</sub> | AR<sub>10</sub> | AP<sub>100</sub> | AR<sub>small</sub>| AR<sub>med</sub> | AR<sub>large</sub> |
| :--  | :--   | :--  | :-- |:--  |:--  |:-- |:-- |
| 50  | 5.8611 | 0.353 | 0.587 |0.629 |0.416 |0.673 |0.819 |

Comparison by changing the convolution backbone of ResNet-50 vs ResNet-101

| Method             | Backbone   | loss | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>small</sub>| AP<sub>med</sub> | AP<sub>large</sub> |
| :--               | :--       | :--  | :--    | :--  | :-- | :-- | :-- | :-- |
| Deformable DETR    | ResNet-50  | 5.8611  |0.445 | 0.635   |0.487 |0.268 s|0.477 |0.595 |
| Deformable DETR    | ResNet-101 | 16.5243 | 0.069 | 0.115 |0.073| 0.063 | 0.092 | 0.051 |
| Deformable DETR-SS | ResNet-50  | 6.2165   |0.394 | 0.597   |0.422    |0.207 |0.430 |0.559  |
| Deformable DETR-SS | ResNet-101 | 18.5253  |0.052 | 0.106 | 0.044 | 0.028 | 0.077 | 0.062 |
| Deformable DETR-DC5| ResNet-50  |6.2433 |0.414 | 0.618   |0.449 |0.237 |0.453 |0.560 |
| Deformable DETR-DC5| ResNet-101 |17.3633  |0.057 | 0.119 | 0.048 | 0.048 | 0.079 | 0.045 |

## Reference

[Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
