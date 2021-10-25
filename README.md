# coding-template

## Summary

The summary can contain but is not limited to:

- Code structure.

- Commands to reproduce your experiments.

- Write-up of your findings and conclusions.

- Ipython notebooks can be organized in `notebooks`.


## DATASETS
### Coco 2017

This dataset contains about training/validation split of 118K/5K  and a total of 330k images  including unannotated images that are labeled uniformly distributed across 80 classes.[Microsoft COCO: Common Objects in Context,2014](https://arxiv.org/pdf/1405.0312.pdf)


## Deformable DETR

For evaluation we for three variants of the model Deformable DETR basic, the single-scale meaning only using res5 feature map (of stride 32) as input feature maps for Deformable Transformer Encoder and the single-scale DC5 means the stride in C5 stage of ResNet and add a dilation of 2 instead.
The epoch was varied from 50-500 across all three models there was no observable changes in both evaluation loss, average precision and average recall for all three models used.
### Deformable DETR Single-Scale DC5

Average Precision 

| Epoch | Loss            | AP    | AP$$_{50}$$ | AP$$_{75}$$ | AP$$_{small}$$| AP$$_{med}$$ | AP$$_{large}$$ |
| :--  | :--            | :--  | :--    |:--     |:--       |:--      |:--        |
| 50    | 6.2433 | 0.414 | 0.618   |0.449    |0.237      |0.453     |0.560

Average Recall 

| Epoch       | Loss            | AR$_{1}$ | AR$_{10}$ | AP$_{100}$ | AR$_{small}$| AR$_{med}$ | AR$_{large}$ |
| :---        | :---            | :---  | :---    |:---      |:---       |:---      |:---        |
| 50          | 6.2433  | 0.340 | 0.556   |0.571     |0.373      |0.624     |0.803


### Deformable DETR Single-Scale

Average Precision  

| Epoch       | Loss            | AP    | AP_{50} | AP_{75} | AP_{small}| AP_{med} | AP_{large} |
| :---        | :---            | :---  | :---    |:---     |:---       |:---      |:---        |
| 50          | 6.2165 (6.9726) | 0.394 | 0.597   |0.422    |0.207      |0.430     |0.559       |

Average Recall

| Epoch       | Loss            | AR$_{1}$ | AR$_{10}$ | AP$_{100}$ | AR$_{small}$| AR$_{med}$ | AR$_{large}$ |
| :---        | :---            | :---  | :---    |:---      |:---       |:---      |:---        |
| 50          | 6.2165 (6.9726) | 0.326 | 0.534   |0.571     |0.328      |0.624     |0.800       |

### Deformable DETR

Average Precision

| Epoch       | Loss            | AP    | AP$_{50}$ | AP$_{75}$ | AP$_{small}$| AP$_{med}$ | AP$_{large}$ |
| :---        | :---            | :---  | :---    |:---     |:---       |:---      |:---        |
| 50          | 5.8611 (6.2284) | 0.445 | 0.635   |0.487    |0.268      |0.477     |0.595       |

Average Recall

| Epoch       | Loss            | AR$_{1}$ | AR$_{10}$ | AP$_{100}$ | AR$_{small}$| AR$_{med}$ | AR$_{large}$ |
| :---        | :---            | :---  | :---    |:---      |:---       |:---      |:---        |
| 50          | 5.8611 (6.2284) | 0.353 | 0.587   |0.629     |0.416      |0.673     |0.819       |

### Ablation with different backbone

| Method             | Backbone   | loss | AP    | AP$_{50}$ | AP$_{75}$ | AP$_{small}$| AP$_{med}$ | AP$_{large}$ |
| :---               | :---       | :---  | :---    | :---    | :---      | :---     | :---   | :---  |
| Deformable DETR    | ResNet-50  | 5.8611 (6.2284) |0.445 | 0.635   |0.487    |0.268      |0.477     |0.595   |
| Deformable DETR    | ResNet-101 | 16.5243 (15.4339)| 0.069 | 0.115 |0.073| 0.063 | 0.092 | 0.051 |
| Deformable DETR-SS | ResNet-50  | 6.2165 (6.9726)  |0.394 | 0.597   |0.422    |0.207      |0.430     |0.559  |
| Deformable DETR-SS | ResNet-101 | 18.5253 (18.0738) |0.052 | 0.106 | 0.044 | 0.028 | 0.077 | 0.062 |
| Deformable DETR-DC5| ResNet-50  |6.2433 (6.6677)|0.414 | 0.618   |0.449    |0.237      |0.453     |0.560 |
| Deformable DETR-DC5| ResNet-101 |17.3633 (17.9159) |0.057 | 0.119 | 0.048 | 0.048 | 0.079 | 0.045 |



## Reference

Any code that you borrow or other reference should be properly cited.
