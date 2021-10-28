# Convert PASCAL VOC data to COCO format 

All SOTA objection detection pipelines are built upon COCO. So we convert PASCAL VOC data into COCO format. Another benefit is we can use COCO evaluator to output IOU conveniently.

## Download 
First download the PASCAL VOC 2012 training/val data at http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit, and unzip it.

Replace the VOC_TARGET_DIR and voc_root variables in the split_voc.py file.

Run the split_voc.py file which splits the PASCAL VOC 2012 into train data and validation data

Now we need to convert VOC annotations (xml files) to COCO format (json files). This is done by VOC2COCO (https://github.com/Tony607/voc2coco).

```bash
python voc2coco.py --xml_dir voc_root/voc2012/train2012_annotations --voc_root/voc2012/annotations/train2012.json
python voc2coco.py --xml_dir voc_root/voc2012/val2012_annotations --voc_root/voc2012/annotations/val2012.json
```
Now we can get the following directory structure
```
voc_root/
    └── voc2012/
        ├── train2012/
        ├── val2012/
        ├── train2012_annotations/
        ├── val2012_annotations/
        └── annotations/
        	├── train2012.json
        	└── train2012.json
```