import os
from shutil import copyfile


VOC_TARGET_DIR = '/scratch1/08401/ywen/data/object_detection/voc2012'
voc_root='/scratch1/08401/ywen/data/object_detection/VOCdevkit/VOC2012'

image_dir = os.path.join(voc_root, "JPEGImages")
annotations_dir = os.path.join(voc_root, "Annotations")

image_set='train'
splits_dir = os.path.join(voc_root, "ImageSets", 'Main')
split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
with open(os.path.join(split_f), "r") as f:
    file_names_train = [x.strip() for x in f.readlines()]

image_set='val'
splits_dir = os.path.join(voc_root, "ImageSets", 'Main')
split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
with open(os.path.join(split_f), "r") as f:
    file_names_potential_val = [x.strip() for x in f.readlines()]

file_names_train.extend(file_names_potential_val[:4000])
file_names_train = sorted(file_names_train)
file_names_val = sorted(file_names_potential_val[4000:])

train_images = [os.path.join(image_dir, x + ".jpg") for x in file_names_train]
target_dir=[os.path.join(VOC_TARGET_DIR, 'train2012', xx +'.jpg') for xx in file_names_train]
for x,y in zip(train_images,target_dir):
    copyfile(x,y)

val_images = [os.path.join(image_dir, x + ".jpg") for x in file_names_val]
target_dir=[os.path.join(VOC_TARGET_DIR, 'val2012', xx +'.jpg') for xx in file_names_val]
for x,y in zip(val_images,target_dir):
    copyfile(x,y)

train_annotations = [os.path.join(annotations_dir, x + ".xml") for x in file_names_train]
target_dir=[os.path.join(VOC_TARGET_DIR, 'train2012_annotations',xx+'.xml') for xx in file_names_train]
for x,y in zip(train_annotations, target_dir):
    copyfile(x,y)

val_annotations = [os.path.join(annotations_dir, x + ".xml") for x in file_names_val]
target_dir=[os.path.join(VOC_TARGET_DIR, 'val2012_annotations',xx+'.xml') for xx in file_names_val]
for x,y in zip(val_annotations, target_dir):
    copyfile(x,y)