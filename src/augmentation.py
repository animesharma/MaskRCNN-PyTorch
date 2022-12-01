import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from PIL import Image
import albumentations as A
import random

import json
with open("C:/Users/WanyingMo/Desktop/data/000019.json") as f:
    img_id, annotations = json.load(f)
    #bboxes = np.array([annotations["boxes"][i] for i in range(len(annotations["labels"]))])
img = cv2.imread("C:/Users/WanyingMo/Desktop/images/000019.jpg")[:,:,::-1]

masks = annotations["masks"]
max_mask = np.zeros((np.shape(img)[0], np.shape(img)[1], 1))
for mask in annotations["masks"]:
    mask = np.array(mask).astype('uint8')
    mask_ = np.expand_dims(mask, -1)
    max_mask += mask_
max_mask[max_mask > 1] = 1
mask = max_mask

bbox = np.array([annotations["boxes"][i] + [annotations["labels"][i]] for i in range(len(annotations["labels"]))])

if len(annotations["labels"])>1:
    bboxes = [bbox[0],bbox[1]]
else:
    bboxes = bbox
transform = A.Compose([
    A.geometric.rotate.SafeRotate(limit=90, p=0.6),
    A.HorizontalFlip(p=0.4),
    A.geometric.resize.Resize(height=600,width=600),
], bbox_params=A.BboxParams(format='pascal_voc'))

transformed = transform(
  image=img,
  mask=mask,
  bboxes=bboxes
)

transformed_image = transformed["image"]
transformed_mask = transformed["mask"]
transformed_bboxes = transformed["bboxes"]
print(bboxes,'b')
print(transformed_bboxes,'t')
plotted_img = cv2.rectangle(transformed_image, (int(transformed_bboxes[0][0]),int(transformed_bboxes[0][1])), (int(transformed_bboxes[0][2]),int(transformed_bboxes[0][3])), (255,0,0), 4)
plt.figure(figsize = (7, 7))
plt.imshow(plotted_img)
plt.imshow(transformed_mask,alpha=0.4)
plt.show()
