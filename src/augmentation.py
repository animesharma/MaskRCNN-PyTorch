import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from PIL import Image
import albumentations as A
import random

import json
#from PIL import Image

with open("C:/Users/WanyingMo/Desktop/data/000040.json") as f:
    img_id, annotations = json.load(f)
    #bboxes = np.array([annotations["boxes"][i] for i in range(len(annotations["labels"]))])
img = cv2.imread("C:/Users/WanyingMo/Desktop/images/000040.jpg")[:,:,::-1]

masks = annotations["masks"]


mask = list(np.array(masks))

bbox = np.array([annotations["boxes"][i] + [annotations["labels"][i]] for i in range(len(annotations["labels"]))])

transform = A.Compose([
    A.geometric.rotate.SafeRotate(limit=50, p=0.5, border_mode=cv2.BORDER_REPLICATE),
    A.HorizontalFlip(p=0.5),
    A.augmentations.geometric.transforms.Affine([0.8,1],keep_ratio=True,p=0.5),
    A.geometric.resize.LongestMaxSize(max_size=600)
], bbox_params=A.BboxParams(format='pascal_voc'))

transformed = transform(
  image=img,
  masks=mask,
  bboxes=bbox
)

transformed_image = transformed["image"]
transformed_mask = transformed["masks"]
print(len(transformed_mask))
transformed_bboxes = transformed["bboxes"]
print(transformed_bboxes)

plotted_img = transformed_image
cv2.rectangle(transformed_image, (int(transformed_bboxes[0][0]),int(transformed_bboxes[0][1])), (int(transformed_bboxes[0][2]),int(transformed_bboxes[0][3])), (255,0,0), 4)
cv2.rectangle(transformed_image, (int(transformed_bboxes[1][0]),int(transformed_bboxes[1][1])), (int(transformed_bboxes[1][2]),int(transformed_bboxes[1][3])), (255,0,0), 4)

height, width, channels = np.shape(transformed_image)
# Create a black image
x = height if height > width else width
y = height if height > width else width
square= np.zeros((x,y,3), np.uint8)
#
#This does the job
#
square[0:int(y-(y-height)), 0:int(x-(x-width))] = transformed_image


square_masks=[]
for i in range(len(transformed_mask)):
    temp = np.zeros((x,y,1), np.uint8)
    temp[0:int(y-(y-height)), 0:int(x-(x-width))] = np.expand_dims(transformed_mask[i], -1)
    square_masks.append(temp)

plt.figure(figsize = (7, 7))
plt.imshow(square)
plt.imshow(square_masks[0],alpha=0.4)
plt.imshow(square_masks[1],alpha=0.4)
plt.show()
