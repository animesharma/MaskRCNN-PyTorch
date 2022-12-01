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
max_mask = np.zeros((np.shape(img)[0], np.shape(img)[1], 1))
for mask in annotations["masks"]:
    mask = np.array(mask).astype('uint8')
    mask_ = np.expand_dims(mask, -1)
    max_mask += mask_
max_mask[max_mask > 1] = 1
mask = max_mask

bbox = np.array([annotations["boxes"][i] + [annotations["labels"][i]] for i in range(len(annotations["labels"]))])
print(bbox)
#bbox = 
#if len(annotations["labels"])>1:
#    bboxes = [bbox[0],bbox[1]]
#else:
#    bboxes = bbox
#print(bboxes)
transform = A.Compose([
    A.geometric.rotate.SafeRotate(limit=50, p=0.5, border_mode=cv2.BORDER_REPLICATE),
    A.HorizontalFlip(p=0.5),
    A.augmentations.geometric.transforms.Affine([0.8,1],keep_ratio=True,p=0.5),
    A.geometric.resize.LongestMaxSize(max_size=600)
], bbox_params=A.BboxParams(format='pascal_voc'))

transformed = transform(
  image=img,
  mask=mask,
  bboxes=bbox
)

transformed_image = transformed["image"]
transformed_mask = transformed["mask"]
transformed_bboxes = transformed["bboxes"]
print(transformed_bboxes)
#print(bboxes,'b')
#print(transformed_bboxes,'t')
plotted_img = transformed_image
cv2.rectangle(transformed_image, (int(transformed_bboxes[0][0]),int(transformed_bboxes[0][1])), (int(transformed_bboxes[0][2]),int(transformed_bboxes[0][3])), (255,0,0), 4)
cv2.rectangle(transformed_image, (int(transformed_bboxes[1][0]),int(transformed_bboxes[1][1])), (int(transformed_bboxes[1][2]),int(transformed_bboxes[1][3])), (255,0,0), 4)
#print(type(img))
#print(type(transformed_image))
#new_size=(892,500)
#transformed_image = transformed_image.copy()
#transformed_image = transformed_image.resize(new_size)
height, width, channels = np.shape(transformed_image)
print (height, width, channels)
# Create a black image
x = height if height > width else width
y = height if height > width else width
square= np.zeros((x,y,3), np.uint8)
#
#This does the job
#
square[0:int(y-(y-height)), 0:int(x-(x-width))] = transformed_image



square_mask= np.zeros((x,y,1), np.uint8)
square_mask[0:int(y-(y-height)), 0:int(x-(x-width))] = transformed_mask

plt.figure(figsize = (7, 7))
plt.imshow(square)
plt.imshow(square_mask,alpha=0.4)
plt.show()
