import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple

import torchvision.transforms as transforms
import albumentations as A
#import torchvision

#from utils.data_aug import *
#from utils.bbox_util import *

class OCHumanDataset(torch.utils.data.Dataset):
    """
    
    """
    def __init__(self, root_dir, img_ids, transforms, train=True) -> None:
        """
        Constructor
        """
        self.root_dir = root_dir     
        self.img_ids = img_ids
        self.transforms = transforms
        self.train = train

    def __len__(self) -> int:
        """
        
        """
        return len(self.img_ids)

    @staticmethod
    def _get_area(box: list) -> int:
        """
        
        """
        x1, y1, x2, y2 = box
        return((x2 - x1) * (y2 - y1))

    def __getitem__(self, index: int) -> Tuple[np.array, dict]:
        """
        
        """
        img_path = os.path.join(self.root_dir, "images", "train", self.img_ids[index] + ".jpg")
        img = np.array(Image.open(img_path))

        annotation_path = os.path.join(self.root_dir, "annotations", "train", self.img_ids[index] + ".json")
        with open(annotation_path) as f:
            image_id, annotations = json.load(f)

        bboxes = np.array([annotations["boxes"][i] + [annotations["labels"][i]] for i in range(len(annotations["labels"]))])

        

        if self.train:
            transform = A.Compose([
                A.geometric.rotate.SafeRotate(limit=50, p=0.5, border_mode=cv2.BORDER_REPLICATE),
                A.HorizontalFlip(p=0.5),
                A.augmentations.geometric.transforms.Affine([0.8,1],keep_ratio=True,p=0.5),
                A.geometric.resize.LongestMaxSize(max_size=600)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        else:
            transform = A.Compose([
                    A.geometric.resize.LongestMaxSize(max_size=600)
                ])

        transformed = transform(
                          image=img,
                          masks=list(np.array(annotations["masks"])),
                          bboxes=bboxes
                        )

        transformed_image = transformed["image"]
        transformed_masks = transformed["masks"]
        transformed_bboxes = transformed["bboxes"]

        height, width, channels = np.shape(transformed_image)
        # Create a black image
        x = height if height > width else width
        y = height if height > width else width
        resized_img = np.zeros((x, y, channels), np.uint8)
        resized_img[0 : int(y-(y-height)), 0 : int(x-(x-width))] = transformed_image

        resized_masks = []
        for i in range(len(transformed_masks)):
            temp = np.zeros((x, y, 1), np.uint8)
            temp[0 : int(y-(y-height)), 0 : int(x-(x-width))] = np.expand_dims(transformed_masks[i], -1)
            resized_masks.append(np.squeeze(temp))

        #img_, bboxes_ = Resize(600)(img, bboxes)

        #for mask in annotations["masks"]:
        #    mask = np.array(mask).astype('uint8')
        #    mask_ = np.expand_dims(mask, -1)
        #    mask_, _ = Resize(600)(mask_, bboxes)
        #    mask_ = np.squeeze(mask_)
        #    masks.append(mask_)

        bboxes__ = np.array([bbox[:-1] for bbox in transformed_bboxes])
        area = torch.as_tensor(list(map(self._get_area, bboxes__)), dtype=torch.int64)

        bboxes_tensor = torch.as_tensor(bboxes__, dtype=torch.float32)

        tensor_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        #print(resized_img.shape)
        img_tensor = tensor_transform(Image.fromarray(resized_img))
        masks_tensor = torch.from_numpy(np.array(resized_masks))

        target = {"image_id": image_id}
        target["boxes"] = bboxes_tensor
        target["labels"] = torch.as_tensor(annotations["labels"], dtype=torch.int64)
        target["masks"] = masks_tensor
        target["image_id"] = torch.as_tensor(int(image_id), dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = torch.as_tensor([False] * len(annotations["labels"]), dtype=torch.bool)

        return img_tensor, target
