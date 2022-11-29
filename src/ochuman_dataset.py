import os
import json
import torch
import numpy as np
from PIL import Image
from typing import Tuple

import torchvision.transforms as transforms
import torchvision

from utils.data_aug import *
from utils.bbox_util import *

class OCHumanDataset(torch.utils.data.Dataset):
    """
    
    """
    def __init__(self, root_dir, img_ids, transforms) -> None:
        """
        Constructor
        """
        self.root_dir = root_dir     
        self.img_ids = img_ids
        self.transforms = transforms

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
        img_path = os.path.join(self.root_dir, "images", self.img_ids[index] + ".jpg")
        img = np.array(Image.open(img_path))

        annotation_path = os.path.join(self.root_dir, "annotations", self.img_ids[index] + ".json")
        with open(annotation_path) as f:
            image_id, annotations = json.load(f)

        bboxes = np.array([annotations["boxes"][i] + [annotations["labels"][i]] for i in range(len(annotations["labels"]))])
        
        img_, bboxes_ = Resize(600)(img, bboxes)

        masks = []

        for mask in annotations["masks"]:
            mask = np.array(mask).astype('uint8')
            mask_ = np.expand_dims(mask, -1)
            mask_, _ = Resize(600)(mask_, bboxes)
            mask_ = np.squeeze(mask_)
            masks.append(mask_)

        bboxes__ = np.array([bbox[:-1] for bbox in bboxes_])
        area = torch.as_tensor(list(map(self._get_area, bboxes__)), dtype=torch.int64)

        bboxes_tensor = torch.as_tensor(bboxes__, dtype=torch.float32)
        transform = transforms.Compose([
                transforms.ToTensor()
            ])
        img_pil = Image.fromarray(img_)
        #img_tensor = torch.from_numpy(img_)
        img_tensor = transform(img_pil)
        #img_tensor = torch.as_tensor(torchvision.transforms.functional.convert_image_dtype(img_, dtype = torch.float32))
        #print(f"{image_id}\t{type(img_tensor)}")
        #print(f"Image ID: {image_id}\tBBoxes: {bboxes_tensor}\n")
        masks_tensor = torch.from_numpy(np.array(masks))

        target = {"image_id": image_id}
        target["boxes"] = bboxes_tensor
        target["labels"] = torch.as_tensor(annotations["labels"], dtype=torch.int64)
        target["masks"] = masks_tensor
        target["image_id"] = torch.as_tensor(int(image_id), dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = torch.as_tensor([False] * len(annotations["labels"]), dtype=torch.bool)

        return img_tensor, target
