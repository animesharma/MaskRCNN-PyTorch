import os
import torch
from ochuman_dataset import OCHumanDataset
from utils.eval import *
from utils.misc import collate_fn

if __name__ == "__main__":
    file_ids = []
    for _, _, file_names in os.walk(os.path.join("./dataset", "annotations", "test")):
        for file_name in file_names:
            file_ids.append(file_name.split(".")[0])

    dataset_test = OCHumanDataset(
                        root_dir="./dataset/",
                        img_ids=file_ids,
                        transforms=None,
                        train=False
                    )
    data_loader_test = torch.utils.data.DataLoader(
                        dataset_test, 
                        batch_size=1, 
                        shuffle=False, 
                        num_workers=1,
                        collate_fn = collate_fn
                    )

    device = torch.device("cuda")
    cpu_device = torch.device("cpu") 

    with torch.no_grad():
        model = torch.load("./out/weights/100.pth")
        model.to(device)
        model.eval()
        
        bboxes = []
        scores = []

        for i, (images, targets) in enumerate(data_loader_test):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for output in outputs:
                print(output.keys())

