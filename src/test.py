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

    pred_dict = {}
    gt_dict = {}

    with torch.no_grad():
        model = torch.load("./out/weights/100.pth")
        model.to(device)
        model.eval()
        
        bboxes = []
        scores = []

        for i, (images, targets) in enumerate(data_loader_test):
            print(i)
            images = list(img.to(device) for img in images)

            outputs = model(images)
            pred_boxes = outputs[0]["boxes"].to(cpu_device).numpy()
            pred_scores = outputs[0]["scores"].to(cpu_device).numpy()
            image_id = targets[0]["image_id"].item()
            gt_boxes = targets[0]["boxes"].to(cpu_device).numpy()
            pred_dict[image_id] = {"boxes": pred_boxes, "scores": pred_scores}
            gt_dict[image_id] = gt_boxes

            print(pred_dict, gt_dict)

            

    #print(pred_dict)

