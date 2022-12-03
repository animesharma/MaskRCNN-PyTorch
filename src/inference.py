import os
import torch
from ochuman_dataset import OCHumanDataset
from utils.misc import collate_fn
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_img(index, images, outputs, cpu_device=torch.device("cpu")):
    overall_mask = np.zeros((600, 600, 1))
    for i in range(len(outputs)):
        for j in range(len(outputs[i]["masks"])):
            mask = outputs[i]["masks"][j].to(cpu_device).permute(1, 2, 0).detach().numpy()
            mask[mask >= 0.25] = 1
            mask[mask < 0.25] = 0
            overall_mask += mask
            print(f"Number of masks: {j + 1}")

        img = images[i].to(cpu_device).permute(1, 2, 0).numpy()

        for x in range(600):
            for y in range(600):
                for j in range(3):
                    img[x][y][j] = int(img[x][y][j]*255)

        overall_mask[overall_mask > 0] = 1

        #print(np.shape(overall_mask))
        #print(np.shape(img))

        cyan = np.full_like(img,(255,255,0))

# add cyan to img and save as new image
        blend = 0.5
        img_cyan = cv2.addWeighted(img, blend, cyan, 1-blend, 0)


        new_img = np.zeros([600, 600, 3],dtype = int)
        overall_mask = overall_mask.reshape(600,600)

        for y in range(600):
            for x in range(600):
                if int(overall_mask[y,x]) == 0:
                    new_img[y,x] = img[y,x]
                else:
                    new_img[y,x] = img_cyan[y,x]

        
        plt.figure(figsize = (7, 7))
        #plt.imshow(img)
        plt.imshow(new_img)
        #plt.savefig(os.path.join("./out", f"{index}_{i}.jpg"))
        plt.show()
        #plt.close()

if __name__ == "__main__":
    file_ids = []
    for _, _, file_names in os.walk(os.path.join("./dataset")):
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

        for i, (images, targets) in enumerate(data_loader_test):
            print(i)
            images = list(img.to(device) for img in images)
            outputs = model(images)
            show_img(i, images, outputs)     
