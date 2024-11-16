import os
import random

import cv2
import numpy as np

random.seed(1100)

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class CatDataset(Dataset):
    def __init__(self) -> None:
        classes = sorted(os.listdir(f"assets/dataset"))
        self.num_classes = len(classes)
        self.classes_dict = {c: i for (i, c) in enumerate(classes)}

        self.images = []
        for c in classes:
            cls_img_list = list(os.listdir(f"assets/dataset/{c}"))
            cls_img_list = [f"{c}/{i}" for i in cls_img_list]
            self.images += cls_img_list

        random.shuffle(self.images)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))),
            transforms.Resize((200, 200), antialias=True)
        ])


    def __getitem__(self, index):
        img_path = self.images[index]
        full_img_path = f"assets/dataset/{img_path}"

        img = cv2.imread(full_img_path, cv2.IMREAD_COLOR)
        img_t = self.transform(img).float()

        lbl = self.classes_dict[img_path.split("/")[0]]
        lbl_t = torch.zeros(size=(self.num_classes, ))
        lbl_t[lbl] = 1

        return img_t, lbl_t
    
    def __len__(self):
        return len(self.images)


def getCatDatasetLoaders(init_dataset, batch_size=8, test_size=0.2):

    train_idx_set, test_idx_set = train_test_split(
        np.arange(len(init_dataset)), 
        test_size=test_size, 
        random_state=42, 
        shuffle=True)

    train_dataset = Subset(init_dataset, train_idx_set)
    test_dataset = Subset(init_dataset, test_idx_set)

    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

    loader_train.__setattr__("len", len(train_dataset) // batch_size + 1)
    loader_test.__setattr__("len", len(test_dataset))
    
    return {'train': loader_train, 'test': loader_test}
