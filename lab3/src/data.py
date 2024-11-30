import os
import random

import cv2

random.seed(1100)

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class CatDataset(Dataset):
    def __init__(self, dataset_root, img_list, augment=False) -> None:
        self.dataset_root = dataset_root
        classes = sorted(os.listdir(self.dataset_root))
        self.num_classes = len(classes)
        self.classes_dict = {c: i for (i, c) in enumerate(classes)}

        self.images = img_list
        random.shuffle(self.images)

        if augment:
            self.augmentation_pipeline = A.Compose([
                A.Rotate(limit=(-45, 45)),
                A.RandomScale(scale_limit=(-0.5, 0.5)),
                A.OneOf([
                    A.GaussianBlur(blur_limit=7, sigma_limit=0.2),
                    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    A.ToGray()
                ], p=1),
                A.Resize(height=200, width=200),
                A.Normalize(normalization="min_max", p=1.0),
                ToTensorV2()
            ])
            self.transform = lambda x: self.augmentation_pipeline(image=x)["image"]

        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))),
                transforms.Resize((200, 200), antialias=True)
            ])


    def __getitem__(self, index):
        img_path = self.images[index]
        full_img_path = f"{self.dataset_root}/{img_path}"

        img = cv2.imread(full_img_path, cv2.IMREAD_COLOR)
        img_t = self.transform(img).float()

        lbl = self.classes_dict[img_path.split("/")[0]]
        lbl_t = torch.zeros(size=(self.num_classes, ))
        lbl_t[lbl] = 1

        return img_t, lbl_t
    
    
    def __len__(self):
        return len(self.images)


def getCatDatasetLoaders(train_dataset, test_dataset, batch_size=8):

    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loader_train.__setattr__("len", len(train_dataset) // batch_size + 1)
    loader_test.__setattr__("len", len(test_dataset) // batch_size + 1)
    
    return {'train': loader_train, 'test': loader_test}
