from data_split import create_dataset_splits
import numpy as np
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    Transform, 
    MapTransform,
    LoadImaged, 
    NormalizeIntensityd,
    EnsureChannelFirstd, 
    ToTensord, 
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandScaleIntensityd,
    Lambdad
    )
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import random


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


"""
    Load the images and masks
"""

splits = create_dataset_splits("/cluster/home/selmahi/datasets/250325_mib1_selma_4096_ds2_5x_sematic_seg_tumor")

train_images = splits["train_images"]
val_images   = splits["val_images"]
test_images  = splits["test_images"]

train_labels = splits["train_labels"]
val_labels   = splits["val_labels"]
test_labels  = splits["test_labels"]

train_files = [{"image": img, "mask": msk} for img, msk in zip(sorted(train_images), sorted(train_labels))]
val_files = [{"image": img, "mask": msk} for img, msk in zip(sorted(val_images), sorted(val_labels))]
test_files = [{"image": img, "mask": msk} for img, msk in zip(sorted(test_images), sorted(test_labels))]

print("Using all images")


"""
Create transforms for training and validation set
"""

class NormalizeImage(Transform):
    def __call__(self, img):
        min_value = np.min(img)
        max_value = np.max(img)
        if max_value - min_value > 0:
            return (img - min_value) / (max_value - min_value)
        else:
            return img

class NormalizeImageD(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        normalizer = NormalizeImage()
        for key in self.keys:
            d[key] = normalizer(d[key]) 
        return d

train_transform = Compose([
    LoadImaged(keys=["image", "mask"], reader="pilreader"),
    EnsureChannelFirstd(keys=["image", "mask"]),  
    RandFlipd(keys=["image", "mask"], prob=0.3, spatial_axis=[0, 1]), 
    RandRotate90d(keys=["image", "mask"], prob=0.3), 
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
    NormalizeImageD(keys=["image"]),
    ToTensord(keys=["image", "mask"])
])

val_test_transform = Compose([
    LoadImaged(keys=["image", "mask"], reader="pilreader"),
    EnsureChannelFirstd(keys=["image", "mask"]),  
    NormalizeImageD(keys=["image"]),
    ToTensord(keys=["image", "mask"])
])


"""
Create DataLoaders
"""

def custom_collate_fn(batch):
    for item in batch:
        mask = item["mask"]
        if mask.dim() == 2:
            item["mask"] = mask.unsqueeze(0)
    return torch.utils.data._utils.collate.default_collate(batch)

def get_dataloaders(batch_size):
    
    train_ds = Dataset(data=train_files, transform=train_transform)
    val_ds = Dataset(data=val_files, transform=val_test_transform)
    test_ds = Dataset(data=test_files, transform=val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader