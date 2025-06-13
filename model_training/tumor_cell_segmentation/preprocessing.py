import glob
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
from data_split import create_dataset_splits
import random
from pathlib import Path

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


"""
    Load the images and masks, and split validation in half to make the test set
"""

ROOT_DATA_DIR = Path("/cluster/home/selmahi/datasets/cell_seg_dataset_slide_split") 

CATEGORIZED_DATASET_NAMES = [
    "ACROBAT",
    "ID",
    "STO",
]

IMAGE_EXTENSIONS = ('*.png', '*.jpg', '*.jpeg')
LABEL_EXTENSIONS = ('*.tif', '*.tiff') 


file_paths = {
    "train": {"images": [], "labels": []},
    "val":   {"images": [], "labels": []},
    "test":  {"images": [], "labels": []},
}

for dataset_name in CATEGORIZED_DATASET_NAMES:
    base_path = ROOT_DATA_DIR / dataset_name
    if not base_path.is_dir():
        print(f"Warning: Dataset folder {base_path} not found. Skipping.")
        continue

    for split_name in ["train", "val", "test"]:
        images_dir = base_path / split_name / "images"
        if images_dir.is_dir():
            for ext in IMAGE_EXTENSIONS:
                file_paths[split_name]["images"].extend(images_dir.glob(ext))
        else:
            print(f"  Warning: Image directory {images_dir} not found. Skipping images for {split_name} in {dataset_name}.")

        labels_dir = base_path / split_name / "labels"
        if labels_dir.is_dir():
            for ext in LABEL_EXTENSIONS:
                file_paths[split_name]["labels"].extend(labels_dir.glob(ext))
        else:
            print(f"  Warning: Label directory {labels_dir} not found. Skipping labels for {split_name} in {dataset_name}.")


train_images = sorted(file_paths["train"]["images"])
val_images   = sorted(file_paths["val"]["images"])
test_images  = sorted(file_paths["test"]["images"])

train_labels = sorted(file_paths["train"]["labels"])
val_labels   = sorted(file_paths["val"]["labels"])
test_labels  = sorted(file_paths["test"]["labels"])

train_files = [{"image": str(img), "mask": str(msk)} for img, msk in zip(train_images, train_labels)]
val_files   = [{"image": str(img), "mask": str(msk)} for img, msk in zip(val_images, val_labels)]
test_files  = [{"image": str(img), "mask": str(msk)} for img, msk in zip(test_images, test_labels)]

# Uncomment to run on half the train and validatoin set

#num_half_train_samples = len(train_files) // 2
#num_half_val_samples = len(val_files) // 2

#train_files = random.sample(train_files, num_half_train_samples)
#val_files = random.sample(val_files, num_half_val_samples)

#print(f"Using {len(train_files)} training samples (half of original).")
#print(f"Using {len(val_files)} validation samples (half of original).")
#print(f"Using {len(test_files)} testing samples (full set).")


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