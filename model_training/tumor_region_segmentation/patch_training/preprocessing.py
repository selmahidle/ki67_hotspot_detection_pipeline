from data_split import create_dataset_splits
import numpy as np
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    Transform,
    MapTransform,
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityd,
    RandScaleIntensityd,
    RandCropByPosNegLabeld
)

import torch
import random
from torch.utils.data.dataloader import default_collate

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


"""
    Load the file paths for images and masks
"""

try:
    dataset_base_path = "/cluster/home/selmahi/datasets/250325_mib1_selma_4096_ds2_5x_sematic_seg_tumor"
    splits = create_dataset_splits(dataset_base_path)

    train_images = splits["train_images"]
    val_images   = splits["val_images"]
    test_images  = splits["test_images"] 

    train_labels = splits["train_labels"]
    val_labels   = splits["val_labels"]
    test_labels  = splits["test_labels"]

    train_files = [{"image": img, "mask": msk} for img, msk in zip(sorted(train_images), sorted(train_labels))]
    val_files = [{"image": img, "mask": msk} for img, msk in zip(sorted(val_images), sorted(val_labels))]
    test_files = [{"image": img, "mask": msk} for img, msk in zip(sorted(test_images), sorted(test_labels))]
    print(f"Loaded {len(train_files)} train, {len(val_files)} val, {len(test_files)} test file pairs.")

except Exception as e:
    print(f"Error loading dataset splits from '{dataset_base_path}': {e}")
    print("Please ensure 'data_split.py' is correct and the path is valid.")
    train_files, val_files, test_files = [], [], []


"""
Define Patch Size and Transforms
"""
patch_size = (1024, 1024) 
num_patches_per_image_train = 1

class NormalizeImage(Transform):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        min_value = np.min(img)
        max_value = np.max(img)
        if max_value - min_value > 1e-6: 
            return (img - min_value) / (max_value - min_value)
        else:
            return img 

class NormalizeImageD(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)
        self.normalizer = NormalizeImage() 

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d and isinstance(d[key], np.ndarray):
                d[key] = self.normalizer(d[key])
        return d

train_transform = Compose([
    LoadImaged(keys=["image", "mask"], reader="pilreader", image_only=False),
    EnsureChannelFirstd(keys=["image", "mask"]),
    RandCropByPosNegLabeld(
        keys=["image", "mask"],
        label_key="mask",
        spatial_size=patch_size,
        pos=1.0,  
        neg=1.0,  
        num_samples=num_patches_per_image_train,
        allow_smaller=True
    ),
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0), 
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1), 
    RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),
    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    ToTensord(keys=["image", "mask"], track_meta=False)
])


val_test_transform = Compose([
    LoadImaged(keys=["image", "mask"], reader="pilreader", image_only=False),
    EnsureChannelFirstd(keys=["image", "mask"]),
    RandCropByPosNegLabeld(
        keys=["image", "mask"],
        label_key="mask",
        spatial_size=patch_size,
        pos=1.0,
        neg=0.0,
        num_samples=1,
        allow_smaller=True
    ),
    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    ToTensord(keys=["image", "mask"], track_meta=False)
])


"""
Create DataLoaders
"""
def custom_collate_fn(batch):
    flattened = []
    for item in batch:
        if isinstance(item, list) and all(isinstance(x, dict) for x in item):
            flattened.extend(item)
        else:
            flattened.append(item)
    return default_collate(flattened)



def get_dataloaders(batch_size: int):
    """
    Creates and returns patch-based DataLoaders for training, validation, and testing.
    Args:
        batch_size: The number of patches in each batch.
    """

    train_ds = Dataset(data=train_files, transform=train_transform)
    val_ds = Dataset(data=val_files, transform=val_test_transform)
    test_ds = Dataset(data=test_files, transform=val_test_transform) 

    num_workers_to_use = 0 

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=num_workers_to_use,
        pin_memory=torch.cuda.is_available(), 
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers_to_use,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=num_workers_to_use,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    print(f"DataLoaders created with batch_size={batch_size} for patches.")
    return train_loader, val_loader, test_loader