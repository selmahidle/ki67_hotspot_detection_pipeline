"""
UNet training script for semantic segmentation of tumor region in 1024x1024 WSI patches (tested on 4096x4096 patches)
"""

from preprocessing import get_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import argparse
import numpy as np
import random
import cv2
from sklearn.metrics import confusion_matrix
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import os
from monai.networks.nets import UNet
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord
from monai.inferers import SlidingWindowInferer
from monai.data import Dataset as MonaiDataset, DataLoader as MonaiDataLoader 
from preprocessing import test_files as full_image_test_files


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
parser.add_argument("--init_lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
parser.add_argument("--num_res_units", type=int, default=3, help="Numer of residual units")
parser.add_argument("--slurm_job_id", type=int, default=None, help="Slurm job ID")
parser.add_argument("--comment", type=str, default=None, help="Comment")
args = parser.parse_args()


run = wandb.init(
    entity="selma_mib",
    project="Tumor Segmentation",
    config={
            "comments": args.comment, 
            "init_learning_rate": args.init_lr,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "model_architecture": "Unet",
            "loss_function": "Tversky",
            "slurm_job_id": args.slurm_job_id
        })

train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_memory(stage):
    gpu_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    ram_usage = psutil.virtual_memory().percent

    print(f"[{stage}] GPU allocated: {gpu_allocated:.2f} GB | GPU reserved: {gpu_reserved:.2f} GB | RAM usage: {ram_usage}%")


"""
Define the model, loss function and optimizer
"""

model = UNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=1,
    channels=(32, 64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    num_res_units = args.num_res_units
).to(device)

wandb.watch(model, log='all', log_freq=100)

optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
loss_function = TverskyLoss(sigmoid=True) 
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
dice_metric = DiceMetric()
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


"""
Training and validation loop
"""

best_metric = -1
best_metric_epoch = -1

for epoch in range(args.num_epochs):
    print(f"\n---Epoch {epoch + 1}/{args.num_epochs}---")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, masks = batch_data["image"].to(device), batch_data["mask"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    print(f"Average Tversky Loss: {epoch_loss:.6f}")
    wandb.log({
        "training_loss": epoch_loss,
        "epoch": epoch
    }, step=epoch)

    # Validation
    model.eval()
    with torch.no_grad():
        val_dice_scores = []
        epochs_sum_val_loss = 0
        for val_data in val_loader:
            val_inputs, val_masks = val_data["image"].to(device), val_data["mask"].to(device)
            val_outputs = model(val_inputs)
            val_loss = loss_function(val_outputs, val_masks)
            epochs_sum_val_loss += val_loss.item()  

            transformed_val_outputs = post_trans(val_outputs)
            dice_metric(y_pred=transformed_val_outputs, y=val_masks)
            dice_score = dice_metric.aggregate().item()
            val_dice_scores.append(dice_score)
            dice_metric.reset()

        avg_dice_value = sum(val_dice_scores) / len(val_dice_scores)
        epochs_sum_val_loss /= len(val_loader)
        wandb.log({
            "validation_loss": epochs_sum_val_loss,
            "validation_dice": avg_dice_value,
            "epoch": epoch
        }, step=epoch)
        print(f"Validation Metrics: Dice: {avg_dice_value:.4f}, Tversky Loss: {epochs_sum_val_loss:.4f}")

        lr_scheduler.step(epochs_sum_val_loss) 
        current_lr = lr_scheduler.get_last_lr()[0] 
        wandb.log({"learning_rate": current_lr, "epoch": epoch})

        if avg_dice_value > best_metric:
            best_metric = avg_dice_value
            best_metric_epoch = epoch + 1
            checkpoint_filename = f"/cluster/home/selmahi/mib_pipeline/mib_pipeline_scripts/checkpoints/tumor_unet/{args.slurm_job_id}_best_tumor_semseg_unet_model.pth"
            torch.save(model.state_dict(), checkpoint_filename)

del val_inputs, val_masks, val_outputs, inputs, masks, outputs, loss, val_loss
torch.cuda.empty_cache()

print(f"Training Complete. Best Dice Score: {best_metric:.4f} at Epoch: {best_metric_epoch}")





"""
Testing the model on FULL 4096x4096 IMAGES using SlidingWindowInferer
"""

if os.path.exists(checkpoint_filename) and best_metric > -1 : 
    print(f"\nLoading best model from epoch {best_metric_epoch} for testing: {checkpoint_filename}")
    model.load_state_dict(torch.load(checkpoint_filename, map_location=device))
else:
    print("\nWARNING: No best model checkpoint found or training did not improve. Testing with the model from the last epoch.")
model.to(device)
model.eval() 


print("\nTesting the model...")
full_image_test_transform = Compose([
    LoadImaged(keys=["image", "mask"], reader="pilreader", image_only=False),
    EnsureChannelFirstd(keys=["image", "mask"]),
    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    ToTensord(keys=["image", "mask"], track_meta=False)
])


full_image_test_ds = MonaiDataset(data=full_image_test_files, transform=full_image_test_transform)
full_image_test_loader = MonaiDataLoader(full_image_test_ds, batch_size=1, shuffle=False, num_workers=0)


patch_roi_size = (1024, 1024) 
sw_batch_size = 4             
overlap_ratio = 0.5          

inferer = SlidingWindowInferer(
    roi_size=patch_roi_size,
    sw_batch_size=sw_batch_size,
    overlap=overlap_ratio,
    mode="gaussian", 
    sigma_scale=0.2, 
    padding_mode="constant",
    cval=0,
    progress=True
)


dice_metric.reset()
cm_total_full = np.zeros((2, 2), dtype=int) 
full_image_dice_scores = []

columns_full = ["Original Image Filename", "Full Input Image (Resized)", "Full Ground Truth (Resized)", "Full Prediction (Resized)", "Full Image Dice Score"]
table_full = wandb.Table(columns=columns_full)
num_images_to_log_full = 5

with torch.no_grad():
    for idx, full_test_data_item in enumerate(full_image_test_loader):
        full_inputs = full_test_data_item["image"].to(device) 
        full_masks = full_test_data_item["mask"].to(device)   
        original_filename = full_test_data_item["image_meta_dict"]["filename_or_obj"][0] 
        
        print(f"Inferring on full image {idx+1}/{len(full_image_test_loader)}: {os.path.basename(original_filename)}")

        full_outputs_logits = inferer(inputs=full_inputs, network=model) 
        full_outputs_pred = post_trans(full_outputs_logits)

        dice_metric.reset()
        dice_metric(y_pred=full_outputs_pred, y=full_masks)
        current_full_image_dice = dice_metric.aggregate().item()
        full_image_dice_scores.append(current_full_image_dice)
        print(f"  Dice for {os.path.basename(original_filename)}: {current_full_image_dice:.4f}")

        preds_flat_full = full_outputs_pred.cpu().numpy().astype(int).flatten()
        labels_flat_full = full_masks.cpu().numpy().astype(int).flatten()
        
        cm_current_full_image = confusion_matrix(labels_flat_full, preds_flat_full, labels=[0, 1])
        if cm_current_full_image.shape == (2,2):
            cm_total_full += cm_current_full_image
        else:
            temp_cm = np.zeros((2, 2), dtype=int)
            present_labels_cm = np.unique(np.concatenate((labels_flat_full, preds_flat_full)))
            sub_cm = confusion_matrix(labels_flat_full, preds_flat_full, labels=present_labels_cm)
            for i_cm, true_label_cm in enumerate(present_labels_cm):
                for j_cm, pred_label_cm in enumerate(present_labels_cm):
                    if true_label_cm < 2 and pred_label_cm < 2:
                        temp_cm[true_label_cm, pred_label_cm] = sub_cm[i_cm, j_cm]
            cm_total_full += temp_cm

        if 20 <= idx < 31:
            display_size = (1024, 1024)
            
            input_img_np = full_inputs[0].cpu().permute(1, 2, 0).numpy() 
            resized_input_wandb = cv2.resize(input_img_np, display_size, interpolation=cv2.INTER_AREA)
            
            gt_mask_np = full_masks[0].squeeze().cpu().numpy() 
            resized_gt_wandb = cv2.resize(gt_mask_np, display_size, interpolation=cv2.INTER_NEAREST)
            
            pred_mask_np = full_outputs_pred[0].squeeze().cpu().numpy()
            resized_pred_wandb = cv2.resize(pred_mask_np, display_size, interpolation=cv2.INTER_NEAREST)

            input_img_ref_full = wandb.Image(resized_input_wandb, caption="Full Input (Resized)")
            gt_img_ref_full = wandb.Image(resized_gt_wandb * 255, caption="Full GT (Resized)")
            pred_img_ref_full = wandb.Image(resized_pred_wandb * 255, caption="Full Prediction (Resized)")

            table_full.add_data(os.path.basename(original_filename), input_img_ref_full, gt_img_ref_full, pred_img_ref_full, f"{current_full_image_dice:.4f}")

        log_memory(f"After Full Image Test {idx+1}")

avg_test_dice_full = np.mean(full_image_dice_scores) if full_image_dice_scores else 0.0
print(f"\nAverage Test Dice on FULL IMAGES: {avg_test_dice_full:.4f}")
wandb.log({"test_dice_full_images_avg": avg_test_dice_full})

wandb.log({"Test Predictions (Full Images)": table_full})

print("\nFinal Confusion Matrix (Full Images Test Set):")
print(cm_total_full)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_total_full, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Background", "Tumor"],
            yticklabels=["Background", "Tumor"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Full Images Test Set)")
plt.tight_layout()
wandb.log({"Confusion Matrix (Full Images)": wandb.Image(plt)})
plt.close()

log_memory("Full Image Testing End")
wandb.finish()