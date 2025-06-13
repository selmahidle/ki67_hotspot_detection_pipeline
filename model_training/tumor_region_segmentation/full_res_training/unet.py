"""
UNet training script for semantic segmentation of tumor in 4096x4096 WSI patches
"""

from preprocessing import get_dataloaders
import torch
import wandb
import argparse
import numpy as np
from monai.networks.nets import UNet
import torch.optim as optim
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
import torch.nn as nn
import random
import cv2
from sklearn.metrics import confusion_matrix
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    num_res_units = args.num_res_units,
).to(device)

wandb.watch(model, log='all', log_freq=100)

optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
loss_function = TverskyLoss(sigmoid=True) 
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

        if avg_dice_value > best_metric:
            best_metric = avg_dice_value
            best_metric_epoch = epoch + 1
            checkpoint_filename = f"/cluster/home/selmahi/mib_pipeline/mib_pipeline_scripts/checkpoints/tumor_unet/{args.slurm_job_id}_best_tumor_semseg_unet_model_monai.pth"
            torch.save(model.state_dict(), checkpoint_filename)

del val_inputs, val_masks, val_outputs, inputs, masks, outputs, loss, val_loss
torch.cuda.empty_cache()

print(f"Training Complete. Best Dice Score: {best_metric:.4f} at Epoch: {best_metric_epoch}")





"""
Testing the model
"""

checkpoint_filename = f"/cluster/home/selmahi/mib_pipeline/mib_pipeline_scripts/checkpoints/tumor_unet/{args.slurm_job_id}_best_tumor_semseg_unet_model_monai.pth"
if os.path.exists(checkpoint_filename) and best_metric > -1 :
    print(f"\nLoading best model from epoch {best_metric_epoch} for testing: {checkpoint_filename}")
    model.load_state_dict(torch.load(checkpoint_filename, map_location=device))
else:
    print("\nWARNING: No best model checkpoint found or training did not improve. Testing with the model from the last epoch.")
model.eval() 


print("\nTesting the model...")
cm_total = np.zeros((2, 2), dtype=int)

with torch.no_grad():
    test_dice_scores = []
    columns = ["Input Image", "Ground Truth", "Prediction", "Dice Score"]
    table = wandb.Table(columns=columns)
    for idx, test_data in enumerate(test_loader):
        test_inputs, test_masks = test_data["image"].to(device), test_data["mask"].to(device)
        test_outputs = post_trans(model(test_inputs))

        dice_metric(y_pred=test_outputs, y=test_masks)
        ex_img_dice_score = dice_metric.aggregate().item()
        test_dice_scores.append(ex_img_dice_score)
        dice_metric.reset()

        preds_flat = test_outputs.cpu().numpy().astype(int).flatten()
        labels_flat = test_masks.cpu().numpy().astype(int).flatten()

        cm_batch = confusion_matrix(labels_flat, preds_flat, labels=[0, 1])
        cm_total += cm_batch

        if 10 <= idx < 21:
            input_image = test_inputs[0].cpu().permute(1, 2, 0).numpy()
            ground_truth = test_masks[0].squeeze().cpu().numpy()
            prediction = test_outputs[0].squeeze().cpu().numpy()

            gt_bin = torch.tensor(ground_truth > 0.5).float()
            pred_bin = torch.tensor(prediction > 0.5).float()

            intersection = (pred_bin * gt_bin).sum()
            dice = (2. * intersection) / (pred_bin.sum() + gt_bin.sum() + 1e-8)
            dice_score = dice.item()

            resized_input = cv2.resize(input_image, (512, 512), interpolation=cv2.INTER_AREA)
            resized_gt = cv2.resize(ground_truth, (512, 512), interpolation=cv2.INTER_NEAREST)
            resized_pred = cv2.resize(prediction, (512, 512), interpolation=cv2.INTER_NEAREST)

            input_img_ref = wandb.Image(resized_input, caption="Input Image")
            gt_img_ref = wandb.Image(resized_gt, caption="Ground Truth")
            pred_img_ref = wandb.Image(resized_pred, caption="Prediction")

            table.add_data(input_img_ref, gt_img_ref, pred_img_ref, f"{dice_score:.4f}")

    wandb.log({"Test Predictions": table})
    avg_test_dice_value = sum(test_dice_scores) / len(test_dice_scores)
    print(f"Test Dice: {avg_test_dice_value:.4f}")
    wandb.log({"test_dice": avg_test_dice_value})

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues", xticklabels=["Background", "Tumor"], yticklabels=["Background", "Tumor"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    wandb.log({"Confusion Matrix": wandb.Image(plt)})

wandb.finish()
