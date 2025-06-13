"""
ResNet18 + DeepLabV3Plus for Tumor Segmentation in WSI Patches
"""

import os
import warnings
import random
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import segmentation_models_pytorch as smp
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from sklearn.metrics import confusion_matrix
import wandb
from preprocessing import get_dataloaders


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--init_lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_groups_gn", type=int, default=8,
                    help="Number of groups for GroupNorm")
parser.add_argument("--slurm_job_id", type=int, default=None)
parser.add_argument("--comments", type=str, default=None)
args = parser.parse_args()

run = wandb.init(
    entity="selma_mib",
    project="Tumor Segmentation",
    config={
        "comments": args.comments,
        "init_learning_rate": args.init_lr,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
        "model_architecture": "DeepLabV3+ ResNet18 (scratch + GN)",
        "normalization": "GroupNorm",
        "num_groups_gn": args.num_groups_gn,
        "loss_function": "Tversky",
        "slurm_job_id": args.slurm_job_id,
    },
)


def replace_bn_with_gn(module: nn.Module, num_groups: int) -> int:
    """Recursively replace all BatchNorm2d layers with GroupNorm."""
    n_replaced = 0
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            n_channels = child.num_features
            if n_channels % num_groups == 0:
                setattr(module, name, nn.GroupNorm(num_groups, n_channels))
                n_replaced += 1
            else:
                warnings.warn(
                    f"Skipping BN→GN for layer '{name}': {n_channels} channels not divisible "
                    f"by num_groups={num_groups}."
                )
        else:
            n_replaced += replace_bn_with_gn(child, num_groups)
    return n_replaced


train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.DeepLabV3Plus(
    encoder_name="resnet18",
    encoder_weights=None, 
    in_channels=3,
    classes=1,
    activation=None,
)

replaced = replace_bn_with_gn(model, args.num_groups_gn)
model = model.to(DEVICE)

loss_fn = TverskyLoss(sigmoid=True)
optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
dice_metric = DiceMetric(include_background=True)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

wandb.watch(model, log="all", log_freq=100)


best_dice = -1.0
best_epoch = -1
ckpt_dir = "/cluster/home/selmahi/mib_pipeline_scripts/checkpoints/tumor_deeplabresnet_scratch"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(
    ckpt_dir,
    f"{args.slurm_job_id}_best_semseg_deeplabv3resnet_model.pth",
)

print("Starting training…")
for epoch in range(1, args.num_epochs + 1):
    print(f"\n—— Epoch {epoch}/{args.num_epochs} ——")
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        imgs = batch["image"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        preds = model(imgs)
        loss = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    wandb.log({"train_loss": train_loss, "epoch": epoch})
    print(f"Train loss: {train_loss:.4f}")

    model.eval()
    dice_scores, val_losses = [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)
            preds = model(imgs)
            val_loss = loss_fn(preds, masks)
            val_losses.append(val_loss.item())

            preds_bin = post_trans(preds)
            dice_metric(y_pred=preds_bin, y=masks)
            dice_scores.append(dice_metric.aggregate().item())
            dice_metric.reset()

    val_loss_mean = np.mean(val_losses)
    val_dice_mean = np.mean(dice_scores)
    wandb.log({"val_loss": val_loss_mean, "val_dice": val_dice_mean, "epoch": epoch})
    print(f"Val Dice: {val_dice_mean:.4f} | Val loss: {val_loss_mean:.4f}")

    if val_dice_mean > best_dice:
        best_dice = val_dice_mean
        best_epoch = epoch
        torch.save(model.state_dict(), ckpt_path)
        print(f"New best model saved to {ckpt_path}")

print(f"\nTraining finished. Best Dice {best_dice:.4f} at epoch {best_epoch}.")


print("\nLoading best model for testing…")
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.eval()

cm_total = np.zeros((2, 2), dtype=int)
test_dice_scores = []
columns = ["Input", "Ground Truth", "Prediction", "Dice"]
wandb_table = wandb.Table(columns=columns)

with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        imgs = batch["image"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)
        preds = post_trans(model(imgs))

        dice_metric(y_pred=preds, y=masks)
        dice_val = dice_metric.aggregate().item()
        test_dice_scores.append(dice_val)
        dice_metric.reset()

        preds_np = preds.cpu().numpy().astype(int).flatten()
        masks_np = masks.cpu().numpy().astype(int).flatten()
        cm_total += confusion_matrix(masks_np, preds_np, labels=[0, 1])

        if idx < 10:
            img_vis = imgs[0].cpu().permute(1, 2, 0).numpy()
            mask_vis = masks[0].squeeze().cpu().numpy()
            pred_vis = preds[0].squeeze().cpu().numpy()
            wandb_table.add_data(
                wandb.Image(img_vis),
                wandb.Image(mask_vis),
                wandb.Image(pred_vis),
                f"{dice_val:.4f}",
            )

wandb.log({"Test predictions": wandb_table})
print(f"Test Dice: {np.mean(test_dice_scores):.4f}")

plt.figure(figsize=(5, 4))
sns.heatmap(cm_total, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Background", "Tumor"],
            yticklabels=["Background", "Tumor"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
wandb.log({"Confusion Matrix": wandb.Image(plt)})

wandb.finish()


