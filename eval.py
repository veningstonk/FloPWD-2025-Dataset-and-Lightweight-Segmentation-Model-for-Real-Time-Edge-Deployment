import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from dataset import DalLake
from model_final2 import ENet
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())
# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
dal_lake_dir = r"D:\\Suhaib\\DalLake"
crop_h, crop_w = 720, 1280

batch_size = 2

output_dir = Path(r"D:\\Suhaib\\DalLake\\output(ENet-final2)")
model_ckpt = output_dir / "best_model.pth"

train_data = DalLake(root=dal_lake_dir, split='train', image_size=(crop_w, crop_h))
print("Computing training class distribution...")
train_pixel_counts = np.zeros(2)
for _, target, _ in train_data:
    target_np = np.asarray(target)
    for cls in range(2):
        train_pixel_counts[cls] += np.sum(target_np == cls)
train_percentages = train_pixel_counts / train_pixel_counts.sum() * 100
print(f"Training class distribution: {train_pixel_counts}")
print(f"Training class percentages: {train_percentages}")

if train_pixel_counts.sum() > 0:
    inverse_weights = 1.0 / (train_pixel_counts / train_pixel_counts.sum())
    class_weights = inverse_weights / (inverse_weights.max() / 2.0)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Computed class weights: {class_weights}")
else:
    class_weights = [0.05, 0.95]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"Using fallback class weights: {class_weights}")

# ------------------------------------------------------------------------------
# Load Validation Dataset
# ------------------------------------------------------------------------------
print("Loading validation dataset...")
val_data = DalLake(root=dal_lake_dir, split="val", image_size=(crop_w, crop_h))    #custom h, w    base w, h
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
print(f"Loaded {len(val_data)} validation samples.")

# ------------------------------------------------------------------------------
# Metric Calculation Function
# ------------------------------------------------------------------------------
def compute_metrics(preds, target, num_classes=2):
    preds = torch.argmax(preds, dim=1)

    pixel_acc = torch.zeros(num_classes, device=preds.device)
    intersection = torch.zeros(num_classes, device=preds.device)
    union = torch.zeros(num_classes, device=preds.device)

    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = target == cls

        correct = (pred_cls & (pred_cls == target_cls)).sum().float()
        total = target_cls.sum().float()
        pixel_acc[cls] = correct / total if total > 0 else 0

        intersection[cls] = (pred_cls & target_cls).sum().float()
        union[cls] = (pred_cls | target_cls).sum().float()

    mPA = pixel_acc.mean().item()
    mIoU = (intersection / union).nanmean().item()
    return mPA, mIoU, pixel_acc.cpu()

# ------------------------------------------------------------------------------
# Load Model and Weights
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ENet(in_channels=3, num_classes=2).to(device)

try:
    state_dict = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    print(f"Loaded model weights from {model_ckpt}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

model.eval()

# ------------------------------------------------------------------------------
# Loss Function (Using Class Weights)
# ------------------------------------------------------------------------------
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

# ------------------------------------------------------------------------------
# Run Evaluation
# ------------------------------------------------------------------------------
val_loss = 0.0
val_mPA_sum = 0.0
val_mIoU_sum = 0.0
per_class_acc_sum = torch.zeros(2)

start_time = time.time()

with torch.no_grad():
    for imgs, targets, _ in val_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)
        #outputs = outputs.permute(0, 1, 3, 2) # for custom
        loss = criterion(outputs, targets)
        mPA, mIoU, per_class_acc = compute_metrics(outputs, targets)

        val_loss += loss.item()
        val_mPA_sum += mPA
        val_mIoU_sum += mIoU
        per_class_acc_sum += per_class_acc

# ------------------------------------------------------------------------------
# Compute Averages
# ------------------------------------------------------------------------------
num_batches = len(val_loader)
avg_val_loss = val_loss / num_batches
avg_val_mPA = val_mPA_sum / num_batches
avg_val_mIoU = val_mIoU_sum / num_batches
avg_pc_acc = per_class_acc_sum / num_batches
elapsed_time = time.time() - start_time
fps = len(val_data) / elapsed_time

# ------------------------------------------------------------------------------
# Print and Save Results
# ------------------------------------------------------------------------------
print("\n──── Final Validation Results ────")
print(f"Loss         : {avg_val_loss:.4f}")
print(f"mPA          : {avg_val_mPA:.4f}")
print(f"mIoU         : {avg_val_mIoU:.4f}")
print(f"Acc (cls0)   : {avg_pc_acc[0]:.4f}")
print(f"Acc (cls1)   : {avg_pc_acc[1]:.4f}")
print(f"FPS          : {fps:.2f}")

# Save to CSV
eval_csv_path = output_dir / "evaluation_metrics2.csv"
pd.DataFrame({
    "loss": [avg_val_loss],
    "mPA": [avg_val_mPA],
    "mIoU": [avg_val_mIoU],
    "acc_cls0": [avg_pc_acc[0].item()],
    "acc_cls1": [avg_pc_acc[1].item()],
    "fps": [fps]
}).to_csv(eval_csv_path, index=False)
print(f"\nSaved evaluation metrics to: {eval_csv_path}")
