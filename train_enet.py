import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time
from thop import profile # type: ignore

from dataset import DalLake
from model2 import ENet

dal_lake_dir = "D:\\Suhaib\\DalLake"
crop_h, crop_w = 720, 1280
image_size = (crop_h, crop_w)  # PyTorch expects (height, width)
batch_size = 8
epochs = 50
learning_rate = 0.001

output_dir = Path("D:\\Suhaib\\DalLake\\output(ENet-batch8)")
output_dir.mkdir(exist_ok=True)
model_save_path = output_dir / "best_model.pth"
pred_save_dir = output_dir / "predictions"
pred_save_dir.mkdir(exist_ok=True)
csv_dir = output_dir / "metrics"
csv_dir.mkdir(exist_ok=True)
combined_csv_path = output_dir / "training_metrics.csv"
plot_dir = output_dir / "plots"
plot_dir.mkdir(exist_ok=True)

print(f"Data path: {dal_lake_dir}")
print(f"Image size: {crop_h}x{crop_w}, Batch size: {batch_size}, Epochs: {epochs}")

print("Loading datasets...")
try:
    train_data = DalLake(root=dal_lake_dir, split='train', image_size=(crop_w, crop_h))
    val_data = DalLake(root=dal_lake_dir, split='val', image_size=(crop_w, crop_h))
except Exception as e:
    print(f"Error loading datasets: {e}")
    raise

print(f"Train dataset size: {len(train_data)}")
print(f"Val dataset size: {len(val_data)}")

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

print("Computing validation class distribution...")
val_pixel_counts = np.zeros(2)
for _, target, _ in val_data:
    target_np = np.asarray(target)
    for cls in range(2):
        val_pixel_counts[cls] += np.sum(target_np == cls)
val_percentages = val_pixel_counts / val_pixel_counts.sum() * 100
print(f"Validation class distribution: {val_pixel_counts}")
print(f"Validation class percentages: {val_percentages}")

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

print(f"DataLoader sizes: Train={len(train_loader)}, Val={len(val_loader)}")

print("Initializing model...")
model = ENet(in_channels=3, num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model initialized on: {device}")

print("Estimating FLOPS...")
dummy_input = torch.randn(batch_size, 3, crop_h, crop_w).to(device)
flops, params = profile(model, inputs=(dummy_input,), verbose=False)
print(f"Estimated FLOPS: {flops / 1e9:.2f} GFLOPS")
print(f"Estimated Parameters: {params / 1e6:.2f} M")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

metrics = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'train_mPA': [],
    'val_mPA': [],
    'train_mIoU': [],
    'val_mIoU': [],
    'train_fps': [],
    'val_fps': []
}
best_val_loss = float('inf')

def compute_metrics(preds, target, num_classes=2):
    preds = torch.argmax(preds, dim=1)
    pixel_acc = torch.zeros(num_classes, device=preds.device)
    total_pixels = torch.zeros(num_classes, device=preds.device)
    intersection = torch.zeros(num_classes, device=preds.device)
    union = torch.zeros(num_classes, device=preds.device)
    
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (target == cls)
        correct = (pred_cls & (pred_cls == target_cls)).sum().float()
        total = target_cls.sum().float()
        pixel_acc[cls] = correct / total if total > 0 else 0
        total_pixels[cls] = total
        intersection[cls] = (pred_cls & target_cls).sum().float()
        union[cls] = (pred_cls | target_cls).sum().float()
    
    mPA = pixel_acc.mean().item()
    mIoU = (intersection / union).nanmean().item()
    return mPA, mIoU

def save_predicted_images(model, val_loader, save_dir, target_size=(1280, 720)):
    model.eval()
    with torch.no_grad():
        for i, (img, target, names) in enumerate(val_loader):
            img = img.to(device)
            output = model(img)
            preds = torch.argmax(output, dim=1)
            
            for j in range(img.shape[0]):
                pred = preds[j].cpu().numpy()  # Shape: [1280, 720]
                print(f"Processing image {names[j]}: pred shape: {pred.shape}")
                pred_img = np.zeros((1280, 720, 3), dtype=np.uint8)
                non_plastic_mask = (pred == 0)
                plastic_mask = (pred == 1)
                pred_img[non_plastic_mask] = [0, 0, 0]
                pred_img[plastic_mask] = [255, 0, 0]
                pred_img = Image.fromarray(pred_img)
                pred_img = pred_img.resize(target_size, Image.NEAREST)
                pred_img.save(save_dir / f"{names[j]}_pred.png")
                print(f"Saved prediction for {names[j]} at {save_dir / f'{names[j]}_pred.png'}")

def plot_metrics(metrics, epoch, plot_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(metrics['epoch'], metrics['train_mPA'], label='Train mPA')
    plt.plot(metrics['epoch'], metrics['val_mPA'], label='Val mPA')
    plt.xlabel('Epoch')
    plt.ylabel('mPA')
    plt.title('Training and Validation mPA')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"training_plots_epoch_{epoch}.png")
    plt.close()

def save_individual_metrics(metrics, csv_dir):
    for metric_name in metrics.keys():
        if metric_name == 'epoch':
            continue
        df = pd.DataFrame({
            'epoch': metrics['epoch'],
            metric_name: metrics[metric_name]
        })
        df.to_csv(csv_dir / f"{metric_name}.csv", index=False)

print("Starting training...")
try:
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_mPA_sum = 0.0
        train_mIoU_sum = 0.0
        start_time = time.time()
        
        for batch_idx, (img, target, names) in enumerate(train_loader):
            print(f"Processing batch {batch_idx+1}/{len(train_loader)} in epoch {epoch+1}")
            print(f"Image shape: {img.shape}, Target shape: {target.shape}")
            img = img.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            mPA, mIoU = compute_metrics(output, target)
            train_mPA_sum += mPA
            train_mIoU_sum += mIoU
        
        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mPA = train_mPA_sum / len(train_loader)
        avg_train_mIoU = train_mIoU_sum / len(train_loader)
        train_fps = len(train_loader) * batch_size / epoch_time if epoch_time > 0 else 0
        
        model.eval()
        val_loss = 0.0
        val_mPA_sum = 0.0
        val_mIoU_sum = 0.0
        start_time = time.time()
        
        with torch.no_grad():
            for img, target, names in val_loader:
                img = img.to(device)
                target = target.to(device)
                output = model(img)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                mPA, mIoU = compute_metrics(output, target)
                val_mPA_sum += mPA
                val_mIoU_sum += mIoU
        
        epoch_time = time.time() - start_time
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mPA = val_mPA_sum / len(val_loader)
        avg_val_mIoU = val_mIoU_sum / len(val_loader)
        val_fps = len(val_loader) * batch_size / epoch_time if epoch_time > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train mPA: {avg_train_mPA:.4f}, Train mIoU: {avg_train_mIoU:.4f}, Train FPS: {train_fps:.2f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val mPA: {avg_val_mPA:.4f}, Val mIoU: {avg_val_mIoU:.4f}, Val FPS: {val_fps:.2f}")
        
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(avg_val_loss)
        metrics['train_mPA'].append(avg_train_mPA)
        metrics['val_mPA'].append(avg_val_mPA)
        metrics['train_mIoU'].append(avg_train_mIoU)
        metrics['val_mIoU'].append(avg_val_mIoU)
        metrics['train_fps'].append(train_fps)
        metrics['val_fps'].append(val_fps)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")

        plot_metrics(metrics, epoch + 1, plot_dir)
        save_individual_metrics(metrics, csv_dir)

except Exception as e:
    print(f"Training failed with error: {e}")
    raise

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(combined_csv_path, index=False)
print(f"Saved combined training metrics to {combined_csv_path}")
print(f"Saved individual metric CSVs to {csv_dir}")
print(f"Saved training plots to {plot_dir}")

print("Loading best model for final validation predictions...")
try:
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
except Exception as e:
    print(f"Error loading best model: {e}")
    raise
print("Generating predictions for all validation images...")
save_predicted_images(model, val_loader, pred_save_dir)
print(f"Saved all validation predictions to {pred_save_dir}")