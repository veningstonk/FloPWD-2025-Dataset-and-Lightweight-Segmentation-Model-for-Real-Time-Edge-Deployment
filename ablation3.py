import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time
from thop import profile  # type: ignore
from sklearn.model_selection import KFold # type: ignore

from dataset import DalLake
from model3 import ENet

# Define dataset path and parameters
dal_lake_dir = "D:\\Suhaib\\DalLake"
crop_h, crop_w = 720, 1280
image_size = (crop_w, crop_h)
batch_size = 2
epochs = 10
learning_rate = 0.001
n_splits = 3  # Number of folds for cross-validation

# Directories for saving
output_dir = Path("D:\\Suhaib\\DalLake\\ablation3")
output_dir.mkdir(exist_ok=True)

# Ablation study configurations
ablation_configs = [
    {"component": "dropout", "variation": "0.0", "dropout_probs": [0.0, 0.0, 0.0, 0.0, 0.0]},
    {"component": "dropout", "variation": "0.05", "dropout_probs": [0.05, 0.05, 0.05, 0.05, 0.05]},
    {"component": "dropout", "variation": "0.2", "dropout_probs": [0.2, 0.2, 0.2, 0.2, 0.2]},
    {"component": "dilation", "variation": "no_dilation", "dilations_stage2": [1, 1, 1], "dilations_stage3": [1, 1, 1]},
    {"component": "dilation", "variation": "reduced", "dilations_stage2": [2, 2, 4], "dilations_stage3": [2, 2, 4, 8]},
    {"component": "dilation", "variation": "aggressive", "dilations_stage2": [4, 8, 16], "dilations_stage3": [4, 8, 16, 32]},
    {"component": "asymmetric", "variation": "no_asymmetric", "use_asymmetric": False},
    {"component": "blocks", "variation": "reduced", "blocks": [3, 5, 5, 1, 1]},
    {"component": "blocks", "variation": "increased", "blocks": [6, 10, 10, 3, 2]},
    {"component": "blocks", "variation": "minimal", "blocks": [2, 3, 3, 1, 1]},
    {"component": "internal_channels", "variation": "double", "channel_factor": 2},
    {"component": "internal_channels", "variation": "middle", "channel_factor": 3},
    {"component": "internal_channels", "variation": "halve", "channel_factor": 8},
    {"component": "default", "variation": "base", 
     "dropout_probs": [0.01, 0.1, 0.1, 0.1, 0.1],
     "dilations_stage2": [2, 4, 8],
     "dilations_stage3": [2, 4, 8, 16],
     "use_asymmetric": True,
     "blocks": [5, 8, 8, 2, 1],
     "channel_factor": 4}
]

# Base configuration
base_config = {
    "dropout_probs": [0.01, 0.1, 0.1, 0.1, 0.1],
    "dilations_stage2": [2, 4, 8],
    "dilations_stage3": [2, 4, 8, 16],
    "use_asymmetric": True,
    "blocks": [5, 8, 8, 2, 1],
    "channel_factor": 4
}

# Load full dataset (combine train and val for cross-validation)
print("Loading dataset...")
full_data = DalLake(root=dal_lake_dir, split='train', image_size=image_size)  # Adjust split as needed
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Class weights
class_weights = [0.05, 0.95]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Compute metrics function
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

# Plot and save metrics
def plot_metrics(metrics, epoch, plot_dir, fold):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss (Fold {fold})')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(metrics['epoch'], metrics['train_mPA'], label='Train mPA')
    plt.plot(metrics['epoch'], metrics['val_mPA'], label='Val mPA')
    plt.xlabel('Epoch')
    plt.ylabel('mPA')
    plt.title(f'Training and Validation mPA (Fold {fold})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"epoch_{epoch}_fold_{fold}.png")
    plt.close()

# Store results for all configurations
all_results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ablation study loop with cross-validation
for config in ablation_configs:
    component = config["component"]
    variation = config["variation"]
    print(f"Running ablation study: {component} - {variation}")
    
    # Set up directories
    exp_dir = output_dir / f"{component}_{variation}"
    exp_dir.mkdir(exist_ok=True)
    
    # Cross-validation loop
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_data)))):
        print(f"  Fold {fold + 1}/{n_splits}")
        
        # Create train and val subsets
        train_subset = Subset(full_data, train_idx)
        val_subset = Subset(full_data, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Set up fold-specific directories
        fold_dir = exp_dir / f"fold_{fold + 1}"
        model_save_path = fold_dir / "best_model.pth"
        pred_save_dir = fold_dir / "predictions"
        csv_dir = fold_dir / "metrics"
        plot_dir = fold_dir / "plots"
        fold_dir.mkdir(exist_ok=True)
        pred_save_dir.mkdir(exist_ok=True)
        csv_dir.mkdir(exist_ok=True)
        plot_dir.mkdir(exist_ok=True)
        
        # Merge base and variation config
        exp_config = base_config.copy()
        exp_config.update(config)
        
        # Initialize model
        model = ENet(
            in_channels=3, num_classes=2,
            dropout_probs=exp_config["dropout_probs"],
            dilations_stage2=exp_config["dilations_stage2"],
            dilations_stage3=exp_config["dilations_stage3"],
            use_asymmetric=exp_config["use_asymmetric"],
            blocks=exp_config["blocks"],
            channel_factor=exp_config["channel_factor"]
        ).to(device)
        
        # Estimate FLOPS
        dummy_input = torch.randn(batch_size, 3, crop_h, crop_w).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"  Estimated FLOPS: {flops / 1e9:.2f} GFLOPS, Parameters: {params / 1e6:.2f} M")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Metrics tracking
        metrics = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'train_mPA': [], 'val_mPA': [], 'train_mIoU': [], 'val_mIoU': [],
            'train_fps': [], 'val_fps': []
        }
        best_val_loss = float('inf')
        best_val_mIoU = 0.0
        best_metrics = None
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_mPA_sum = 0.0
            train_mIoU_sum = 0.0
            start_time = time.time()
            
            for img, target, names in train_loader:
                img, target = img.to(device), target.to(device)
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
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_mPA_sum = 0.0
            val_mIoU_sum = 0.0
            start_time = time.time()
            
            with torch.no_grad():
                for img, target, names in val_loader:
                    img, target = img.to(device), target.to(device)
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
            
            # Save metrics
            metrics['epoch'].append(epoch + 1)
            metrics['train_loss'].append(avg_train_loss)
            metrics['val_loss'].append(avg_val_loss)
            metrics['train_mPA'].append(avg_train_mPA)
            metrics['val_mPA'].append(avg_val_mPA)
            metrics['train_mIoU'].append(avg_train_mIoU)
            metrics['val_mIoU'].append(avg_val_mIoU)
            metrics['train_fps'].append(train_fps)
            metrics['val_fps'].append(val_fps)
            
            # Save model if validation loss improves
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_save_path)
            
            # Track best val_mIoU and corresponding metrics
            if avg_val_mIoU > best_val_mIoU:
                best_val_mIoU = avg_val_mIoU
                best_metrics = {
                    'epoch': epoch + 1,
                    'val_loss': avg_val_loss,
                    'val_mPA': avg_val_mPA,
                    'val_mIoU': avg_val_mIoU,
                    'val_fps': val_fps
                }
            
            # Plot metrics
            plot_metrics(metrics, epoch + 1, plot_dir, fold + 1)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(csv_dir / "training_metrics.csv", index=False)
        print(f"  Saved metrics for {component}_{variation}_fold_{fold + 1} to {csv_dir}")
        
        # Store best metrics for this fold
        fold_results.append({
            'fold': fold + 1,
            'val_mIoU': best_metrics['val_mIoU'],
            'val_fps': best_metrics['val_fps'],
            'val_loss': best_metrics['val_loss'],
            'val_mPA': best_metrics['val_mPA'],
            'epoch': best_metrics['epoch'],
            'flops': flops / 1e9,
            'params': params / 1e6
        })
    
    # Average metrics across folds
    avg_val_mIoU = np.mean([res['val_mIoU'] for res in fold_results])
    avg_val_fps = np.mean([res['val_fps'] for res in fold_results])
    avg_val_loss = np.mean([res['val_loss'] for res in fold_results])
    avg_val_mPA = np.mean([res['val_mPA'] for res in fold_results])
    
    # Store averaged results for this configuration
    all_results.append({
        'component': component,
        'variation': variation,
        'val_mIoU': avg_val_mIoU,
        'val_fps': avg_val_fps,
        'val_loss': avg_val_loss,
        'val_mPA': avg_val_mPA,
        'flops': fold_results[0]['flops'],  # FLOPS and params are same across folds
        'params': fold_results[0]['params']
    })
    
    # Save fold results to CSV
    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(exp_dir / "fold_metrics.csv", index=False)
    print(f"Saved fold metrics for {component}_{variation} to {exp_dir / 'fold_metrics.csv'}")

# After all configurations, find the best one
print("\n=== Ablation Study Summary ===")
# Normalize val_fps for scoring (assuming a reasonable range, e.g., 0 to 50 FPS)
max_fps = max([res['val_fps'] for res in all_results]) if all_results else 50.0
min_fps = min([res['val_fps'] for res in all_results]) if all_results else 0.0
fps_range = max_fps - min_fps if max_fps != min_fps else 1.0

# Compute a score: 0.7 * val_mIoU + 0.3 * normalized_val_fps
for res in all_results:
    normalized_fps = (res['val_fps'] - min_fps) / fps_range
    res['score'] = 0.7 * res['val_mIoU'] + 0.3 * normalized_fps

# Find the best configuration
best_config = max(all_results, key=lambda x: x['score'])
print("Best Configuration:")
print(f"Component: {best_config['component']}")
print(f"Variation: {best_config['variation']}")
print(f"Average val_mIoU: {best_config['val_mIoU']:.4f}")
print(f"Average val_fps: {best_config['val_fps']:.4f}")
print(f"Score: {best_config['score']:.4f}")
print(f"Additional Metrics: val_loss={best_config['val_loss']:.4f}, val_mPA={best_config['val_mPA']:.4f}")
print(f"FLOPS: {best_config['flops']:.2f} GFLOPS, Parameters: {best_config['params']:.2f} M")

# Save summary to CSV
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)
print(f"Saved ablation study summary to {output_dir / 'ablation_summary.csv'}")