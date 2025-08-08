import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torch
from pathlib import Path
from torchvision import transforms
import albumentations as A  # type: ignore

from dataset import DalLake
from model_custom_noAsym import CustomENet

# Define paths and parameters
dal_lake_dir = "D:\\Suhaib\\DalLake"
crop_h, crop_w = 720, 1280
image_size = (crop_h, crop_w)  # (height, width)
batch_size = 8
epochs = 200
learning_rate = 0.001

model_save_path = Path(dal_lake_dir) / "output(ENet-custom 200batch8)" / "best_model.pth"
pred_save_dir = Path(dal_lake_dir) / "output(ENet-custom 200batch8)" / "predictions_new"
pred_save_dir.mkdir(exist_ok=True)
image_size = (720, 1280)  # Height, Width (matching training code)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Data path: {dal_lake_dir}")
print(f"Image size: {crop_h}x{crop_w}, Batch size: {batch_size}, Epochs: {epochs}")

print("Loading datasets...")
try:
    train_data = DalLake(root=dal_lake_dir, split='train', image_size=(crop_h, crop_w))
    val_data = DalLake(root=dal_lake_dir, split='val', image_size=(crop_h, crop_w))
except Exception as e:
    print(f"Error loading datasets: {e}")
    raise

print(f"Train dataset size: {len(train_data)}")
print(f"Val dataset size: {len(val_data)}")


val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

print(f"DataLoader sizes: Val={len(val_loader)}")




# Load the model
print("Loading CustomENet model...")
model = CustomENet(in_channels=3, num_classes=2)
try:
    state_dict = torch.load(model_save_path, map_location=device, weights_only=True)
    # Filter out unexpected keys
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
except Exception as e:
    print(f"Error loading model: {e}")
    raise
model = model.to(device)
model.eval()
print(f"Loaded model from {model_save_path}")

# Define preprocessing transforms (same as used in DalLake dataset)
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resize_transform = A.Compose([
    A.Resize(height=image_size[0], width=image_size[1]),
])

# Function to preprocess a single image

def save_predicted_images(model, val_loader, save_dir, target_size=(1280, 720)):
    model.eval()
    with torch.no_grad():
        for i, (img, target, names) in enumerate(val_loader):
            img = img.to(device)
            output = model(img)
            preds = torch.argmax(output, dim=1)
            
            for j in range(img.shape[0]):
                pred = preds[j].cpu().numpy()  # Shape: [720, 1280]
                print(f"Processing image {names[j]}: pred shape: {pred.shape}")
                
                # Create output image with correct dimensions (720, 1280, 3)
                pred_img = np.zeros((720, 1280, 3), dtype=np.uint8)
                non_plastic_mask = (pred == 0)  # Shape: [720, 1280]
                plastic_mask = (pred == 1)      # Shape: [720, 1280]
                
                # Apply masks directly to the correct dimensions
                pred_img[non_plastic_mask] = [0, 0, 0]  # Black for non-plastic
                pred_img[plastic_mask] = [255, 0, 0]    # Red for plastic
                
                # Convert to PIL Image and resize to target size
                pred_img = Image.fromarray(pred_img)
                pred_img = pred_img.resize(target_size, Image.NEAREST)
                pred_img.save(save_dir / f"{names[j]}_pred.png")
                print(f"Saved prediction for {names[j]} at {save_dir / f'{names[j]}_pred.png'}")

print("Generating predictions for all validation images...")
save_predicted_images(model, val_loader, pred_save_dir)
print(f"Saved all validation predictions to {pred_save_dir}")
