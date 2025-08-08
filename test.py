import os
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from torchvision import transforms
import albumentations as A  # type: ignore

from dataset import DalLake
from model_custom_noAsym import CustomENet

# Define paths and parameters
dal_lake_dir = "D:\\Suhaib\\DalLake"
test_dir = Path(dal_lake_dir) / "test"
model_save_path = Path(dal_lake_dir) / "output(ENet-custom 200batch8)" / "best_model.pth"
pred_save_dir = Path(dal_lake_dir) / "output(ENet-custom 200batch8)" / "test_predictions"
pred_save_dir.mkdir(exist_ok=True)
image_size = (720, 1280)  # Height, Width (matching training code)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
def preprocess_image(img_path):
    # Load and convert image
    image = Image.open(img_path).convert('RGB')
    image = np.array(image)
    
    # Apply resize transform
    transformed = resize_transform(image=image)
    image = transformed['image']
    
    # Convert to tensor and normalize
    image = to_tensor(image)
    image = normalize(image)
    
    # Add batch dimension
    image = image.unsqueeze(0)  # Shape: [1, 3, 720, 1280]
    return image

# Function to save concatenated image (original + prediction)
def save_predicted_image(pred, save_path, original_img_path, target_size=(1280, 720)):
    # Create prediction image (initialize with shape [height, width, 3])
    pred_img = np.zeros((720, 1280, 3), dtype=np.uint8)  # Match pred's shape [720, 1280]
    pred_img[pred == 0] = [0, 0, 0]    # Non-plastic: black
    pred_img[pred == 1] = [255, 0, 0]  # Plastic: red
    pred_img = Image.fromarray(pred_img)
    pred_img = pred_img.resize(target_size, Image.NEAREST)  # Resize to (1280, 720)

    # Load and resize original image
    original_img = Image.open(original_img_path).convert('RGB')
    original_img = original_img.resize(target_size, Image.BILINEAR)

    # Concatenate images side by side
    concat_img = Image.new('RGB', (target_size[0] * 2, target_size[1]))
    concat_img.paste(original_img, (0, 0))
    concat_img.paste(pred_img, (target_size[0], 0))

    # Save concatenated image
    concat_img.save(save_path)
    print(f"Saved concatenated image to {save_path}")

# Process all images in the test folder
print(f"Processing images from {test_dir}...")
for img_file in os.listdir(test_dir):
    if not img_file.endswith(('.jpg', '.png')) or any(s in img_file for s in ['colorized_', 'instanceIds_', '_mask', '_trainLabelId']):
        continue
    
    img_path = os.path.join(test_dir, img_file)
    print(f"Processing {img_file}...")
    
    # Preprocess the image
    image = preprocess_image(img_path)
    image = image.to(device)
    
    # Generate prediction
    with torch.no_grad():
        output = model(image)  # Shape: [1, 2, 720, 1280]
        pred = torch.argmax(output, dim=1)  # Shape: [1, 720, 1280]
        pred = pred.squeeze(0).cpu().numpy()  # Shape: [720, 1280]
    
    # Save the concatenated image
    save_path = pred_save_dir / f"{os.path.splitext(img_file)[0]}_concat.png"
    save_predicted_image(pred, save_path, img_path)

print(f"All concatenated predictions saved to {pred_save_dir}")