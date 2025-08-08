import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A # type: ignore

# Palette for colorized predictions (unchanged)
palette = [
    0, 0, 0,      # non-plastic (black)
    255, 0, 0     # plastic (red)
]

class DalLake(Dataset):
    def __init__(self, root, split='train', image_size=(720, 1280)):
        super(DalLake, self).__init__()
        self.root = root
        self.split = split
        self.image_size = image_size
        self.images = []
        self.labels = []
        self.names = []
        
        # Define paths
        split_dir = os.path.join(root, split)
        label_dir = os.path.join(root, 'testlabel') if split == 'test' else split_dir
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Directory {split_dir} does not exist.")
        if not os.path.exists(label_dir):
            raise ValueError(f"Label directory {label_dir} does not exist.")
        
        # Load original images
        total_images = 0
        for img_file in os.listdir(split_dir):
            if (img_file.endswith(('.jpg', '.png')) and
                not any(s in img_file for s in ['colorized_', 'instanceIds_', '_mask', '_trainLabelId'])):
                total_images += 1
                img_path = os.path.join(split_dir, img_file)
                base_name = os.path.splitext(img_file)[0]
                label_file = f"{base_name}_trainLabelId.png"
                label_path = os.path.join(label_dir, label_file)
                
                if os.path.exists(label_path):
                    self.images.append(img_path)
                    self.labels.append(label_path)
                    self.names.append(img_file)
                else:
                    print(f"Warning: Label file {label_path} not found for {img_file}, skipping.")
        
        if not self.images:
            raise ValueError(f"No valid image-label pairs found in {split_dir}.")
        
        print(f"Total images found in {split} split: {total_images}")
        print(f"Loaded {len(self.images)} images for {split} split (after filtering).")
        
        # Define transforms
        if split == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.3),
                A.Resize(height=image_size[0], width=image_size[1]),
            ], additional_targets={'label': 'mask'})
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
            ], additional_targets={'label': 'mask'})
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        label_path = self.labels[index]
        name = self.names[index]
        
        # Load and convert images
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # Ensure label is grayscale
        
        # Convert to numpy arrays
        image = np.array(image)
        label = np.array(label)
        
        # Apply transformations
        transformed = self.transform(image=image, label=label)
        image = transformed['image']
        label = transformed['label']
        
        # Convert to tensors
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        # Ensure label is binarized (0 and 1)
        label = torch.from_numpy(label).long()
        label = (label > 0).long()  # Binarize to 0 and 1
        
        # Verify shapes (using self.image_size)
        if image.shape[1:] != (self.image_size[0], self.image_size[1]):
            raise ValueError(f"Image shape {image.shape} does not match expected {self.image_size}")
        if label.shape != (self.image_size[0], self.image_size[1]):
            raise ValueError(f"Label shape {label.shape} does not match expected {self.image_size}")
        
        return image, label, name