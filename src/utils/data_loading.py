import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import json
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from PIL import Image

def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized labels"""
    elem = batch[0]
    batch_dict = {}
    
    # Handle images normally (they should be same size after resizing)
    if 'image' in elem:
        batch_dict['image'] = default_collate([d['image'] for d in batch])
    
    # Handle labels (variable length)
    if 'labels' in elem:
        batch_dict['labels'] = [d['labels'] for d in batch]  # Keep as list
    
    # Handle image IDs normally
    if 'image_id' in elem:
        batch_dict['image_id'] = default_collate([d['image_id'] for d in batch])
    
    return batch_dict

class UnifiedDataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=(640, 640)):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load unified annotations
        anno_path = self.data_dir / "unified_annotations.json"
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {anno_path}")
            
        with open(anno_path, 'r') as f:
            self.annotations = json.load(f)
            
        self.image_ids = [img['id'] for img in self.annotations['images']]
        print(f"Loaded {len(self.image_ids)} images")
        
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Find image info
        img_info = next(img for img in self.annotations['images'] if img['id'] == img_id)
        
        # Load and process image
        img_path = self.data_dir / "images" / img_info['file_name']
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        # Read image using OpenCV
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(image)
        
        # Apply transforms
        image = self.transform(image)
        
        # Load labels
        label_path = self.data_dir / "labels" / f"{img_info['original_name']}.txt"
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    values = [float(x) for x in line.strip().split()]
                    labels.append(values)
        labels = torch.tensor(labels) if labels else torch.zeros((0, 5))
        
        return {
            'image': image,
            'labels': labels,
            'image_id': img_id
        }

def get_dataloader(data_dir, split='train', batch_size=16, num_workers=4):
    try:
        dataset = UnifiedDataset(data_dir, split)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        return dataloader
    except Exception as e:
        print(f"Error creating dataloader: {str(e)}")
        raise 