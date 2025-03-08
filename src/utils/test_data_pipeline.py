from data_preprocessing import DataPreprocessor
from data_loading import get_dataloader
import torch
import os
import json
from pathlib import Path

def test_pipeline():
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor("data")
        
        # Preprocess datasets
        print("Preprocessing COCO dataset...")
        preprocessor.preprocess_coco(split='train2017')
        
        print("\nPreprocessing xView dataset...")
        preprocessor.preprocess_xview()
        
        print("\nCreating unified annotations...")
        preprocessor.create_unified_annotations()
        
        # Verify preprocessing results
        processed_dir = os.path.join("data", "processed")
        n_images = len(list(Path(processed_dir).glob("images/*.jpg")))
        n_labels = len(list(Path(processed_dir).glob("labels/*.txt")))
        print(f"\nProcessed data statistics:")
        print(f"Number of images: {n_images}")
        print(f"Number of label files: {n_labels}")
        
        # Load and verify annotations
        anno_file = os.path.join(processed_dir, "unified_annotations.json")
        if os.path.exists(anno_file):
            with open(anno_file, 'r') as f:
                annotations = json.load(f)
            print(f"Number of images in annotations: {len(annotations['images'])}")
            print(f"Number of annotations: {len(annotations['annotations'])}")
        else:
            print("Warning: unified_annotations.json not found!")
        
        # Test data loading
        print("\nTesting data loader...")
        train_loader = get_dataloader("data", split='train', batch_size=2)
        print(f"Dataset size: {len(train_loader.dataset)}")
        
        # Test a batch
        batch = next(iter(train_loader))
        print("\nBatch contents:")
        print(f"Images shape: {batch['image'].shape}")  # Should be [2, 3, 640, 640]
        print("Labels shapes:", [labels.shape for labels in batch['labels']])
        print(f"Image IDs: {batch['image_id']}")
        
        # Verify image values
        print("\nImage statistics:")
        print(f"Image value range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        print(f"Image mean: {batch['image'].mean():.3f}")
        print(f"Image std: {batch['image'].std():.3f}")
        
        print("\nSuccess! Data pipeline is working correctly.")
        
    except Exception as e:
        print(f"\nError in test pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    test_pipeline() 