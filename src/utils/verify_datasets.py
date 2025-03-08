import os
import json
from pathlib import Path
import numpy as np

def verify_datasets(data_dir):
    """
    Verify the dataset structure and print summary statistics
    """
    data_dir = Path(data_dir)
    print("Verifying datasets...\n")

    # Verify COCO
    print("=== COCO Dataset ===")
    coco_dir = data_dir / "coco"
    if coco_dir.exists():
        train_imgs = len(list((coco_dir / "train2017").glob("*.jpg")))
        val_imgs = len(list((coco_dir / "val2017").glob("*.jpg")))
        
        print(f"Training images: {train_imgs}")
        print(f"Validation images: {val_imgs}")
        
        # Verify annotations
        anno_file = coco_dir / "annotations" / "instances_train2017.json"
        if anno_file.exists():
            with open(anno_file) as f:
                anno_data = json.load(f)
                print(f"Number of annotations: {len(anno_data['annotations'])}")
    else:
        print("COCO dataset not found!")

    # Verify xView
    print("\n=== xView Dataset ===")
    xview_dir = data_dir / "xView"
    if xview_dir.exists():
        train_imgs = len(list((xview_dir / "train_images" / "train_images").glob("*.tif")))
        val_imgs = len(list((xview_dir / "val_images/val_images").glob("*.tif")))
        
        print(f"Training images: {train_imgs}")
        print(f"Validation images: {val_imgs}")
        
        # Verify geojson
        geojson_file = xview_dir / "train_labels" / "xView_train.geojson"
        if geojson_file.exists():
            print("Training labels (geojson) found")
    else:
        print("xView dataset not found!")

    # Verify VQA
    print("\n=== VQA Dataset ===")
    vqa_dir = data_dir / "vqa"
    if vqa_dir.exists():
        # Check annotations
        train_anno = vqa_dir / "v2_Annotations_Train_mscoco" / "v2_mscoco_train2014_annotations.json"
        val_anno = vqa_dir / "v2_Annotations_Val_mscoco" / "v2_mscoco_val2014_annotations.json"
        
        if train_anno.exists() and val_anno.exists():
            with open(train_anno) as f:
                train_data = json.load(f)
            with open(val_anno) as f:
                val_data = json.load(f)
                
            print(f"Training QA pairs: {len(train_data['annotations'])}")
            print(f"Validation QA pairs: {len(val_data['annotations'])}")
    else:
        print("VQA dataset not found!")

if __name__ == "__main__":
    verify_datasets("data") 