import os
from pathlib import Path
import json
import numpy as np
import cv2
from tqdm import tqdm
import torch
from pycocotools.coco import COCO

class DataPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "processed"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create directories for processed data
        self.processed_images = self.output_dir / "images"
        self.processed_labels = self.output_dir / "labels"
        os.makedirs(self.processed_images, exist_ok=True)
        os.makedirs(self.processed_labels, exist_ok=True)

    def preprocess_coco(self, split='train2017'):
        """Preprocess COCO dataset"""
        coco_dir = self.data_dir / "coco"
        anno_file = coco_dir / "annotations" / f"instances_{split}.json"
        
        # Initialize COCO API
        coco = COCO(anno_file)
        
        # Process each image
        for img_id in tqdm(coco.getImgIds(), desc=f"Processing COCO {split}"):
            img_info = coco.loadImgs(img_id)[0]
            
            # Load image
            img_path = coco_dir / split / img_info['file_name']
            img = cv2.imread(str(img_path))
            
            # Get annotations
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            # Process annotations
            boxes = []
            categories = []
            for ann in anns:
                bbox = ann['bbox']  # [x,y,width,height]
                # Convert to YOLO format [x_center, y_center, width, height]
                x_center = (bbox[0] + bbox[2]/2) / img_info['width']
                y_center = (bbox[1] + bbox[3]/2) / img_info['height']
                width = bbox[2] / img_info['width']
                height = bbox[3] / img_info['height']
                
                boxes.append([x_center, y_center, width, height])
                categories.append(ann['category_id'])
            
            # Save processed data
            out_img_path = self.processed_images / f"{img_id}.jpg"
            out_label_path = self.processed_labels / f"{img_id}.txt"
            
            cv2.imwrite(str(out_img_path), img)
            
            # Save labels in YOLO format
            with open(out_label_path, 'w') as f:
                for box, cat in zip(boxes, categories):
                    f.write(f"{cat} {' '.join(map(str, box))}\n")

    def preprocess_xview(self):
        """Preprocess xView dataset"""
        xview_dir = self.data_dir / "xView"
        train_dir = xview_dir / "train_images" / "train_images"
        label_file = xview_dir / "train_labels" / "xView_train.geojson"
        
        # Load geojson annotations
        with open(label_file) as f:
            annotations = json.load(f)
            
        # Process each image
        for img_path in tqdm(train_dir.glob("*.tif"), desc="Processing xView"):
            # Convert TIF to JPG and normalize
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            img_name = img_path.stem
            
            # Save processed image
            out_img_path = self.processed_images / f"xview_{img_name}.jpg"
            out_label_path = self.processed_labels / f"xview_{img_name}.txt"
            
            cv2.imwrite(str(out_img_path), img)
            
            # Create empty label file (we'll implement proper annotation conversion later)
            with open(out_label_path, 'w') as f:
                f.write("")  # Placeholder for actual annotations

    def preprocess_vqa(self):
        """Preprocess VQA annotations"""
        # Implementation here

    def create_unified_annotations(self):
        """Create unified annotation format for all datasets"""
        unified_annotations = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add COCO categories
        coco_anno_file = self.data_dir / "coco/annotations/instances_train2017.json"
        if coco_anno_file.exists():
            with open(coco_anno_file, 'r') as f:
                coco_data = json.load(f)
                unified_annotations['categories'].extend(coco_data['categories'])
        
        # Process COCO images and annotations
        processed_images = list(self.processed_images.glob("*.jpg"))
        next_id = 1  # Counter for generating unique IDs
        
        for img_path in processed_images:
            stem = img_path.stem
            
            # Generate unique ID based on filename
            if stem.startswith('xview_'):
                # For xView images, use incremental ID
                img_id = next_id
                next_id += 1
                dataset = 'xview'
            else:
                # For COCO images, use original ID
                try:
                    img_id = int(stem)
                    dataset = 'coco'
                except ValueError:
                    img_id = next_id
                    next_id += 1
                    dataset = 'other'
            
            # Add image info
            unified_annotations['images'].append({
                'id': img_id,
                'file_name': img_path.name,
                'dataset': dataset,
                'original_name': stem
            })
            
            # Add corresponding annotations
            label_path = self.processed_labels / f"{stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for i, line in enumerate(f):
                        cat_id, *bbox = map(float, line.strip().split())
                        unified_annotations['annotations'].append({
                            'id': len(unified_annotations['annotations']),
                            'image_id': img_id,
                            'category_id': int(cat_id),
                            'bbox': bbox,  # [x_center, y_center, width, height] in normalized coordinates
                        })
        
        # Save unified annotations
        print(f"Total images: {len(unified_annotations['images'])}")
        print(f"Total annotations: {len(unified_annotations['annotations'])}")
        print(f"Total categories: {len(unified_annotations['categories'])}")
        
        with open(self.output_dir / "unified_annotations.json", 'w') as f:
            json.dump(unified_annotations, f)