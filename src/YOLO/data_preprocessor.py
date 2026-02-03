"""
Data Preprocessor for YOLO11 Training
Converts JSON annotations to YOLO format and creates train/val splits
"""

import json
import os
import shutil
import sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
from typing import Dict, List, Tuple

class YOLODataPreprocessor:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_file = self.data_dir / "annotations.json"
        
        # Class mapping for expiration date components
        self.class_mapping = {
            "day": 0,
            "month": 1, 
            "year": 2
        }
        
        # Create output directories
        self.create_output_directories()
    

    # Create necessary directory structure for YOLO training
    def create_output_directories(self):
        dirs = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "labels" / "train", 
            self.output_dir / "labels" / "val"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    

    # Load JSON annotations file
    def load_annotations(self) -> Dict:
        with open(self.annotations_file, 'r') as f:
            return json.load(f)
    

    # Convert bbox format [x1, y1, x2, y2] to YOLO format [x_center, y_center, width, height]
    # All values normalized to [0, 1]
    def convert_bbox_to_yolo(self, bbox: List[int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        
        # Calculate center coordinates and dimensions
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Normalize to [0, 1]
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        return x_center_norm, y_center_norm, width_norm, height_norm
    
    # Convert JSON annotations to YOLO format
    # Returns dictionary mapping image filenames to YOLO annotation strings
    def convert_annotations_to_yolo(self, annotations: Dict) -> Dict[str, List]:
        yolo_annotations = {}
        
        for img_filename, img_data in annotations.items():
            img_width = img_data['width']
            img_height = img_data['height']
            annotations_list = img_data.get('ann', [])
            
            yolo_labels = []
            for ann in annotations_list:
                class_name = ann['cls']
                bbox = ann['bbox']
                
                # Get class index
                class_idx = self.class_mapping.get(class_name)
                if class_idx is None:
                    print(f"Warning: Unknown class '{class_name}' in image {img_filename}")
                    continue
                
                # Convert bbox to YOLO format
                x_center, y_center, width, height = self.convert_bbox_to_yolo(
                    bbox, img_width, img_height
                )
                
                # Create YOLO label line: class_idx x_center y_center width height
                yolo_label = f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_labels.append(yolo_label)
            
            yolo_annotations[img_filename] = yolo_labels
        return yolo_annotations
    
    # Train validation split
    def split_data(self, image_files: List[str], test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str]]:
        train_files, val_files = train_test_split(
            image_files, test_size=test_size, random_state=random_state
        )
        return train_files, val_files
    
    def copy_images_and_labels(self, image_files: List[str], yolo_annotations: Dict, split: str):
        """Copy images and create label files for the given split"""
        split_dir = "train" if split == "train" else "val"
        
        for img_file in image_files:
            # Copy image
            src_img_path = self.images_dir / img_file
            dst_img_path = self.output_dir / "images" / split_dir / img_file
            
            if src_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path)
            else:
                print(f"Warning: Image {img_file} not found")
                continue
            
            # Create label file
            label_file = Path(img_file).with_suffix('.txt')
            dst_label_path = self.output_dir / "labels" / split_dir / label_file
            
            labels = yolo_annotations.get(img_file, [])
            with open(dst_label_path, 'w') as f:
                for label in labels:
                    f.write(label + '\n')
    
    def create_yaml_config(self):
        """Create YAML configuration file for YOLO training"""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(self.class_mapping),
            'names': list(self.class_mapping.keys())
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return yaml_path
    
    def preprocess(self, test_size: float = 0.2):
        """Main preprocessing pipeline"""
        print("Starting data preprocessing...")
        
        # Load annotations
        print("Loading annotations...")
        annotations = self.load_annotations()
        print(f"Loaded {len(annotations)} annotations")
        
        # Convert to YOLO format
        print("Converting annotations to YOLO format...")
        yolo_annotations = self.convert_annotations_to_yolo(annotations)
        
        # Get list of image files
        image_files = list(annotations.keys())
        print(f"Found {len(image_files)} images")
        
        # Split data
        print(f"Splitting data (test_size={test_size})...")
        train_files, val_files = self.split_data(image_files, test_size)
        print(f"Training: {len(train_files)} images")
        print(f"Validation: {len(val_files)} images")
        
        # Copy images and create labels
        print("Copying images and creating labels...")
        self.copy_images_and_labels(train_files, yolo_annotations, "train")
        self.copy_images_and_labels(val_files, yolo_annotations, "val")
        
        # Create YAML config
        print("Creating YAML configuration...")
        yaml_path = self.create_yaml_config()
        
        print(f"Preprocessing completed! Data saved to: {self.output_dir}")
        print(f"YAML configuration: {yaml_path}")
        
        return yaml_path

def main():
    # Configuration
    data_dir = "data/dates"
    output_dir = "data/yolo_formatted"
    
    # Create preprocessor and run
    preprocessor = YOLODataPreprocessor(data_dir, output_dir)
    yaml_path = preprocessor.preprocess(test_size=0.2)
    
    print("\nDataset ready for YOLO11 training!")
    print(f"Classes: {list(preprocessor.class_mapping.keys())}")

if __name__ == "__main__":
    main()
