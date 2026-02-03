"""
YOLO11 Inference Script for Expiration Date Detection
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import json
from typing import List, Dict, Tuple

class YOLO11Inference:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, 
                 iou_threshold: float = 0.45):
        """
        Initialize YOLO11 inference
        
        Args:
            model_path: Path to trained model file (.pt)
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Class names for expiration date components
        self.class_names = ['day', 'month', 'year']
        
        # Load model
        self.load_model()
        
        # Check device
        self.device = self.get_device()
    
    def load_model(self):
        """Load the trained YOLO11 model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Get model metadata
        self.model_info = self.model.info()
        print(f"Model loaded successfully!")
        print(f"Classes: {self.model_info['names']}")
    
    def get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using GPU: {gpu_name} (Total GPUs: {gpu_count})")
        else:
            device = "cpu"
            print("Using CPU for inference")
        
        return device
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image for inference
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB (YOLO expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def predict_single(self, image_path: str, save_results: bool = False, 
                      output_dir: str = "inference_results") -> Dict:
        """
        Run inference on a single image
        
        Args:
            image_path: Path to input image
            save_results: Whether to save visualization results
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing detection results
        """
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # Process results
        detections = self.process_results(results[0], image_path)
        
        # Save visualization if requested
        if save_results:
            self.save_results(image, results[0], detections, image_path, output_dir)
        
        return detections
    
    def predict_batch(self, image_dir: str, output_dir: str = "inference_results") -> List[Dict]:
        """
        Run inference on a directory of images
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
            
        Returns:
            List of detection results for each image
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        print(f"Found {len(image_files)} images for inference")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        all_results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            try:
                result = self.predict_single(
                    str(image_file), 
                    save_results=True, 
                    output_dir=output_dir
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {image_file.name}: {str(e)}")
                continue
        
        # Save summary results
        self.save_batch_results(all_results, output_dir)
        
        return all_results
    
    def process_results(self, result, image_path: str) -> Dict:
        """
        Process YOLO results into a structured format
        
        Args:
            result: YOLO result object
            image_path: Path to original image
            
        Returns:
            Dictionary containing detection results
        """
        # Get boxes, scores, and classes
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return {
                'image_path': image_path,
                'detections': [],
                'num_detections': 0
            }
        
        detections = []
        
        for i in range(len(boxes)):
            # Get box coordinates (xyxy format)
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            
            # Get confidence score
            confidence = float(boxes.conf[i].cpu().numpy())
            
            # Get class index and name
            class_idx = int(boxes.cls[i].cpu().numpy())
            class_name = self.class_names[class_idx]
            
            detection = {
                'class': class_name,
                'class_id': class_idx,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2],
                'center': [(x1 + x2) // 2, (y1 + y2) // 2],
                'width': x2 - x1,
                'height': y2 - y1
            }
            detections.append(detection)
        
        return {
            'image_path': image_path,
            'detections': detections,
            'num_detections': len(detections)
        }
    
    def save_results(self, image: np.ndarray, result, detections: Dict, 
                    image_path: str, output_dir: str):
        """
        Save visualization results
        
        Args:
            image: Original image array
            result: YOLO result object
            detections: Processed detection results
            image_path: Path to original image
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create visualization
        vis_image = image.copy()
        
        # Draw bounding boxes
        for detection in detections['detections']:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Choose color based on class
            colors = {
                'day': (0, 255, 0),      # Green
                'month': (255, 0, 0),    # Red
                'year': (0, 0, 255)      # Blue
            }
            color = colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Convert RGB back to BGR for OpenCV saving
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        
        # Save visualization
        image_name = Path(image_path).stem
        vis_path = output_path / f"{image_name}_detection.jpg"
        cv2.imwrite(str(vis_path), vis_image)
        
        # Save detection results as JSON
        json_path = output_path / f"{image_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(detections, f, indent=2)
    
    def save_batch_results(self, all_results: List[Dict], output_dir: str):
        """
        Save batch inference summary
        
        Args:
            all_results: List of detection results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        
        # Save complete results
        results_path = output_path / "batch_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create summary statistics
        total_images = len(all_results)
        total_detections = sum(r['num_detections'] for r in all_results)
        
        class_counts = {}
        for result in all_results:
            for detection in result['detections']:
                class_name = detection['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        summary = {
            'total_images': total_images,
            'total_detections': total_detections,
            'average_detections_per_image': total_detections / total_images if total_images > 0 else 0,
            'class_distribution': class_counts
        }
        
        # Save summary
        summary_path = output_path / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nBatch inference completed!")
        print(f"Total images processed: {total_images}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {summary['average_detections_per_image']:.2f}")
        print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Inference for Expiration Date Detection")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input image or directory")
    parser.add_argument("--output", type=str, default="inference_results",
                       help="Output directory for results")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold for NMS")
    
    args = parser.parse_args()
    
    try:
        # Create inference object
        inferencer = YOLO11Inference(
            model_path=args.model,
            confidence_threshold=args.conf,
            iou_threshold=args.iou
        )
        
        # Check if input is file or directory
        input_path = Path(args.input)
        if input_path.is_file():
            # Single image inference
            print(f"Running inference on single image: {input_path}")
            result = inferencer.predict_single(
                str(input_path), 
                save_results=True, 
                output_dir=args.output
            )
            
            print(f"\nDetection results for {input_path.name}:")
            print(f"Number of detections: {result['num_detections']}")
            for detection in result['detections']:
                print(f"  {detection['class']}: {detection['confidence']:.3f}")
        
        elif input_path.is_dir():
            # Batch inference
            print(f"Running batch inference on directory: {input_path}")
            all_results = inferencer.predict_batch(str(input_path), args.output)
        
        else:
            raise FileNotFoundError(f"Input path not found: {args.input}")
        
        print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
