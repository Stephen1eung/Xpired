"""
YOLO11 Training Script for Expiration Date Detection
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

class YOLO11Trainer:
    def __init__(self, config_path: str = "yolo_config.yaml"):
        """
        Initialize YOLO11 trainer
        
        Args:
            config_path: Path to YOLO configuration file
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
        # Set up paths
        self.setup_paths()
        
        # Check device availability
        self.device = self.get_device()
        
    def load_config(self) -> dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def setup_paths(self):
        """Setup and validate paths"""
        # Convert relative paths to absolute
        if not Path(self.config['path']).is_absolute():
            config_dir = self.config_path.parent
            self.config['path'] = str(config_dir / self.config['path'])
        
        # Validate dataset path
        dataset_path = Path(self.config['path'])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Check for required directories
        required_dirs = [
            dataset_path / self.config['train'],
            dataset_path / self.config['val_dir']
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
        
        print(f"Dataset path validated: {dataset_path}")
    
    def get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            device = f"cuda:{self.config.get('device', 0)}"
            print(f"Using GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            print(f"Total GPUs available: {gpu_count}")
            
            # Check if we have enough memory for the batch size
            batch_size = self.config.get('batch_size', 64)
            estimated_memory = batch_size * 0.5  # Rough estimate per image in GB
            
            if estimated_memory > gpu_memory * 0.8:  # Leave 20% margin
                print(f"Warning: Estimated memory usage ({estimated_memory:.1f} GB) may exceed GPU memory")
                print(f"Consider reducing batch_size if you encounter CUDA out of memory errors")
            
        else:
            device = "cpu"
            print("CUDA not available, using CPU for training")
            print("Note: Training will be significantly slower on CPU")
            print("Consider using a GPU with CUDA support for faster training")
        
        return device
    
    def create_model(self) -> YOLO:
        """Create and configure YOLO11 model"""
        model_name = self.config.get('model', 'yolo11n.pt')
        
        print(f"Loading model: {model_name}")
        
        # Create YOLO model
        model = YOLO(model_name)
        
        return model
    
    def train(self):
        """Run the training process"""
        print("Starting YOLO11 training...")
        print(f"Configuration: {self.config_path}")
        
        # Create model
        model = self.create_model()
        
        # Training parameters from config
        training_params = {
            'data': 'data/yolo_formatted/dataset.yaml',
            'epochs': self.config.get('epochs', 100),
            'batch': self.config.get('batch_size', 32),
            'imgsz': self.config.get('imgsz', 640),
            'device': self.device,
            'workers': self.config.get('workers', 8),
            'project': self.config.get('project', 'runs/train'),
            'name': self.config.get('name', 'exp'),
            'exist_ok': self.config.get('exist_ok', False),
            'pretrained': True,
            'optimizer': 'SGD',
            'lr0': self.config.get('lr0', 0.01),
            'lrf': self.config.get('lrf', 0.01),
            'momentum': self.config.get('momentum', 0.937),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            'warmup_epochs': self.config.get('warmup_epochs', 3),
            'warmup_momentum': self.config.get('warmup_momentum', 0.8),
            'warmup_bias_lr': self.config.get('warmup_bias_lr', 0.1),
            'box': self.config.get('box', 7.5),
            'cls': self.config.get('cls', 0.5),
            'dfl': self.config.get('dfl', 1.5),
            'hsv_h': self.config.get('hsv_h', 0.015),
            'hsv_s': self.config.get('hsv_s', 0.7),
            'hsv_v': self.config.get('hsv_v', 0.4),
            'degrees': self.config.get('degrees', 0.0),
            'translate': self.config.get('translate', 0.1),
            'scale': self.config.get('scale', 0.5),
            'shear': self.config.get('shear', 0.0),
            'perspective': self.config.get('perspective', 0.0),
            'flipud': self.config.get('flipud', 0.0),
            'fliplr': self.config.get('fliplr', 0.5),
            'mosaic': self.config.get('mosaic', 1.0),
            'mixup': self.config.get('mixup', 0.0),
            'cache': self.config.get('cache', False),
            'amp': self.config.get('amp', True),
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': self.config.get('val', True),
            'plots': self.config.get('plots', True),
            'save': self.config.get('save', True),
            'save_period': self.config.get('save_period', -1),
            'patience': self.config.get('patience', 50),
            'verbose': self.config.get('verbose', True),
            'seed': self.config.get('seed', 42),
            'deterministic': self.config.get('deterministic', True),
            'single_cls': False,
            'rect': self.config.get('rect', False),
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'auto_augment': self.config.get('auto_augment', 'randaugment'),
            'augment': True
        }
        
        print("Training parameters:")
        for key, value in training_params.items():
            if key in ['epochs', 'batch', 'imgsz', 'device']:
                print(f"  {key}: {value}")
        
        # Start training
        try:
            results = model.train(**training_params)
            
            print("\nTraining completed successfully!")
            print(f"Best model saved at: {results.save_dir / 'weights' / 'best.pt'}")
            print(f"Last model saved at: {results.save_dir / 'weights' / 'last.pt'}")
            
            return results
            
        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            raise
    
    def validate(self, model_path: str = None):
        """Validate the trained model"""
        if model_path is None:
            # Use the best model from the latest training run
            model_path = f"{self.config.get('project', 'runs/train')}/{self.config.get('name', 'exp')}/weights/best.pt"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Validating model: {model_path}")
        
        # Load model
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(
            data='data/yolo_formatted/dataset.yaml',
            device=self.device,
            verbose=True
        )
        
        print("Validation completed!")
        print(f"Overall mAP50: {results.box.map50:.4f}")
        print(f"Overall mAP50-95: {results.box.map:.4f}")
        
        # Print per-class metrics
        if hasattr(results.box, 'maps'):
            class_names = ['day', 'month', 'year']
            for i, class_name in enumerate(class_names):
                if i < len(results.box.maps):
                    print(f"{class_name} mAP50: {results.box.maps[i]:.4f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Training for Expiration Date Detection")
    parser.add_argument("--config", type=str, default="yolo_config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "validate"], default="train",
                       help="Mode: train or validate")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model for validation")
    
    args = parser.parse_args()
    
    try:
        # Create trainer
        trainer = YOLO11Trainer(args.config)
        
        if args.mode == "train":
            # Run training
            results = trainer.train()
            
        elif args.mode == "validate":
            # Run validation
            results = trainer.validate(args.model)
        
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
