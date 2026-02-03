# YOLO11 Training for Expiration Date Detection

This directory contains all the necessary scripts to train a YOLO11 model for detecting expiration date components (day, month, year) in images.

## Project Structure

```
src/YOLO/
├── data_preprocessor.py    # Convert JSON annotations to YOLO format
├── yolo_config.yaml        # YOLO11 training configuration
├── train_yolo.py           # Main training script
├── inference.py            # Inference script for testing
└── README.md               # This file
```

## Setup

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your data is in the following structure:
```
data/dates/
├── images/                 # All training images
│   ├── 00000.jpg
│   ├── 00001.jpg
│   └── ...
└── annotations.json        # JSON annotations file
```

## Usage

### 1. Data Preprocessing

First, convert your JSON annotations to YOLO format:

```bash
python src/YOLO/data_preprocessor.py
```

This will:
- Convert JSON annotations to YOLO format
- Split data into training/validation sets (80/20 split)
- Create the required directory structure
- Generate `dataset.yaml` configuration file

The processed data will be saved to `data/yolo_formatted/`.

### 2. Training

Train the YOLO11 model:

```bash
python src/YOLO/train_yolo.py --config src/YOLO/yolo_config.yaml --mode train
```

Training options:
- `--config`: Path to configuration file (default: yolo_config.yaml)
- `--mode`: Training mode (train/validate, default: train)

The trained model will be saved to `runs/train/exp/weights/`:
- `best.pt`: Best model checkpoint
- `last.pt`: Final model checkpoint

### 3. Validation

Validate the trained model:

```bash
python train_yolo.py --config yolo_config.yaml --mode validate --model runs/train/exp/weights/best.pt
```

### 4. Inference

Run inference on test images:

```bash
# Single image
python inference.py --model runs/train/exp/weights/best.pt --input path/to/image.jpg

# Batch inference on directory
python inference.py --model runs/train/exp/weights/best.pt --input path/to/images/ --output results/
```

Inference options:
- `--model`: Path to trained model
- `--input`: Path to image or directory
- `--output`: Output directory (default: inference_results)
- `--conf`: Confidence threshold (default: 0.5)
- `--iou`: IoU threshold for NMS (default: 0.45)

## Configuration

### Model Configuration (yolo_config.yaml)

Key parameters you can adjust:

- `model`: YOLO11 variant (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
- `epochs`: Number of training epochs (default: 100)
- `batch_size`: Batch size (default: 32)
- `imgsz`: Image size (default: 640)
- `lr0`: Initial learning rate (default: 0.01)

### Class Information

The model is trained to detect 3 classes:
- `day`: Day component of expiration date
- `month`: Month component of expiration date  
- `year`: Year component of expiration date

## Output Results

### Training Output
- Training plots and metrics saved to `runs/train/exp/`
- Model weights in `runs/train/exp/weights/`
- Training logs and statistics

### Inference Output
- Visualization images with bounding boxes
- JSON files with detection results
- Summary statistics for batch processing

## Performance Tips

1. **GPU Training**: Ensure you have a CUDA-enabled GPU for faster training
2. **Batch Size**: Adjust batch size based on your GPU memory
3. **Image Size**: Larger image sizes (640+) improve accuracy but require more memory
4. **Data Augmentation**: The config includes various augmentation techniques
5. **Early Stopping**: Training stops automatically if validation doesn't improve

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Poor Detection**: Increase training epochs or adjust learning rate
3. **No Detections**: Lower confidence threshold during inference
4. **Slow Training**: Ensure GPU is being used (check device output)

### Model Variants

- `yolo11n.pt`: Nano - Fastest, good for lightweight applications
- `yolo11s.pt`: Small - Balance of speed and accuracy
- `yolo11m.pt`: Medium - Better accuracy
- `yolo11l.pt`: Large - High accuracy
- `yolo11x.pt`: Extra Large - Best accuracy, slowest

## Integration

The trained model can be integrated into your OCR pipeline:

1. Use the inference script to detect date components
2. Crop detected regions
3. Apply OCR to each component
4. Combine results to get complete expiration date

## Example Integration

```python
from src.YOLO.inference import YOLO11Inference

# Initialize inference
model = YOLO11Inference("runs/train/exp/weights/best.pt")

# Detect date components
results = model.predict_single("test_image.jpg")

# Process detections
for detection in results['detections']:
    class_name = detection['class']
    bbox = detection['bbox']
    confidence = detection['confidence']
    
    # Crop region and apply OCR
    # ... your OCR logic here
```

## Next Steps

1. Train the model on your dataset
2. Evaluate performance on validation set
3. Fine-tune hyperparameters if needed
4. Integrate with your OCR pipeline
5. Deploy model for production use
