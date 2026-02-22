"""Create train/val split for digit images and generate YOLO labels."""

from __future__ import annotations

import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMAGE_DIR = '././data/digits'
OUTPUT_IMAGE_DIR = '././data/digits/output'
SPLITS = ['train', 'val']

def upscale_image(image_path, target_size=(640, 640)):
    """Load an image and upscale it to the target size."""
    # Open the image
    img = Image.open(image_path)

    # Upscale the image to the target size using bilinear interpolation
    resample = getattr(getattr(Image, 'Resampling', Image), 'BILINEAR')
    img_upscaled = img.resize(target_size, resample)

    return img_upscaled

# Create necessary directories
for split_name in SPLITS:
    os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, split_name, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, split_name, 'labels'), exist_ok=True)

def move_and_label(file_list, split_label, folder_path, class_id_value):
    """Move images to the split folder and create a simple YOLO label file."""
    for _, row in file_list.iterrows():
        img_name = row['filename']
        src_path = os.path.join(folder_path, img_name)

        # Upscale the image before saving
        upscaled_img = upscale_image(src_path, target_size=(640, 640))

        # Save the upscaled image
        dst_image_dir = os.path.join(OUTPUT_IMAGE_DIR, split_label, 'images')
        upscaled_img.save(os.path.join(dst_image_dir, img_name))

        # Create YOLO label for the image
        dst_label_dir = os.path.join(OUTPUT_IMAGE_DIR, split_label, 'labels')
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(dst_label_dir, label_name)

        yolo_label = f"{class_id_value} 0.5 0.5 0.6 0.6\n"
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write(yolo_label)

for digit_folder in os.listdir(IMAGE_DIR):
    digit_folder_path = os.path.join(IMAGE_DIR, digit_folder)

    if os.path.isdir(digit_folder_path):
        try:
            class_id = int(digit_folder)
        except ValueError:
            print(f"Skipping folder '{digit_folder}' because it's not a valid class ID.")
            continue

        image_files = [f for f in os.listdir(digit_folder_path) if f.endswith(('.jpg', '.png'))]
        df = pd.DataFrame(image_files, columns=['filename'])

        train_files, val_files = train_test_split(df, test_size=0.2, random_state=42)

        move_and_label(train_files, 'train', digit_folder_path, class_id)
        move_and_label(val_files, 'val', digit_folder_path, class_id)

print("Images and YOLO labels generated for train/val split.")