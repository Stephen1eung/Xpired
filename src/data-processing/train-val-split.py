import os
import pandas as pd
import shutil

from PIL import Image
from sklearn.model_selection import train_test_split

image_dir = '././data/digits'
output_image_dir = '././data/digits/output'
splits = ['train', 'val']

def upscale_image(image_path, target_size=(640, 640)):
    # Open the image
    img = Image.open(image_path)
    
    # Upscale the image to the target size using bilinear interpolation
    img_upscaled = img.resize(target_size, Image.BILINEAR)  # You can change this to Image.BICUBIC for smoother results
    
    return img_upscaled

# Create necessary directories
for split in splits:
    os.makedirs(os.path.join(output_image_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_image_dir, split, 'labels'), exist_ok=True)

for digit_folder in os.listdir(image_dir):
    digit_folder_path = os.path.join(image_dir, digit_folder)

    if os.path.isdir(digit_folder_path):
        try:
            class_id = int(digit_folder)  # Try to convert folder name to integer
        except ValueError:
            print(f"Skipping folder '{digit_folder}' because it's not a valid class ID.")
            continue  # Skip this folder if it's not a valid class ID

        image_files = [f for f in os.listdir(digit_folder_path) if f.endswith(('.jpg', '.png'))]
        df = pd.DataFrame(image_files, columns=['filename'])

        train_files, val_files = train_test_split(df, test_size=0.2, random_state=42)

        def move_and_label(file_list, split):
            for _, row in file_list.iterrows():
                img_name = row['filename']
                src_path = os.path.join(digit_folder_path, img_name)

                # Upscale the image before saving
                upscaled_img = upscale_image(src_path, target_size=(640, 640))

                # Save the upscaled image
                dst_image_dir = os.path.join(output_image_dir, split, 'images')
                upscaled_img.save(os.path.join(dst_image_dir, img_name))

                # Create YOLO label for the image
                dst_label_dir = os.path.join(output_image_dir, split, 'labels')
                label_name = os.path.splitext(img_name)[0] + '.txt'
                label_path = os.path.join(dst_label_dir, label_name)

                # Write the label (you can customize the label format as needed)
                yolo_label = f"{class_id} 0.5 0.5 0.6 0.6\n"

                with open(label_path, 'w') as f:
                    f.write(yolo_label)

        # Move and label both train and validation files
        move_and_label(train_files, 'train')
        move_and_label(val_files, 'val')

print("✅ Done! Images and YOLO labels generated for train/val split.")