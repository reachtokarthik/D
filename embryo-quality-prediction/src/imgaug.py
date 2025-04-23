import os
import cv2
import numpy as np
import albumentations as A
from PIL import Image
from tqdm import tqdm
import os.path as path
import random

# Get the absolute path to the project root directory
SCRIPT_DIR = path.dirname(path.abspath(__file__))
PROJECT_ROOT = path.dirname(SCRIPT_DIR)

# Source and output folders with absolute paths
SOURCE_DIR = path.join(PROJECT_ROOT, "data", "sorted")
AUGMENTED_DIR = path.join(PROJECT_ROOT, "data", "augmented")
TARGET_SIZE = (224, 224)
AUG_PER_IMAGE = 5  # Increased number of augmented copies per original image

# Create different augmentation pipelines for more diverse results
basic_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1])
])

color_transforms = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
    A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1])
])

geometric_transforms = A.Compose([
    A.Rotate(limit=45, p=0.7),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
    A.OpticalDistortion(distort_limit=0.3, shift_limit=0.5, p=0.3),
    A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1])
])

crop_transforms = A.Compose([
    A.RandomCrop(height=200, width=200, p=0.7),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.PadIfNeeded(min_height=TARGET_SIZE[0], min_width=TARGET_SIZE[1], p=1.0),
    A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1])
])

# List of all transform pipelines
transform_pipelines = [basic_transforms, color_transforms, geometric_transforms, crop_transforms]

# Function to count images in a directory
def count_images(directory):
    count = 0
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            count += len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return count

# Print dataset statistics before augmentation
if os.path.exists(SOURCE_DIR):
    original_count = count_images(SOURCE_DIR)
    print(f"📊 Original dataset: {original_count} images")
else:
    print(f"⚠️ Source directory {SOURCE_DIR} not found")
    exit(1)
    
# Ensure target folders exist
for folder in os.listdir(SOURCE_DIR):
    src_folder = os.path.join(SOURCE_DIR, folder)
    if not os.path.isdir(src_folder):
        continue
        
    dest_folder = os.path.join(AUGMENTED_DIR, folder)
    os.makedirs(dest_folder, exist_ok=True)

    for filename in tqdm(os.listdir(src_folder), desc=f"✨ Augmenting {folder}"):
        src_path = os.path.join(src_folder, filename)
        try:
            img = cv2.imread(src_path)
            if img is None:
                continue

            # Apply different augmentation pipelines
            for i in range(AUG_PER_IMAGE):
                # Select a random transform pipeline for each augmentation
                transform = random.choice(transform_pipelines)
                augmented = transform(image=img)
                aug_img = augmented["image"]
                
                # Create more descriptive filenames
                transform_type = "basic" if transform == basic_transforms else "color" if transform == color_transforms else "geom" if transform == geometric_transforms else "crop"
                aug_filename = f"{os.path.splitext(filename)[0]}_aug{i+1}_{transform_type}.jpg"
                
                # Save the augmented image
                cv2.imwrite(os.path.join(dest_folder, aug_filename), aug_img)

        except Exception as e:
            print(f"⚠️ Error augmenting {filename}: {e}")

# Print statistics after augmentation
if os.path.exists(AUGMENTED_DIR):
    augmented_count = count_images(AUGMENTED_DIR)
    print(f"\n📊 Augmentation complete:")
    print(f"   - Original images: {original_count}")
    print(f"   - Augmented images: {augmented_count}")
    print(f"   - Total dataset size: {original_count + augmented_count}")
    print(f"   - Augmentation factor: {augmented_count / original_count:.2f}x")
    
    # Copy original images to augmented directory (optional)
    print("\n💡 Note: The augmented directory contains only the augmented images.")
    print("   For training, you should use both the original and augmented images.")
