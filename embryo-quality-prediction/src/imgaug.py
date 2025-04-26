import os
import cv2
import numpy as np
import albumentations as A
from PIL import Image
from tqdm import tqdm
import os.path as path
import random
import math

# Get the absolute path to the project root directory
SCRIPT_DIR = path.dirname(path.abspath(__file__))
PROJECT_ROOT = path.dirname(SCRIPT_DIR)

# Source and output folders with absolute paths
SOURCE_DIR = path.join(PROJECT_ROOT, "data", "sorted")
AUGMENTED_DIR = path.join(PROJECT_ROOT, "data", "augmented")
TARGET_SIZE = (224, 224)

# Target range for each class
MIN_IMAGES_PER_CLASS = 1000
MAX_IMAGES_PER_CLASS = 1200

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
    A.RandomCrop(height=150, width=150, p=0.7),  # Reduced crop size to handle smaller images
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.PadIfNeeded(min_height=TARGET_SIZE[0], min_width=TARGET_SIZE[1], p=1.0),
    A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1])
])

# List of all transform pipelines
transform_pipelines = [basic_transforms, color_transforms, geometric_transforms, crop_transforms]

# Function to count images in a directory
def count_images(directory):
    count = 0
    class_counts = {}
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            class_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[folder] = class_count
            count += class_count
    return count, class_counts

# Print dataset statistics before augmentation
if os.path.exists(SOURCE_DIR):
    original_total, original_class_counts = count_images(SOURCE_DIR)
    print(f"📊 Original dataset: {original_total} images")
    print("Class distribution:")
    for class_name, count in original_class_counts.items():
        print(f"   - {class_name}: {count} images")
else:
    print(f"⚠️ Source directory {SOURCE_DIR} not found")
    exit(1)

# Calculate exact augmentation needed for each class
augmentation_plans = {}
for folder, count in original_class_counts.items():
    
    # Calculate how many total images we need
    if count == 0:
        augmentation_plans[folder] = {"per_image": 0, "total_needed": 0}
    else:
        # Target is the minimum required (we'll include originals in final count)
        target_count = MIN_IMAGES_PER_CLASS - count  # How many augmented images we need
        
        # Make sure we don't exceed maximum
        max_possible = MAX_IMAGES_PER_CLASS - count
        
        # Calculate how many augmentations per original image
        aug_per_image = math.ceil(target_count / count)
        
        # Calculate how many total augmented images this would produce
        total_augmented = count * aug_per_image
        
        # If this exceeds our maximum, we need to adjust
        if total_augmented > max_possible:
            # Calculate exact number of augmentations needed
            aug_per_image = math.floor(max_possible / count)
            total_augmented = count * aug_per_image
            
            # We might need a few more to reach minimum
            remaining_needed = target_count - total_augmented
            
            augmentation_plans[folder] = {
                "per_image": aug_per_image,
                "total_needed": total_augmented,
                "extra_needed": max(0, remaining_needed),
                "max_possible": max_possible
            }
        else:
            augmentation_plans[folder] = {
                "per_image": aug_per_image,
                "total_needed": total_augmented,
                "extra_needed": 0,
                "max_possible": max_possible
            }
    
    print(f"   - {folder}: Will generate {augmentation_plans[folder]['per_image']} augmentations per image")
    if augmentation_plans[folder].get('extra_needed', 0) > 0:
        print(f"     Plus {augmentation_plans[folder]['extra_needed']} additional augmentations to reach minimum")

# Ensure target folders exist
for folder in os.listdir(SOURCE_DIR):
        
    src_folder = os.path.join(SOURCE_DIR, folder)
    if not os.path.isdir(src_folder):
        continue
        
    dest_folder = os.path.join(AUGMENTED_DIR, folder)
    os.makedirs(dest_folder, exist_ok=True)
    
    # Get augmentation plan for this class
    aug_plan = augmentation_plans.get(folder, {"per_image": 0, "total_needed": 0})
    aug_per_image = aug_plan.get("per_image", 0)
    extra_needed = aug_plan.get("extra_needed", 0)
    
    # Skip if no augmentation needed
    if aug_per_image <= 0 and extra_needed <= 0:
        print(f"Skipping {folder} - no augmentation needed")
        continue
    
    original_files = os.listdir(src_folder)
    for filename in tqdm(original_files, desc=f"✨ Augmenting {folder}"):
        src_path = os.path.join(src_folder, filename)
        try:
            img = cv2.imread(src_path)
            if img is None:
                continue

            # Apply standard augmentations per image
            for i in range(aug_per_image):
                # Select a random transform pipeline for each augmentation
                transform = random.choice(transform_pipelines)
                try:
                    augmented = transform(image=img)
                    aug_img = augmented["image"]
                    
                    # Create more descriptive filenames
                    transform_type = "basic" if transform == basic_transforms else "color" if transform == color_transforms else "geom" if transform == geometric_transforms else "crop"
                    aug_filename = f"{os.path.splitext(filename)[0]}_aug{i+1}_{transform_type}.jpg"
                    
                    # Save the augmented image
                    cv2.imwrite(os.path.join(dest_folder, aug_filename), aug_img)
                except Exception as e:
                    print(f"⚠️ Error in standard augmentation of {filename}: {e}")

        except Exception as e:
            print(f"⚠️ Error augmenting {filename}: {e}")

# Print statistics after augmentation
if os.path.exists(AUGMENTED_DIR):
    augmented_total, augmented_class_counts = count_images(AUGMENTED_DIR)
    print(f"\n📊 Augmentation complete:")
    print(f"   - Original images: {original_total}")
    print(f"   - Augmented images: {augmented_total}")
    
    # Copy original images to augmented directory to reach final dataset
    print("\n🔄 Copying original images to augmented directory...")
    for folder in original_class_counts.keys():
            
        src_folder = os.path.join(SOURCE_DIR, folder)
        if not os.path.isdir(src_folder):
            continue
            
        dest_folder = os.path.join(AUGMENTED_DIR, folder)
        os.makedirs(dest_folder, exist_ok=True)
        
        # Copy original images
        for filename in os.listdir(src_folder):
            src_path = os.path.join(src_folder, filename)
            dest_path = os.path.join(dest_folder, filename)
            if os.path.isfile(src_path) and not os.path.exists(dest_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img = cv2.imread(src_path)
                        if img is not None:
                            # Resize to target size for consistency
                            img = cv2.resize(img, TARGET_SIZE)
                            cv2.imwrite(dest_path, img)
                    except Exception as e:
                        print(f"⚠️ Error copying {filename}: {e}")
    
    # Get statistics after copying originals
    final_total, final_class_counts = count_images(AUGMENTED_DIR)
    print(f"\n📊 Final dataset statistics:")
    print(f"   - Total dataset size: {final_total} images")
    print("   - Class distribution:")
    
    # Check if any class needs additional augmentations or trimming
    for class_name, count in final_class_counts.items():
            
        original = original_class_counts.get(class_name, 0)
        
        # Check if we need to add extra augmentations to reach minimum
        if count < MIN_IMAGES_PER_CLASS:
            needed = MIN_IMAGES_PER_CLASS - count
            print(f"   - Adding {needed} more augmentations to {class_name} to reach minimum")
            
            src_folder = os.path.join(SOURCE_DIR, class_name)
            dest_folder = os.path.join(AUGMENTED_DIR, class_name)
            
            # Get list of original images
            original_files = [f for f in os.listdir(src_folder) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Add extra augmentations
            extra_added = 0
            while extra_added < needed and original_files:
                # Pick a random original image
                random_file = random.choice(original_files)
                src_path = os.path.join(src_folder, random_file)
                
                try:
                    img = cv2.imread(src_path)
                    if img is not None:
                        # Apply a random transform
                        transform = random.choice(transform_pipelines)
                        augmented = transform(image=img)
                        aug_img = augmented["image"]
                        
                        # Create unique filename
                        transform_type = "basic" if transform == basic_transforms else "color" if transform == color_transforms else "geom" if transform == geometric_transforms else "crop"
                        aug_filename = f"{os.path.splitext(random_file)[0]}_extra{extra_added}_{transform_type}.jpg"
                        
                        # Save the augmented image
                        cv2.imwrite(os.path.join(dest_folder, aug_filename), aug_img)
                        extra_added += 1
                except Exception as e:
                    print(f"⚠️ Error adding extra augmentation: {e}")
        
        # Check if we need to remove images to meet maximum
        elif count > MAX_IMAGES_PER_CLASS:
            excess = count - MAX_IMAGES_PER_CLASS
            print(f"   - Removing {excess} images from {class_name} to meet maximum")
            
            dest_folder = os.path.join(AUGMENTED_DIR, class_name)
            
            # Get list of augmented images (not originals)
            augmented_files = [f for f in os.listdir(dest_folder) 
                              if "_aug" in f or "_extra" in f]
            
            # Shuffle to randomly select files to remove
            random.shuffle(augmented_files)
            
            # Remove excess files
            for i in range(min(excess, len(augmented_files))):
                try:
                    os.remove(os.path.join(dest_folder, augmented_files[i]))
                except Exception as e:
                    print(f"⚠️ Error removing file: {e}")
    
    # Get final statistics after adjustments
    final_total, final_class_counts = count_images(AUGMENTED_DIR)
    print(f"\n📊 Final dataset statistics after adjustments:")
    print(f"   - Total dataset size: {final_total} images")
    print("   - Class distribution:")
    
    for class_name, count in final_class_counts.items():
        original = original_class_counts.get(class_name, 0)
        status = "✅" if MIN_IMAGES_PER_CLASS <= count <= MAX_IMAGES_PER_CLASS else "❌"
        print(f"      {status} {class_name}: {count} images (Original: {original}, Target: {MIN_IMAGES_PER_CLASS}-{MAX_IMAGES_PER_CLASS})")
        
    # Check if any class needs additional augmentation to reach minimum
    classes_needing_more = {cls: (MIN_IMAGES_PER_CLASS - count) 
                           for cls, count in final_class_counts.items() 
                           if count < MIN_IMAGES_PER_CLASS}
    
    if classes_needing_more:
        print("\n⚠️ Some classes still need more augmentation to reach the minimum target:")
        for cls, needed in classes_needing_more.items():
            print(f"   - {cls}: needs {needed} more images")
        print("\n💡 Consider running the script again or increasing the augmentation factor.")
    
    # Check if any class exceeds maximum
    classes_exceeding_max = {cls: (count - MAX_IMAGES_PER_CLASS) 
                            for cls, count in final_class_counts.items() 
                            if count > MAX_IMAGES_PER_CLASS}
    
    if classes_exceeding_max:
        print("\n⚠️ Some classes exceed the maximum target:")
        for cls, excess in classes_exceeding_max.items():
            print(f"   - {cls}: {excess} images over the maximum")
        print("\n💡 Consider manually removing some augmented images to meet the maximum requirement.")
