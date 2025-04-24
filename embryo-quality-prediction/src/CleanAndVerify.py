import os
import cv2
import hashlib
from PIL import Image
from tqdm import tqdm

# Your base image folder (e.g., data/sorted/)
import os.path as path

# Get the absolute path to the project root directory
SCRIPT_DIR = path.dirname(path.abspath(__file__))
PROJECT_ROOT = path.dirname(SCRIPT_DIR)

BASE_DIR = path.join(PROJECT_ROOT, "data", "sorted")
TARGET_SIZE = (224, 224)

# Maintain hash to detect duplicates
image_hashes = set()

def is_image_valid(image_path):
    try:
        img = Image.open(image_path)
        img.verify()  # PIL throws error if corrupt
        return True
    except Exception:
        return False

def compute_image_hash(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def clean_and_resize_images(base_dir):
    total_removed = 0
    error_images_count = 0
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Track if we're processing error images
        is_error_folder = folder == "error_images"
        if is_error_folder:
            print(f"🔍 Processing error images folder: {folder}")

        for filename in tqdm(os.listdir(folder_path), desc=f"📁 Processing {folder}"):
            file_path = os.path.join(folder_path, filename)

            # 1. Remove unreadable images
            if not is_image_valid(file_path):
                os.remove(file_path)
                print(f"🗑️ Removed corrupted image: {file_path}")
                total_removed += 1
                continue

            # 2. Remove duplicates
            image_hash = compute_image_hash(file_path)
            if image_hash in image_hashes:
                os.remove(file_path)
                print(f"🔁 Removed duplicate: {file_path}")
                total_removed += 1
                continue
            image_hashes.add(image_hash)

            # 3. Resize image to 224x224 and overwrite
            try:
                img = Image.open(file_path)
                img = img.convert("RGB")
                img = img.resize(TARGET_SIZE, Image.LANCZOS)  # LANCZOS is the recommended replacement for ANTIALIAS
                img.save(file_path)
                
                # Count error images that were successfully processed
                if is_error_folder:
                    error_images_count += 1
            except Exception as e:
                print(f"⚠️ Resize error for {file_path}: {e}")
                os.remove(file_path)

    print(f"\n✅ Cleaning complete. Total removed: {total_removed}")
    if error_images_count > 0:
        print(f"🔍 Successfully processed {error_images_count} error images")

# Run it
clean_and_resize_images(BASE_DIR)
