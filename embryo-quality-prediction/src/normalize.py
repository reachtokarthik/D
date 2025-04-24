import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import os.path as path

# Get the absolute path to the project root directory
SCRIPT_DIR = path.dirname(path.abspath(__file__))
PROJECT_ROOT = path.dirname(SCRIPT_DIR)

# Source directories - both original and augmented images
SOURCE_DIRS = [
    path.join(PROJECT_ROOT, "data", "sorted"),
    path.join(PROJECT_ROOT, "data", "augmented")
]

# Output directory for normalized images
NORMALIZED_DIR = path.join(PROJECT_ROOT, "data", "normalized")

# Target size for all images
TARGET_SIZE = (224, 224)

# Normalization types
# Options: "simple", "imagenet", "dataset_stats"
# simple: just divide by 255 to get 0-1 range
# imagenet: normalize using ImageNet statistics (good for transfer learning)
# dataset_stats: calculate and use dataset-specific statistics
NORM_TYPE = "simple"  # Starting with simple normalization for testing

# ImageNet mean and std for each channel (RGB)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Function to calculate dataset statistics
def calculate_dataset_stats(source_dirs):
    """Calculate mean and standard deviation across the entire dataset"""
    print("\n📊 Calculating dataset statistics...")
    
    # Initialize variables for calculating mean and std
    pixel_sum = np.zeros(3)  # RGB channels
    pixel_sum_squared = np.zeros(3)  # For calculating std
    pixel_count = 0
    
    # First pass: calculate mean
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            continue
            
        for folder in os.listdir(source_dir):
            folder_path = os.path.join(source_dir, folder)
            if not os.path.isdir(folder_path):
                continue
                
            for filename in tqdm(os.listdir(folder_path), desc=f"Calculating mean for {folder}"):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = os.path.join(folder_path, filename)
                try:
                    # Read image and convert to RGB
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Normalize to 0-1 range
                    img = img.astype(np.float32) / 255.0
                    
                    # Update sums
                    pixel_sum += np.sum(img, axis=(0, 1))
                    pixel_sum_squared += np.sum(np.square(img), axis=(0, 1))
                    pixel_count += img.shape[0] * img.shape[1]
                except Exception as e:
                    print(f"⚠️ Error processing {img_path}: {e}")
    
    # Calculate mean and std
    mean = pixel_sum / pixel_count
    std = np.sqrt((pixel_sum_squared / pixel_count) - np.square(mean))
    
    print(f"\n📊 Dataset statistics:\n   - Mean: {mean}\n   - Std: {std}")
    return mean, std

# Function to normalize an image
def normalize_image(img, norm_type, mean=None, std=None):
    """Normalize an image based on the specified normalization type"""
    # Ensure image is in the correct format
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    
    # Scale to 0-1 if not already
    if img.max() > 1.0:
        img = img / 255.0
    
    if norm_type == "simple":
        # Simple normalization: just scale to 0-1
        return img
    elif norm_type == "imagenet":
        # ImageNet normalization - apply per channel
        normalized = np.zeros_like(img)
        for i in range(3):  # RGB channels
            normalized[:,:,i] = (img[:,:,i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
        return normalized
    elif norm_type == "dataset_stats":
        # Custom dataset normalization - apply per channel
        normalized = np.zeros_like(img)
        for i in range(3):  # RGB channels
            normalized[:,:,i] = (img[:,:,i] - mean[i]) / std[i]
        return normalized
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

# Function to save a normalized image
def save_normalized_image(img, output_path):
    """Save a normalized image (converting back to 0-255 range for storage)"""
    try:
        # For visualization and storage, convert back to 0-255 range
        if img.min() < 0 or img.max() > 1:
            # This is a normalized image with values outside 0-1 range
            # Rescale to 0-1 for visualization
            img = (img - img.min()) / (img.max() - img.min())
        
        # Convert to 0-255 range
        img = (img * 255).astype(np.uint8)
        
        # Save the image
        cv2.imwrite(output_path, img)
        return True
    except Exception as e:
        print(f"⚠️ Error saving image to {output_path}: {e}")
        return False

# Main function to normalize all images
def normalize_dataset():
    """Normalize all images in the dataset"""
    # Calculate dataset statistics if needed
    if NORM_TYPE == "dataset_stats":
        mean, std = calculate_dataset_stats(SOURCE_DIRS)
    else:
        mean, std = None, None
    
    # Process each source directory
    processed_files = 0
    skipped_files = 0
    
    for source_dir in SOURCE_DIRS:
        if not os.path.exists(source_dir):
            print(f"⚠️ Source directory {source_dir} not found")
            continue
        
        # Check if directory is empty
        if len(os.listdir(source_dir)) == 0:
            print(f"⚠️ Source directory {source_dir} is empty")
            continue
            
        print(f"\n🔄 Processing images from {source_dir}")
        
        # Process each class folder
        for folder in os.listdir(source_dir):
            src_folder = os.path.join(source_dir, folder)
            if not os.path.isdir(src_folder):
                continue
                
            # Create output folder
            dest_folder = os.path.join(NORMALIZED_DIR, folder)
            os.makedirs(dest_folder, exist_ok=True)
            
            # Process each image
            for filename in tqdm(os.listdir(src_folder), desc=f"Normalizing {folder}"):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                src_path = os.path.join(src_folder, filename)
                dest_path = os.path.join(dest_folder, filename)
                
                # Skip if already processed
                if os.path.exists(dest_path):
                    skipped_files += 1
                    continue
                    
                try:
                    # Read image using PIL instead of OpenCV for better compatibility
                    pil_img = Image.open(src_path)
                    pil_img = pil_img.convert('RGB')  # Ensure RGB format
                    
                    # Resize if needed
                    if pil_img.size != TARGET_SIZE:
                        pil_img = pil_img.resize(TARGET_SIZE)
                        
                    # Convert PIL image to numpy array
                    img = np.array(pil_img)
                    
                    # Normalize the image
                    normalized_img = normalize_image(img, NORM_TYPE, mean, std)
                    
                    # Convert back to BGR for saving with OpenCV
                    normalized_img = cv2.cvtColor(normalized_img, cv2.COLOR_RGB2BGR)
                    
                    # Save the normalized image
                    if save_normalized_image(normalized_img, dest_path):
                        processed_files += 1
                    
                except Exception as e:
                    print(f"⚠️ Error normalizing {src_path}: {e}")
    
    # Count normalized images
    total_normalized = 0
    for folder in os.listdir(NORMALIZED_DIR):
        folder_path = os.path.join(NORMALIZED_DIR, folder)
        if os.path.isdir(folder_path):
            count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            total_normalized += count
            print(f"   - {folder}: {count} images")
    
    print(f"\n✅ Normalization complete: {total_normalized} images normalized using {NORM_TYPE} normalization")
    print(f"   - Files processed: {processed_files}")
    print(f"   - Files skipped (already normalized): {skipped_files}")
    
    if NORM_TYPE == "imagenet":
        print("\n💡 Note: Images are normalized using ImageNet statistics:")
        print(f"   - Mean: {IMAGENET_MEAN}")
        print(f"   - Std: {IMAGENET_STD}")
        print("   This is optimal if you're using pretrained models like ResNet, VGG, etc.")
    elif NORM_TYPE == "dataset_stats":
        print("\n💡 Note: Images are normalized using custom dataset statistics:")
        print(f"   - Mean: {mean}")
        print(f"   - Std: {std}")

def main():
    """Main function to be called from the workflow"""
    print("🔍 Starting image normalization process")
    print(f"   - Normalization type: {NORM_TYPE}")
    print(f"   - Target size: {TARGET_SIZE}")
    
    # Create output directory
    os.makedirs(NORMALIZED_DIR, exist_ok=True)
    
    # Normalize the dataset
    normalize_dataset()

# Run the normalization process
if __name__ == "__main__":
    main()
