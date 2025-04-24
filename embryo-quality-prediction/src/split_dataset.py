import os
import random
import shutil
import numpy as np
from tqdm import tqdm
import os.path as path
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Get the absolute path to the project root directory
SCRIPT_DIR = path.dirname(path.abspath(__file__))
PROJECT_ROOT = path.dirname(SCRIPT_DIR)

# Source directories (normalized and augmented images)
SOURCE_DIRS = [
    path.join(PROJECT_ROOT, "data", "normalized"),  # First try normalized
    path.join(PROJECT_ROOT, "data", "augmented"),   # Then try augmented
    path.join(PROJECT_ROOT, "data", "sorted")      # Finally try sorted as fallback
]

# Output directories
OUTPUT_BASE = path.join(PROJECT_ROOT, "data", "split")
TRAIN_DIR = path.join(OUTPUT_BASE, "train")
VAL_DIR = path.join(OUTPUT_BASE, "val")
TEST_DIR = path.join(OUTPUT_BASE, "test")

# Split ratios
TRAIN_RATIO = 0.8  # Updated to 80% as specified in the screenshot
VAL_RATIO = 0.1   # Updated to 10% as specified in the screenshot
TEST_RATIO = 0.1   # This will be calculated as 1 - (TRAIN_RATIO + VAL_RATIO)

# Random seed for reproducibility
RANDOM_SEED = 42

# Method to use for splitting
# Options: "random", "stratified"
SPLIT_METHOD = "stratified"

# Function to create output directories
def create_output_dirs(class_names):
    """Create output directories for train, val, and test sets"""
    # Create base directories
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Create class subdirectories in each split
    for class_name in class_names:
        os.makedirs(path.join(TRAIN_DIR, class_name), exist_ok=True)
        os.makedirs(path.join(VAL_DIR, class_name), exist_ok=True)
        os.makedirs(path.join(TEST_DIR, class_name), exist_ok=True)

# Function to collect dataset information
def collect_dataset_info(source_dirs):
    """Collect information about the dataset from multiple possible source directories"""
    print("\n📊 Analyzing dataset...")
    
    dataset_info = []
    class_names = set()  # Use a set to avoid duplicates
    
    # Try each source directory in order until we find one with data
    valid_source_found = False
    used_source_dir = None
    
    for source_dir in source_dirs:
        # Check if source directory exists
        if not os.path.exists(source_dir):
            print(f"⚠️ Source directory {source_dir} not found, trying next...")
            continue
            
        # Check if source directory has any subdirectories (class folders)
        subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        if not subdirs:
            print(f"⚠️ Source directory {source_dir} has no class subdirectories, trying next...")
            continue
            
        print(f"🔍 Using source directory: {source_dir}")
        valid_source_found = True
        used_source_dir = source_dir
        break
    
    if not valid_source_found:
        print(f"⚠️ No valid source directories found with class subdirectories")
        return dataset_info, list(class_names), None
    
    # Collect information about each class
    for class_name in os.listdir(used_source_dir):
        class_dir = os.path.join(used_source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_names.add(class_name)
        
        # Special logging for error_images class to track its processing
        if class_name == "error_images":
            print(f"🔍 Found error_images class directory: {class_dir}")
        
        # Collect information about each image in the class
        image_count = 0
        for filename in os.listdir(class_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            image_path = os.path.join(class_dir, filename)
            dataset_info.append({
                'filename': filename,
                'path': image_path,
                'class': class_name
            })
            image_count += 1
            
        # Log the number of images found in each class
        print(f"   - {class_name}: {image_count} images found")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(dataset_info)
    
    # Print dataset statistics
    if len(df) > 0:
        print(f"\n📊 Dataset statistics:")
        print(f"   - Total images: {len(df)}")
        
        # Print class distribution
        class_counts = df['class'].value_counts()
        for class_name, count in class_counts.items():
            print(f"   - {class_name}: {count} images ({count/len(df)*100:.1f}%)")
    else:
        print(f"\n⚠️ No images found in any of the source directories")
    
    return df, list(class_names), used_source_dir

# Function to perform random split
def random_split(df, train_ratio, val_ratio):
    """Split the dataset randomly"""
    # First split: train vs. (val + test)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=RANDOM_SEED
    )
    
    # Second split: val vs. test from the remaining data
    # Calculate the ratio of validation samples from the temp_df
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    
    val_df, test_df = train_test_split(
        temp_df, 
        train_size=val_ratio_adjusted, 
        random_state=RANDOM_SEED
    )
    
    return train_df, val_df, test_df

# Function to perform stratified split
def stratified_split(df, train_ratio, val_ratio):
    """Split the dataset with stratification to maintain class distribution"""
    # First split: train vs. (val + test)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio, 
        stratify=df['class'],
        random_state=RANDOM_SEED
    )
    
    # Second split: val vs. test from the remaining data
    # Calculate the ratio of validation samples from the temp_df
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    
    val_df, test_df = train_test_split(
        temp_df, 
        train_size=val_ratio_adjusted, 
        stratify=temp_df['class'],
        random_state=RANDOM_SEED
    )
    
    return train_df, val_df, test_df

# Function to copy images to their respective split directories
def copy_images(df, split_name, target_dir, used_source_dir):
    """Copy images to the appropriate split directory"""
    print(f"\n💾 Copying {split_name} images from {used_source_dir}...")
    
    # Track counts per class for verification
    class_counts = {}
    success_count = 0
    error_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split_name} set"):
        source_path = row['path']
        class_name = row['class']
        filename = row['filename']
        
        # Verify source file exists
        if not os.path.exists(source_path):
            print(f"⚠️ Source file does not exist: {source_path}")
            # Try to find the file in the normalized directory
            normalized_path = os.path.join(PROJECT_ROOT, "data", "normalized", class_name, filename)
            if os.path.exists(normalized_path):
                print(f"   🔍 Found file in normalized directory instead: {normalized_path}")
                source_path = normalized_path
            else:
                print(f"   ❌ Could not find file in normalized directory either")
                error_count += 1
                continue
        
        # Track counts
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
        
        # Special logging for error_images
        if class_name == "error_images":
            print(f"🔍 Copying error image to {split_name} set: {filename}")
        
        # Ensure target directory exists
        target_class_dir = os.path.join(target_dir, class_name)
        if not os.path.exists(target_class_dir):
            print(f"   📁 Creating missing class directory: {target_class_dir}")
            os.makedirs(target_class_dir, exist_ok=True)
        
        target_path = os.path.join(target_class_dir, filename)
        
        # Copy the image
        try:
            shutil.copy2(source_path, target_path)
            success_count += 1
        except Exception as e:
            print(f"⚠️ Error copying {source_path} to {target_path}: {e}")
            error_count += 1
    
    # Print summary of copied images by class
    print(f"\n📊 {split_name} set class distribution:")
    for class_name, count in class_counts.items():
        print(f"   - {class_name}: {count} images copied")
    
    print(f"   - Total successful copies: {success_count}")
    print(f"   - Total errors: {error_count}")
        
    # Special check for error_images
    if "error_images" in class_counts:
        print(f"🔍 Successfully copied {class_counts['error_images']} error images to {split_name} set")
    else:
        print(f"⚠️ Warning: No error_images were copied to the {split_name} set!")
        
    # Verify files were actually copied
    actual_count = 0
    for class_name in class_counts.keys():
        class_dir = os.path.join(target_dir, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            actual_count += len(files)
            print(f"   - Verified {len(files)} files in {class_name} directory")
    
    if actual_count == 0:
        print(f"❌ ERROR: No files were actually copied to {split_name} directory!")
    elif actual_count != success_count:
        print(f"⚠️ Warning: Expected {success_count} files but found {actual_count} files in {split_name} directory")

# Main function to split the dataset
def split_dataset():
    """Split the dataset into train, validation, and test sets"""
    # Collect dataset information from multiple possible sources
    df, class_names, used_source_dir = collect_dataset_info(SOURCE_DIRS)
    
    if len(df) == 0 or used_source_dir is None:
        print("⚠️ No images found in any of the source directories")
        return
    
    # Create output directories
    create_output_dirs(class_names)
    
    # Split the dataset
    print(f"\n🔍 Splitting dataset using {SPLIT_METHOD} method...")
    print(f"   - Train: {TRAIN_RATIO*100:.0f}%")
    print(f"   - Validation: {VAL_RATIO*100:.0f}%")
    print(f"   - Test: {(1-TRAIN_RATIO-VAL_RATIO)*100:.0f}%")
    
    if SPLIT_METHOD == "random":
        train_df, val_df, test_df = random_split(df, TRAIN_RATIO, VAL_RATIO)
    elif SPLIT_METHOD == "stratified":
        train_df, val_df, test_df = stratified_split(df, TRAIN_RATIO, VAL_RATIO)
    else:
        print(f"\u26a0\ufe0f Unknown split method: {SPLIT_METHOD}")
        return
    
    # Print split statistics
    print(f"\n📊 Split statistics:")
    print(f"   - Train set: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   - Validation set: {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   - Test set: {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")
    print(f"   - Source directory: {used_source_dir}")
    
    # Print class distribution in each split
    print("\n📊 Class distribution in train set:")
    train_class_counts = train_df['class'].value_counts()
    for class_name, count in train_class_counts.items():
        print(f"   - {class_name}: {count} images ({count/len(train_df)*100:.1f}%)")
    
    print("\n📊 Class distribution in validation set:")
    val_class_counts = val_df['class'].value_counts()
    for class_name, count in val_class_counts.items():
        print(f"   - {class_name}: {count} images ({count/len(val_df)*100:.1f}%)")
    
    print("\n📊 Class distribution in test set:")
    test_class_counts = test_df['class'].value_counts()
    for class_name, count in test_class_counts.items():
        print(f"   - {class_name}: {count} images ({count/len(test_df)*100:.1f}%)")
    
    # Copy images to their respective split directories
    copy_images(train_df, "train", TRAIN_DIR, used_source_dir)
    copy_images(val_df, "validation", VAL_DIR, used_source_dir)
    copy_images(test_df, "test", TEST_DIR, used_source_dir)
    
    print("\n✅ Dataset splitting complete!")
    print(f"   - Train set: {len(train_df)} images")
    print(f"   - Validation set: {len(val_df)} images")
    print(f"   - Test set: {len(test_df)} images")
    print(f"   - Source directory used: {used_source_dir}")
    print(f"\n💡 The split dataset is available at: {OUTPUT_BASE}")

def main():
    print("🔍 Starting dataset splitting process...")
    
    # First, ensure the output directories exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Check if any of the output directories already have data
    train_has_data = os.path.exists(TRAIN_DIR) and len(os.listdir(TRAIN_DIR)) > 0
    val_has_data = os.path.exists(VAL_DIR) and len(os.listdir(VAL_DIR)) > 0
    test_has_data = os.path.exists(TEST_DIR) and len(os.listdir(TEST_DIR)) > 0
    
    if train_has_data and val_has_data and test_has_data:
        print("\n✅ Split directories already contain data. Skipping split process.")
        print(f"   - Train directory: {TRAIN_DIR}")
        print(f"   - Validation directory: {VAL_DIR}")
        print(f"   - Test directory: {TEST_DIR}")
    else:
        # Run the split process
        split_dataset()

# Run the script
if __name__ == "__main__":
    main()
