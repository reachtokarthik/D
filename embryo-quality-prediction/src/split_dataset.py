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

# Source directory (augmented images)
SOURCE_DIR = path.join(PROJECT_ROOT, "data", "augmented")

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
def collect_dataset_info(source_dir):
    """Collect information about the dataset"""
    print("\nud83dudcca Analyzing dataset...")
    
    dataset_info = []
    class_names = []
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"\u26a0\ufe0f Source directory {source_dir} not found")
        return dataset_info, class_names
    
    # Collect information about each class
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_names.append(class_name)
        
        # Collect information about each image in the class
        for filename in os.listdir(class_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            image_path = os.path.join(class_dir, filename)
            dataset_info.append({
                'filename': filename,
                'path': image_path,
                'class': class_name
            })
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(dataset_info)
    
    # Print dataset statistics
    print(f"\nud83dudcca Dataset statistics:")
    print(f"   - Total images: {len(df)}")
    
    # Print class distribution
    class_counts = df['class'].value_counts()
    for class_name, count in class_counts.items():
        print(f"   - {class_name}: {count} images ({count/len(df)*100:.1f}%)")
    
    return df, class_names

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
def copy_images(df, split_name, target_dir):
    """Copy images to the appropriate split directory"""
    print(f"\nud83dudcbe Copying {split_name} images...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split_name} set"):
        source_path = row['path']
        class_name = row['class']
        filename = row['filename']
        
        target_path = os.path.join(target_dir, class_name, filename)
        
        # Copy the image
        try:
            shutil.copy2(source_path, target_path)
        except Exception as e:
            print(f"\u26a0\ufe0f Error copying {source_path} to {target_path}: {e}")

# Main function to split the dataset
def split_dataset():
    """Split the dataset into train, validation, and test sets"""
    # Collect dataset information
    df, class_names = collect_dataset_info(SOURCE_DIR)
    
    if len(df) == 0:
        print("\u26a0\ufe0f No images found in the source directory")
        return
    
    # Create output directories
    create_output_dirs(class_names)
    
    # Split the dataset
    print(f"\nud83dudd0e Splitting dataset using {SPLIT_METHOD} method...")
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
    print(f"\nud83dudcca Split statistics:")
    print(f"   - Train set: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   - Validation set: {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   - Test set: {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")
    
    # Print class distribution in each split
    print("\nud83dudcca Class distribution in train set:")
    train_class_counts = train_df['class'].value_counts()
    for class_name, count in train_class_counts.items():
        print(f"   - {class_name}: {count} images ({count/len(train_df)*100:.1f}%)")
    
    print("\nud83dudcca Class distribution in validation set:")
    val_class_counts = val_df['class'].value_counts()
    for class_name, count in val_class_counts.items():
        print(f"   - {class_name}: {count} images ({count/len(val_df)*100:.1f}%)")
    
    print("\nud83dudcca Class distribution in test set:")
    test_class_counts = test_df['class'].value_counts()
    for class_name, count in test_class_counts.items():
        print(f"   - {class_name}: {count} images ({count/len(test_df)*100:.1f}%)")
    
    # Copy images to their respective split directories
    copy_images(train_df, "train", TRAIN_DIR)
    copy_images(val_df, "validation", VAL_DIR)
    copy_images(test_df, "test", TEST_DIR)
    
    print("\n\u2705 Dataset splitting complete!")
    print(f"   - Train set: {len(train_df)} images")
    print(f"   - Validation set: {len(val_df)} images")
    print(f"   - Test set: {len(test_df)} images")
    print(f"\nud83dudca1 The split dataset is available at: {OUTPUT_BASE}")

# Run the script
if __name__ == "__main__":
    print("ud83dudd0d Starting dataset splitting process...")
    split_dataset()
