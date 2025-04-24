#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Embryo Classification Complete Workflow
This script runs the entire embryo classification workflow in sequence:
1. Label data
2. Check image sizes
3. Clean and verify data
4. Normalize images
5. Augment images
6. Split dataset
7. Train model
"""

import os
import sys
import time
import importlib
from os import path
from datetime import datetime
import torch

# Get the absolute path to the project root directory
SCRIPT_DIR = path.dirname(path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

# Add src directory to path for imports
sys.path.append(path.join(PROJECT_ROOT, "src"))

def print_section_header(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)
    
def run_module(module_name, module_path=None):
    """Run a module by importing and executing its main code"""
    print_section_header(f"Running {module_name}")
    
    start_time = time.time()
    
    try:
        # If module_path is provided, use it to import the module
        if module_path:
            sys.path.insert(0, path.dirname(module_path))
            module = importlib.import_module(path.basename(module_path).replace('.py', ''))
            sys.path.pop(0)
        else:
            # Otherwise import from src package
            module = importlib.import_module(f"src.{module_name}")
        
        # If the module has a main function, call it
        if hasattr(module, 'main'):
            module.main()
            
        elapsed_time = time.time() - start_time
        print(f"\n✅ {module_name} completed successfully in {elapsed_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"\n❌ Error in {module_name}: {str(e)}")
        return False

def main():
    """Run the complete embryo classification workflow"""
    print_section_header("EMBRYO CLASSIFICATION WORKFLOW")
    print(f"Starting workflow at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}")
    
    # Create necessary directories
    os.makedirs(path.join(PROJECT_ROOT, "data", "raw"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "raw", "roboflow"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "raw", "other"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "sorted"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "augmented"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "normalized"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "split"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "outputs", "plots"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "outputs", "results"), exist_ok=True)
    
    # Check if data directories exist and provide dataset selection options
    roboflow_dir = path.join(PROJECT_ROOT, "data", "raw", "roboflow")
    other_dir = path.join(PROJECT_ROOT, "data", "raw", "other")
    
    roboflow_has_data = os.path.exists(roboflow_dir) and len(os.listdir(roboflow_dir)) > 0
    other_has_data = os.path.exists(other_dir) and len(os.listdir(other_dir)) > 0
    
    # Set environment variable for dataset selection
    os.environ["EMBRYO_DATASET_SELECTION"] = ""
    
    if not roboflow_has_data and not other_has_data:
        print("\n⚠️ Warning: No data found in either raw/roboflow or raw/other directories.")
        print("Please ensure you have placed your data in at least one of these directories before proceeding.")
        user_input = input("Do you want to continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            print("Workflow stopped due to missing data")
            return
    else:
        print("\n" + "=" * 50)
        print("DATASET SELECTION")
        print("=" * 50)
        
        available_options = []
        if roboflow_has_data:
            available_options.append("roboflow")
            print("1. Roboflow dataset (original dataset)")
        
        if other_has_data:
            available_options.append("other")
            print(f"{2 if roboflow_has_data else 1}. Other dataset (new dataset)")
        
        if roboflow_has_data and other_has_data:
            available_options.append("both")
            print("3. Both datasets")
        
        while True:
            try:
                selection = input("\nSelect which dataset to use (enter the number): ")
                selection_idx = int(selection) - 1
                
                if 0 <= selection_idx < len(available_options):
                    dataset_choice = available_options[selection_idx]
                    print(f"Selected: {dataset_choice.capitalize()} dataset")
                    os.environ["EMBRYO_DATASET_SELECTION"] = dataset_choice
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    # Phase 1: Data Preparation
    print_section_header("PHASE 1: DATA PREPARATION")
    
    # Step 1: Label data
    if not run_module("label"):
        print("Workflow stopped due to error in label.py")
        return
    
    # Step 2: Check image sizes
    if not run_module("check_image_size"):
        print("Workflow stopped due to error in check_image_size.py")
        return
    
    # Step 3: Clean and verify data
    if not run_module("CleanAndVerify"):
        print("Workflow stopped due to error in CleanAndVerify.py")
        return
    
    # Step 4: Augment images
    if not run_module("imgaug"):
        print("Workflow stopped due to error in imgaug.py")
        return
    
    # Step 5: Normalize images (now after augmentation)
    if not run_module("normalize"):
        print("Workflow stopped due to error in normalize.py")
        return
    
    # Step 6: Split dataset
    if not run_module("split_dataset"):
        print("Workflow stopped due to error in split_dataset.py")
        return
    
    # Verify split directories exist and contain data
    split_dir = path.join(PROJECT_ROOT, "data", "split")
    train_dir = path.join(split_dir, "train")
    val_dir = path.join(split_dir, "val")
    test_dir = path.join(split_dir, "test")
    
    # Check if directories exist and contain class subdirectories
    train_has_classes = path.exists(train_dir) and len(os.listdir(train_dir)) > 0
    val_has_classes = path.exists(val_dir) and len(os.listdir(val_dir)) > 0
    test_has_classes = path.exists(test_dir) and len(os.listdir(test_dir)) > 0
    
    if not (train_has_classes and val_has_classes and test_has_classes):
        print("\n⚠️ Warning: Split directories are missing or empty. Running split_dataset again...")
        if not run_module("split_dataset"):
            print("Workflow stopped due to error in split_dataset.py")
            return
        
        # Double-check after running split_dataset
        train_has_classes = path.exists(train_dir) and len(os.listdir(train_dir)) > 0
        val_has_classes = path.exists(val_dir) and len(os.listdir(val_dir)) > 0
        test_has_classes = path.exists(test_dir) and len(os.listdir(test_dir)) > 0
        
        if not (train_has_classes and val_has_classes and test_has_classes):
            print("\n❌ Error: Split directories still missing or empty after running split_dataset.py")
            print("Please ensure you have data in the 'data/augmented' directory")
            return
        else:
            print("\n✅ Split directories successfully created and contain data.")
    
    # Phase 2: Model Development
    print_section_header("PHASE 2: MODEL DEVELOPMENT")
    
    # Check GPU availability before training
    print_section_header("GPU CHECK")
    print("Checking GPU availability...")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
        print("✅ GPU is available for training")
    else:
        print("⚠️ WARNING: No GPU available. Training will be slow on CPU.")
        user_input = input("Do you want to continue with CPU training? (y/n): ")
        if user_input.lower() != 'y':
            print("Workflow stopped by user due to no GPU availability")
            return
    
    # Step 7: Train model
    if not run_module("train_model"):
        print("Workflow stopped due to error in train_model.py")
        return
    
    # Workflow complete
    print_section_header("WORKFLOW COMPLETE")
    print(f"Workflow completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("All steps executed successfully!")

if __name__ == "__main__":
    main()
