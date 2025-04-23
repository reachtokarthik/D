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
    os.makedirs(path.join(PROJECT_ROOT, "data", "sorted"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "augmented"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "normalized"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "data", "split"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "outputs", "plots"), exist_ok=True)
    os.makedirs(path.join(PROJECT_ROOT, "outputs", "results"), exist_ok=True)
    
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
    
    # Step 4: Normalize images
    if not run_module("normalize"):
        print("Workflow stopped due to error in normalize.py")
        return
    
    # Step 5: Augment images
    if not run_module("imgaug"):
        print("Workflow stopped due to error in imgaug.py")
        return
    
    # Step 6: Split dataset
    if not run_module("split_dataset"):
        print("Workflow stopped due to error in split_dataset.py")
        return
    
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
