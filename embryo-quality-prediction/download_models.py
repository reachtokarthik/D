#!/usr/bin/env python
"""
Download script for embryo quality prediction model files.

This script downloads the pre-trained model files that are too large to be stored in GitHub.
Users need to run this script after cloning the repository to download the necessary model files.

Usage:
    python download_models.py

Note:
    You need to replace the placeholder URLs with actual URLs where you've hosted the model files.
    Recommended hosting options include:
    - Google Drive
    - Dropbox
    - Hugging Face Model Hub
    - AWS S3
    - Azure Blob Storage
"""

import os
import sys
import requests
from tqdm import tqdm
import shutil

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILES = {
    "resnet152_best.pth": {
        # Replace with actual URL where you've hosted the model
        "url": "REPLACE_WITH_ACTUAL_URL_FOR_BEST_MODEL",
        "description": "Best performing ResNet152 model checkpoint"
    },
    "resnet152_final.pth": {
        # Replace with actual URL where you've hosted the model
        "url": "REPLACE_WITH_ACTUAL_URL_FOR_FINAL_MODEL",
        "description": "Final ResNet152 model checkpoint"
    }
}

def download_file(url, destination):
    """
    Download a file from a URL with a progress bar.
    
    Args:
        url (str): URL to download from
        destination (str): Local path to save the file
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        print(f"Downloading to {destination}...")
        
        with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(size)
                
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        return False

def main():
    """Main function to download all model files."""
    print("Downloading model files for embryo quality prediction...")
    
    success = True
    for filename, info in MODEL_FILES.items():
        filepath = os.path.join(MODELS_DIR, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"File {filename} already exists. Skipping download.")
            continue
            
        print(f"\nDownloading {filename} - {info['description']}")
        if "REPLACE_WITH_ACTUAL_URL" in info["url"]:
            print(f"ERROR: You need to update the URL for {filename} in this script.")
            print("Please edit download_models.py and replace the placeholder URLs with actual URLs.")
            success = False
            continue
            
        success = download_file(info["url"], filepath) and success
    
    if success:
        print("\nAll model files downloaded successfully!")
    else:
        print("\nSome model files could not be downloaded. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
