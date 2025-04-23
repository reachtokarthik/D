from PIL import Image
import os

# Get the absolute path to the project root directory
import os.path as path
SCRIPT_DIR = path.dirname(path.abspath(__file__))
PROJECT_ROOT = path.dirname(SCRIPT_DIR)

# Check an image from each class
classes = ['blastocyst_grade_A', 'morula_grade_A', '8cell_grade_A']

for class_name in classes:
    class_dir = path.join(PROJECT_ROOT, 'data', 'sorted', class_name)
    if not os.path.exists(class_dir):
        print(f"Directory {class_dir} does not exist")
        continue
        
    # Get the first image in the directory
    files = os.listdir(class_dir)
    if not files:
        print(f"No images found in {class_dir}")
        continue
        
    # Check the first image
    image_path = path.join(class_dir, files[0])
    try:
        img = Image.open(image_path)
        print(f"Class: {class_name}, Image: {files[0]}, Size: {img.size}")
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
