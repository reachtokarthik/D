import os
import shutil
import os.path as path

# Get the absolute path to the project root directory
SCRIPT_DIR = path.dirname(path.abspath(__file__))
PROJECT_ROOT = path.dirname(SCRIPT_DIR)

# Use absolute paths based on the project root
ROBOFLOW_SOURCE = path.join(PROJECT_ROOT, "data", "raw", "roboflow")
OTHER_SOURCE = path.join(PROJECT_ROOT, "data", "raw", "other")
DEST_BASE = path.join(PROJECT_ROOT, "data", "sorted")

# Folder mapping from original roboflow labels to unified class folders
roboflow_label_map = {
    "2-1-1": "blastocyst_grade_A",
    "2-1-2": "blastocyst_grade_B",
    "2-1-3": "blastocyst_grade_C",
    "2-2-1": "blastocyst_grade_A",
    "2-2-2": "blastocyst_grade_B",
    "2-2-3": "blastocyst_grade_C",
    "2-3-3": "blastocyst_grade_C",
    "3-1-1": "blastocyst_grade_A",
    "3-1-2": "blastocyst_grade_B",
    "3-1-3": "blastocyst_grade_C",
    "3-2-1": "blastocyst_grade_A",
    "3-2-2": "blastocyst_grade_B",
    "3-2-3": "blastocyst_grade_C",
    "3-3-2": "blastocyst_grade_B",
    "3-3-3": "blastocyst_grade_C",
    "4-2-2": "blastocyst_grade_B",
    "Morula": "morula_grade_A",
    "Early": "8cell_grade_A",
    "Error image": "error_images",
    "Arrested": None  # Skip this category
}

# Mapping for the new dataset structure
other_dataset_map = {
    "Blastocyst": {
        "Grade A": "blastocyst_grade_A",
        "Grade B": "blastocyst_grade_B",
        "Grade C": "blastocyst_grade_C"
    },
    "Cleavage": {
        "Cleavage": {
            "Grade A": "8cell_grade_A",
            "Grade B": "8cell_grade_B",
            "Grade C": "8cell_grade_C"
        }
    },
    "Morula": {
        "Grade A": "morula_grade_A",
        "Grade B": "morula_grade_B",
        "Grade C": "morula_grade_C"
    },
    "Error image": "error_images"
}

# Create all target folders
def create_target_folders():
    # Collect all possible target folders from both mappings
    target_folders = set()
    
    # Add folders from roboflow mapping
    for folder in roboflow_label_map.values():
        if folder:
            target_folders.add(folder)
    
    # Add folders from other dataset mapping
    for main_category, subcategories in other_dataset_map.items():
        if isinstance(subcategories, str):
            target_folders.add(subcategories)
        elif isinstance(subcategories, dict):
            for subcategory, subsubcategories in subcategories.items():
                if isinstance(subsubcategories, str):
                    target_folders.add(subsubcategories)
                elif isinstance(subsubcategories, dict):
                    for _, target_folder in subsubcategories.items():
                        target_folders.add(target_folder)
    
    # Create all target folders
    for folder in target_folders:
        os.makedirs(os.path.join(DEST_BASE, folder), exist_ok=True)
        print(f"📁 Created target folder: {folder}")

# Process the roboflow dataset
def process_roboflow_dataset():
    print("\n" + "=" * 50)
    print("Processing Roboflow dataset...")
    print("=" * 50)
    
    if not os.path.exists(ROBOFLOW_SOURCE):
        print(f"⚠️ Roboflow dataset directory not found: {ROBOFLOW_SOURCE}")
        return
    
    # Walk through folders and move images
    for label_folder in os.listdir(ROBOFLOW_SOURCE):
        source_folder_path = os.path.join(ROBOFLOW_SOURCE, label_folder)

        if not os.path.isdir(source_folder_path):
            continue

        mapped_label = roboflow_label_map.get(label_folder, None)
        if not mapped_label:
            print(f"⚠️ Skipping folder '{label_folder}' - no mapped class")
            continue

        dest_folder_path = os.path.join(DEST_BASE, mapped_label)

        for filename in os.listdir(source_folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                src_file = os.path.join(source_folder_path, filename)
                dest_file = os.path.join(dest_folder_path, filename)

                if os.path.exists(dest_file):
                    print(f"🔁 Skipped duplicate: {filename}")
                else:
                    shutil.copy(src_file, dest_file)
                    print(f"✅ Copied {filename} → {mapped_label}")

# Process the new dataset
def process_other_dataset():
    print("\n" + "=" * 50)
    print("Processing Other dataset...")
    print("=" * 50)
    
    if not os.path.exists(OTHER_SOURCE):
        print(f"⚠️ Other dataset directory not found: {OTHER_SOURCE}")
        return
    
    # Process each main category
    for main_category in os.listdir(OTHER_SOURCE):
        main_category_path = os.path.join(OTHER_SOURCE, main_category)
        
        if not os.path.isdir(main_category_path):
            continue
        
        # Handle Error image folder (which contains direct image files)
        if main_category == "Error image":
            dest_folder = os.path.join(DEST_BASE, "error_images")
            for filename in os.listdir(main_category_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    src_file = os.path.join(main_category_path, filename)
                    dest_file = os.path.join(dest_folder, filename)
                    
                    if os.path.exists(dest_file):
                        print(f"🔁 Skipped duplicate: {filename}")
                    else:
                        shutil.copy(src_file, dest_file)
                        print(f"✅ Copied {filename} → error_images")
            continue
        
        # Handle other categories with subcategories
        subcategories = other_dataset_map.get(main_category, {})
        if not subcategories:
            print(f"⚠️ Skipping unknown category: {main_category}")
            continue
        
        # Special handling for Cleavage which has an extra nesting level
        if main_category == "Cleavage":
            cleavage_subdir = os.path.join(main_category_path, "Cleavage")
            if os.path.exists(cleavage_subdir) and os.path.isdir(cleavage_subdir):
                for grade_folder in os.listdir(cleavage_subdir):
                    grade_path = os.path.join(cleavage_subdir, grade_folder)
                    if not os.path.isdir(grade_path):
                        continue
                    
                    target_class = subcategories["Cleavage"].get(grade_folder)
                    if not target_class:
                        print(f"⚠️ Skipping unknown grade: {grade_folder} in Cleavage")
                        continue
                    
                    dest_folder = os.path.join(DEST_BASE, target_class)
                    for filename in os.listdir(grade_path):
                        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                            src_file = os.path.join(grade_path, filename)
                            dest_file = os.path.join(dest_folder, filename)
                            
                            if os.path.exists(dest_file):
                                print(f"🔁 Skipped duplicate: {filename}")
                            else:
                                shutil.copy(src_file, dest_file)
                                print(f"✅ Copied {filename} → {target_class}")
        else:
            # Process regular categories (Blastocyst, Morula)
            for grade_folder in os.listdir(main_category_path):
                grade_path = os.path.join(main_category_path, grade_folder)
                if not os.path.isdir(grade_path):
                    continue
                
                target_class = subcategories.get(grade_folder)
                if not target_class:
                    print(f"⚠️ Skipping unknown grade: {grade_folder} in {main_category}")
                    continue
                
                dest_folder = os.path.join(DEST_BASE, target_class)
                for filename in os.listdir(grade_path):
                    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        src_file = os.path.join(grade_path, filename)
                        dest_file = os.path.join(dest_folder, filename)
                        
                        if os.path.exists(dest_file):
                            print(f"🔁 Skipped duplicate: {filename}")
                        else:
                            shutil.copy(src_file, dest_file)
                            print(f"✅ Copied {filename} → {target_class}")

def main():
    # Create all necessary target folders
    create_target_folders()
    
    # Check dataset selection from environment variable
    dataset_selection = os.environ.get("EMBRYO_DATASET_SELECTION", "").lower()
    
    if dataset_selection == "roboflow" or dataset_selection == "":
        # Process only roboflow dataset
        process_roboflow_dataset()
    elif dataset_selection == "other":
        # Process only other dataset
        process_other_dataset()
    elif dataset_selection == "both":
        # Process both datasets
        process_roboflow_dataset()
        process_other_dataset()
    else:
        # Default: try to process both if no selection is specified
        print("\n" + "=" * 50)
        print("No specific dataset selected, attempting to process available datasets...")
        print("=" * 50)
        
        if os.path.exists(ROBOFLOW_SOURCE) and len(os.listdir(ROBOFLOW_SOURCE)) > 0:
            process_roboflow_dataset()
        
        if os.path.exists(OTHER_SOURCE) and len(os.listdir(OTHER_SOURCE)) > 0:
            process_other_dataset()
    
    # Print summary
    print("\n" + "=" * 50)
    print("Data labeling complete!")
    print("=" * 50)
    
    # Count files in each target folder
    total_files = 0
    print("\nFiles per category:")
    for folder in os.listdir(DEST_BASE):
        folder_path = os.path.join(DEST_BASE, folder)
        if os.path.isdir(folder_path):
            file_count = len([f for f in os.listdir(folder_path) 
                             if os.path.isfile(os.path.join(folder_path, f))])
            total_files += file_count
            print(f"  - {folder}: {file_count} files")
    
    print(f"\nTotal files processed: {total_files}")

if __name__ == "__main__":
    main()
