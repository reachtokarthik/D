import os

def create_project_structure(base_path):
    # Define consistent class folders
    class_folders = [
        "8cell_grade_A", "8cell_grade_B", "8cell_grade_C",
        "morula_grade_A", "morula_grade_B", "morula_grade_C",
        "blastocyst_grade_A", "blastocyst_grade_B", "blastocyst_grade_C"
    ]

    # Project folders
    folders = [
        # Raw sources
        "data/raw/kaggle_embryo",
        "data/raw/embryonet",
        "data/raw/ivf_resnet",
        "data/raw/bbbc021",
        
        # Processed folders (model-sorted and verified)
        *[f"data/processed/{cf}" for cf in class_folders],

        # Augmented data
        *[f"data/augmented/{cf}" for cf in class_folders],

        # Sorted predictions (output of Roboflow inference script)
        *[f"data/sorted/{cf}" for cf in class_folders],

        # Project structure
        "notebooks",
        "models/best_model_checkpoint",
        "src",
        "outputs/evaluation_reports",
        "outputs/tensorboard_logs",
        "outputs/prediction_samples",
        "app/templates",
        "app/static"
    ]

    # Create all folders
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"✅ Created: {folder_path}")

    # Root-level starter files
    root_files = ["requirements.txt", "README.md", "Project_Charter.pdf"]
    for file in root_files:
        file_path = os.path.join(base_path, file)
        open(file_path, 'a').close()
        print(f"✅ Created: {file_path}")

# Run the function
if __name__ == "__main__":
    base_project_path = "embryo-quality-prediction"
    create_project_structure(base_project_path)
