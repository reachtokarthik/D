import os
import shutil

# Base input: folder with folders like 2-1-1, Morula, Early, etc.
import os.path as path

# Get the absolute path to the project root directory
SCRIPT_DIR = path.dirname(path.abspath(__file__))
PROJECT_ROOT = path.dirname(SCRIPT_DIR)

# Use absolute paths based on the project root
SOURCE_BASE = path.join(PROJECT_ROOT, "data", "raw", "roboflow")
DEST_BASE = path.join(PROJECT_ROOT, "data", "sorted")

# Folder mapping from original labels to your unified class folders
label_map = {
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
    "Arrested": None  # Optionally handle or skip
}

# Create target folders if needed
for folder in set(label_map.values()):
    if folder:
        os.makedirs(os.path.join(DEST_BASE, folder), exist_ok=True)

# Walk through folders and move images
for label_folder in os.listdir(SOURCE_BASE):
    source_folder_path = os.path.join(SOURCE_BASE, label_folder)

    if not os.path.isdir(source_folder_path):
        continue

    mapped_label = label_map.get(label_folder, None)
    if not mapped_label:
        print(f"⚠️  Skipping folder '{label_folder}' - no mapped class")
        continue

    dest_folder_path = os.path.join(DEST_BASE, mapped_label)

    for filename in os.listdir(source_folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            src_file = os.path.join(source_folder_path, filename)
            dest_file = os.path.join(dest_folder_path, filename)

            if os.path.exists(dest_file):
                print(f"🔁 Skipped duplicate: {filename}")
            else:
                shutil.copy(src_file, dest_file)
                print(f"✅ Moved {filename} → {mapped_label}")
