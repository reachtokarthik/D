# Embryo Quality Prediction Project

## Project Overview
This project implements a deep learning-based classification system for embryo quality assessment. It processes embryo images, applies various preprocessing techniques, and trains a convolutional neural network to classify embryos into different quality grades. The system includes a comprehensive evaluation dashboard and single-image validation tool for real-time embryo quality assessment.

## Project Architecture

### Directory Structure
```
embryo-quality-prediction/
├── app/                  # Web application for deployment
│   ├── static/           # Static assets (CSS, JS, images)
│   ├── templates/        # HTML templates for the web interface
│   └── app.py            # Flask application
├── data/                 # Data directory
│   ├── raw/              # Original raw images
│   ├── sorted/           # Images sorted by class
│   ├── normalized/       # Normalized images
│   ├── augmented/        # Augmented images
│   └── split/            # Train/validation/test splits
├── models/               # Saved model checkpoints
├── notebooks/            # Jupyter notebooks for exploration
├── outputs/              # Output files
│   ├── plots/            # Generated plots and visualizations
│   ├── results/          # Evaluation results
│   └── reports/          # HTML evaluation reports
├── src/                  # Source code
│   ├── label.py          # Data labeling script
│   ├── check_image_size.py # Image size verification
│   ├── CleanAndVerify.py # Data cleaning and verification
│   ├── normalize.py      # Image normalization
│   ├── imgaug.py         # Image augmentation
│   ├── split_dataset.py  # Dataset splitting
│   ├── train_model.py    # PyTorch model training
│   ├── evaluate_model.py # Model evaluation script
│   └── predict_image.py  # Single image prediction
├── uploads/              # Temporary storage for uploaded images
├── check_gpu.py          # GPU availability check
├── evaluate_and_visualize.py # Evaluation dashboard launcher
├── MODEL_EVALUATION.md   # Documentation for evaluation system
└── run_workflow.py       # Main workflow script
```

## Workflow Steps

The project implements a complete embryo classification workflow in the following sequence:

1. **Data Labeling**: Organizes and labels the raw embryo images
2. **Image Size Checking**: Verifies and reports image dimensions
3. **Data Cleaning and Verification**: Removes corrupted images and ensures data integrity
4. **Image Normalization**: Standardizes images for consistent processing
5. **Image Augmentation**: Expands the dataset through various transformations
6. **Dataset Splitting**: Divides data into training, validation, and test sets
7. **Model Training**: Trains a deep learning model on the processed data
8. **Model Evaluation**: Comprehensive evaluation with metrics and visualizations
9. **Interactive Dashboard**: Web interface for exploring model performance
10. **Single Image Validation**: Tool for validating individual embryo images

## Embryo Classification Classes

The project classifies embryos into the following categories:
- 8cell_grade_A
- blastocyst_grade_A
- blastocyst_grade_B
- blastocyst_grade_C
- morula_grade_A

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- TensorFlow (optional)
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd embryo-quality-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Downloading Model Files

The trained model files are too large to be stored in GitHub (>600MB each). After cloning the repository, you need to download these files separately:

1. Run the provided download script:
   ```bash
   python download_models.py
   ```

2. Before running the script for the first time, you need to update it with the actual URLs where you've hosted the model files. Edit `download_models.py` and replace the placeholder URLs with actual download links.

3. Recommended hosting options for large model files:
   - Google Drive
   - Dropbox
   - Hugging Face Model Hub
   - AWS S3
   - Azure Blob Storage

**Note:** The model files are essential for running the evaluation dashboard and single image validation. Make sure to download them before using these features.

### Data Preparation

1. Place your raw embryo images in the `data/raw` directory
2. Organize them into appropriate class folders if they're not already labeled

### Running the Workflow

To run the complete workflow:

```bash
python run_workflow.py
```

This will execute all steps in sequence from data labeling to model training.

### Running Individual Steps

You can also run individual steps of the workflow:

```bash
python src/label.py                # Run only the labeling step
python src/normalize.py            # Run only the normalization step
python src/imgaug.py               # Run only the augmentation step
python src/split_dataset.py        # Run only the dataset splitting step
python src/train_model.py          # Run only the model training step
python src/evaluate_model.py       # Run only the model evaluation step
python evaluate_and_visualize.py   # Launch the evaluation dashboard
```

## Model Architecture

The project uses a ResNet152 architecture pre-trained on ImageNet and fine-tuned on embryo images. The model configuration can be modified in `src/train_model.py`.

## Troubleshooting

### Common Issues

1. **Missing Split Directories**: If you encounter an error about missing split directories, ensure that the `split_dataset.py` script has been run successfully and that the `data/augmented` directory contains images.

2. **GPU Not Available**: The training will automatically use CPU if no GPU is available, but this will be significantly slower. Check GPU availability with `python check_gpu.py`.

3. **Out of Memory Errors**: If you encounter CUDA out of memory errors, try reducing the batch size in `src/train_model.py`.

## Results and Evaluation

### Model Evaluation Dashboard

The project includes a comprehensive evaluation dashboard with the following features:

- Interactive web interface for exploring model performance
- Detailed metrics (accuracy, precision, recall, F1 score, ROC curves)
- Side-by-side model comparison
- Per-class performance analysis
- Exportable HTML reports

To launch the dashboard:

```bash
python evaluate_and_visualize.py
```

See `MODEL_EVALUATION.md` for detailed instructions on using the evaluation system.

### Single Image Validation

The system includes a tool for validating individual embryo images:

- Upload any embryo image through the web interface
- Select a trained model for prediction
- Get immediate results with class prediction and confidence scores
- View detailed class probabilities with visualizations

The validation tool is accessible through the main dashboard or directly at `/validate` when the Flask app is running.

### Evaluation Outputs

After training and evaluation, the following outputs are generated:

- Model evaluation results and metrics in `outputs/results`
- Training and evaluation plots in `outputs/plots`
- Interactive HTML reports in `outputs/reports`

## License

[Specify the license under which this project is released]

## Acknowledgments

- [List any acknowledgments, datasets, or papers that the project is based on]

