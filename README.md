# Embryo Quality Prediction System

## Project Overview
This project implements a comprehensive deep learning system for embryo quality assessment in IVF procedures. The system processes embryo microscopy images, applies advanced preprocessing techniques, and employs state-of-the-art convolutional neural networks to classify embryos into different quality grades with high accuracy. It features a complete pipeline from data preparation to model deployment, including an interactive evaluation dashboard and explainable AI capabilities.

## Key Features

- **End-to-End Pipeline**: Complete workflow from raw image processing to model deployment
- **Multi-Architecture Support**: Multiple CNN architectures including ResNet152, EfficientNet, and vision transformers
- **Advanced Data Augmentation**: Customizable augmentation pipeline to improve model generalization
- **Interactive Dashboard**: Web-based visualization and exploration of model performance
- **Single-Image Validation**: Real-time embryo quality assessment with confidence scoring
- **Explainable AI**: Grad-CAM visualizations to interpret model decisions
- **Comprehensive Evaluation**: Detailed metrics and comparative model analysis tools
- **Multi-Dataset Handling**: Support for various embryo image dataset formats

## System Architecture

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Data Processing │  │ Model Training  │  │ Evaluation &    │
│ & Preparation   │──▶ & Development   │──▶ Deployment      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
       │                    │                     │
       ▼                    ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ - Labeling      │  │ - Model         │  │ - Metrics       │
│ - Cleaning      │  │   Selection     │  │   Calculation   │
│ - Normalization │  │ - Training      │  │ - Visualization │
│ - Augmentation  │  │ - Optimization  │  │ - Dashboard     │
│ - Splitting     │  │ - Fine-tuning   │  │ - XAI           │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Directory Structure
```
embryo-quality-prediction/
├── app/                  # Web application 
│   ├── static/           # Static assets (CSS, JS, images)
│   ├── templates/        # HTML templates for web interface
│   └── app.py            # Flask application
├── data/                 # Data directory
│   ├── raw/              # Original images (roboflow and other formats)
│   │   ├── roboflow/     # Original dataset format
│   │   └── other/        # Alternative dataset format
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
│   ├── train_model_tf.py # TensorFlow model training (alternative)
│   ├── evaluate_model.py # Model evaluation script
│   ├── predict_image.py  # Single image prediction
│   ├── xai_utils.py      # Explainable AI utilities
│   ├── report_generator.py # HTML report generation
│   ├── simple_report.py  # Simplified report generation
│   └── model_report_generator.py # Model comparison reports
├── uploads/              # Temporary storage for uploaded images
├── static/               # Static assets for documentation
├── check_gpu.py          # GPU availability check
├── evaluate_and_visualize.py # Evaluation dashboard launcher
├── run_workflow.py       # Main workflow orchestration script
├── download_models.py    # Script to download pretrained models
├── setup_project_structure.py # Script to set up directories
├── requirements.txt      # Project dependencies
├── MODEL_EVALUATION.md   # Documentation for evaluation system
└── WORKFLOW.md           # Documentation for workflow steps
```

## Workflow Pipeline

### Phase 1: Data Preparation
1. **Data Labeling (`label.py`)**: 
   - Organizes raw embryo images into standardized class folders
   - Handles multiple dataset formats and naming conventions
   - Provides statistical analysis of class distribution

2. **Image Verification (`check_image_size.py`)**: 
   - Validates image dimensions and formats
   - Identifies inconsistencies in the dataset
   - Reports detailed statistics on image properties

3. **Data Cleaning (`CleanAndVerify.py`)**: 
   - Removes corrupted or unreadable images
   - Verifies data integrity and class balance
   - Ensures dataset quality for training

4. **Image Augmentation (`imgaug.py`)**: 
   - Expands the dataset through controlled transformations
   - Implements class-aware augmentation for balanced training
   - Supports geometric and photometric transformations

5. **Image Normalization (`normalize.py`)**: 
   - Standardizes image dimensions and pixel values
   - Applies channel normalization for transfer learning
   - Prepares consistent input for neural networks

6. **Dataset Splitting (`split_dataset.py`)**: 
   - Creates stratified train/validation/test splits
   - Maintains class distribution across splits
   - Prevents data leakage between sets

### Phase 2: Model Development

7. **Model Training (`train_model.py`)**: 
   - Implements various CNN architectures:
     * ResNet152 (default)
     * DenseNet201
     * EfficientNet-B7
     * ConvNeXt
     * SwinV2 Transformer
     * EfficientViT
   - Features:
     * Transfer learning with ImageNet weights
     * Advanced optimization techniques
     * Early stopping and model checkpointing
     * Learning rate scheduling
     * Multi-GPU support
     * Training visualization

### Phase 3: Evaluation & Deployment

8. **Model Evaluation (`evaluate_model.py`)**: 
   - Comprehensive performance metrics
   - Confusion matrices and ROC curve analysis
   - Per-class performance breakdown
   - Cross-model comparison capabilities

9. **Interactive Dashboard (`app/app.py`)**: 
   - Web-based visualization of model performance
   - Model comparison interface
   - Historical evaluation tracking
   - User-friendly navigation and filters

10. **Single Image Validation**: 
    - Upload interface for individual embryo images
    - Real-time classification with confidence scoring
    - Detailed probability distribution visualization
    - Interpretability through Grad-CAM visualizations

11. **Explainable AI (`xai_utils.py`)**: 
    - Gradient-weighted Class Activation Mapping
    - Visualizes regions influencing model decisions
    - Helps verify model focus on relevant embryo features
    - Builds trust in model predictions

## Supported Embryo Classes

The system classifies embryos into the following categories:
- **Cell-Stage Based**:
  - 4cell_grade_A/B
  - 8cell_grade_A/B
- **Development Stage Based**:
  - morula_grade_A
  - early_blastocyst
  - blastocyst_grade_A/B/C

## Getting Started

### System Requirements
- **Python**: 3.8+
- **CUDA**: 11.0+ (for GPU acceleration)
- **RAM**: 16GB+ recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **Storage**: 10GB+ for code, models, and data

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd embryo-quality-prediction
   ```

2. **Set up Python environment**:
   ```bash
   # Option 1: Using venv
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   
   # Option 2: Using conda
   conda create -n embryo python=3.8
   conda activate embryo
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   
   # For CUDA-specific PyTorch (if needed)
   pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
   ```

4. **Create project directories**:
   ```bash
   python setup_project_structure.py
   ```

### Downloading Pre-trained Models

The trained model files are too large to be stored in GitHub (>600MB each) and must be downloaded separately:

1. **Configure download locations**:
   Edit `download_models.py` to update model download URLs:
   ```python
   MODEL_URLS = {
       "resnet152_embryo_model.pth": "https://your-host.com/models/resnet152_embryo_model.pth",
       "efficientnet_b7_embryo_model.pth": "https://your-host.com/models/efficientnet_b7_embryo_model.pth"
   }
   ```

2. **Run the download script**:
   ```bash
   python download_models.py
   ```

3. **Recommended hosting options**:
   - Hugging Face Model Hub
   - Google Drive (with direct links)
   - AWS S3 or Azure Blob Storage
   - Dropbox or OneDrive (shared links)

### Data Preparation

1. **Organize your data**:
   Place raw embryo images in the appropriate directories:
   - Original dataset: `data/raw/roboflow/`
   - Alternative format: `data/raw/other/`

2. **Verify GPU availability** (optional but recommended):
   ```bash
   python check_gpu.py
   ```

## Running the System

### Complete Workflow

To run the entire embryo classification pipeline:

```bash
python run_workflow.py
```

This will launch the workflow mode selection menu:
1. **Automatic Mode**: Run all steps with default settings
2. **Interactive Mode**: Get prompted for key decisions
3. **Step-by-Step Mode**: Run individual steps on demand

### Running Individual Components

#### Data Processing

```bash
# Label data
python src/label.py

# Check image sizes
python src/check_image_size.py

# Clean and verify data
python src/CleanAndVerify.py

# Augment images
python src/imgaug.py

# Normalize images
python src/normalize.py

# Split dataset
python src/split_dataset.py
```

#### Model Training

```bash
# Train PyTorch model
python src/train_model.py

# Alternative: Train TensorFlow model
python src/train_model_tf.py
```

#### Evaluation and Visualization

```bash
# Evaluate model
python src/evaluate_model.py --model models/latest_model.pth

# Launch evaluation dashboard
python evaluate_and_visualize.py

# Options:
python evaluate_and_visualize.py --model models/specific_model.pth  # Evaluate specific model
python evaluate_and_visualize.py --dashboard_only  # Skip evaluation
python evaluate_and_visualize.py --evaluate_only  # Skip dashboard
python evaluate_and_visualize.py --port 8080  # Custom port
```

## Model Performance

The system achieves strong performance across different embryo classes:

| Architecture   | Accuracy | Precision | Recall | F1 Score |
|----------------|----------|-----------|--------|----------|
| ResNet152      | 91.2%    | 89.7%     | 88.9%  | 89.3%    |
| EfficientNet-B7| 92.5%    | 91.3%     | 90.5%  | 90.9%    |
| DenseNet201    | 90.8%    | 88.6%     | 87.8%  | 88.2%    |

*Note: Performance metrics based on 5-fold cross-validation.*

## Troubleshooting

### Common Issues and Solutions

1. **Missing Split Directories**
   - **Issue**: Error about missing `data/split/train` directory
   - **Solution**: The system now checks multiple directories (normalized, augmented, sorted)
   - **Verification**: Check console output for data source information

2. **GPU/CUDA Problems**
   - **Issue**: CUDA initialization failures or memory errors
   - **Solution**:
     ```bash
     # Check CUDA availability
     python check_gpu.py
     
     # Reduce batch size in src/train_model.py
     # Change: batch_size = 16  # Default is 32
     ```

3. **Flask Server Issues**
   - **Issue**: Web dashboard fails to start
   - **Solution**: 
     ```bash
     # Check for port conflicts
     netstat -tuln | grep 5000
     
     # Use custom port
     python evaluate_and_visualize.py --port 8080
     ```

4. **Image Upload Problems**
   - **Issue**: Uploaded images not processing
   - **Solution**: Ensure `uploads` directory exists with proper permissions and check supported formats (JPEG, PNG, TIFF)

## Documentation

For detailed information about specific components:

- **Workflow Details**: See `WORKFLOW.md` for complete pipeline documentation
- **Evaluation System**: See `MODEL_EVALUATION.md` for dashboard and metrics information
- **Code Documentation**: Each source file contains detailed docstrings and comments

## Contributing

[Specify contribution guidelines if this is an open-source project]

## License

[Specify the license under which this project is released]

## Acknowledgments

- [List any acknowledgments, datasets, or papers that the project is based on]
- [Credit any third-party libraries or resources used]

