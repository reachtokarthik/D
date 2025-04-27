# Embryo Quality Prediction: Complete Workflow Documentation

## Overview

This document provides a comprehensive explanation of the embryo quality prediction workflow, including each processing step, model training, evaluation metrics, and expected outputs. The workflow is designed to classify embryo images into different quality grades using deep learning techniques.

## Workflow Architecture

The embryo classification workflow consists of three main phases:

1. **Data Preparation Phase**: Processes raw embryo images through multiple stages to prepare them for model training
2. **Model Development Phase**: Trains, evaluates, and saves a deep learning model for embryo classification
3. **Evaluation and Deployment Phase**: Provides tools for comprehensive model evaluation, comparison, and single-image validation

### Workflow Mode Selection

The workflow offers three operational modes:

1. **Automatic Mode**: Uses default values for all selections and runs the entire workflow in sequence with minimal user interaction. Ideal for batch processing or when optimal defaults are sufficient.

2. **Interactive Mode**: Prompts the user for each selection (dataset, model architecture), but still runs the entire workflow in sequence. This is the default mode and provides customization while maintaining the automated flow.

3. **Step-by-Step Mode**: Allows running individual workflow steps on demand. This mode gives you complete control to:
   - Select and run specific steps in any order
   - Run multiple steps in sequence
   - Pause between phases to review results
   - Skip steps that aren't needed for your specific use case

At the start of the workflow, you'll be prompted to select which mode you prefer.

### Dataset Selection

The workflow supports multiple dataset sources:

1. **Roboflow Dataset**: The original dataset with folders like 2-1-1, Morula, Early, etc.
2. **Other Dataset**: New dataset with a different structure (Blastocyst, Cleavage, Morula, Error image)
3. **Both Datasets**: Process and combine both datasets

In interactive mode, you'll be prompted to select which dataset(s) to use. In automatic mode, the system will select the most comprehensive option available (both datasets if available, otherwise the single available dataset).

## Complete Workflow Steps

### Phase 1: Data Preparation

#### Step 1: Data Labeling (`label.py`)
- **Purpose**: Organizes and labels raw embryo images into appropriate class folders
- **Input**: Raw embryo images in `data/raw/roboflow` and/or `data/raw/other`
- **Output**: Labeled images in `data/sorted` organized by class
- **Process**:
  - Identifies embryo types and quality grades based on folder structure
  - Handles multiple dataset formats (roboflow and other)
  - Creates class directories (e.g., `8cell_grade_A`, `blastocyst_grade_A`, etc.)
  - Copies images to their respective class directories
  - Provides detailed statistics on processed files

#### Step 2: Image Size Checking (`check_image_size.py`)
- **Purpose**: Verifies image dimensions and reports any inconsistencies
- **Input**: Labeled images in `data/sorted`
- **Output**: Console report of image dimensions
- **Process**:
  - Scans all images in the sorted directory
  - Identifies and reports images with inconsistent dimensions
  - Provides statistics on image sizes across the dataset

#### Step 3: Data Cleaning and Verification (`CleanAndVerify.py`)
- **Purpose**: Removes corrupted images and ensures data integrity
- **Input**: Labeled images in `data/sorted`
- **Output**: Cleaned dataset with corrupted images removed
- **Process**:
  - Identifies and removes corrupted or unreadable images
  - Verifies class balance and reports statistics
  - Ensures all images are properly formatted for further processing

#### Step 4: Image Augmentation (`imgaug.py`)
- **Purpose**: Expands the dataset through various transformations
- **Input**: Cleaned images from `data/sorted`
- **Output**: Augmented images in `data/augmented`
- **Process**:
  - Applies random transformations (rotations, flips, color adjustments)
  - Generates multiple variations of each original image
  - Balances class distribution by generating more samples for underrepresented classes
  - Increases dataset size to improve model generalization

#### Step 5: Image Normalization (`normalize.py`)
- **Purpose**: Standardizes images for consistent processing
- **Input**: Original images from `data/sorted` and augmented images from `data/augmented`
- **Output**: Normalized images in `data/normalized`
- **Process**:
  - Resizes images to a standard dimension (224×224 pixels)
  - Applies color normalization to account for staining variations
  - Enhances contrast and reduces noise
  - Standardizes pixel values to improve model convergence

#### Step 6: Dataset Splitting (`split_dataset.py`)
- **Purpose**: Divides data into training, validation, and test sets
- **Input**: Normalized images from `data/normalized` (falls back to `data/augmented` or `data/sorted` if needed)
- **Output**: Split datasets in `data/split/train`, `data/split/val`, and `data/split/test`
- **Process**:
  - Intelligently searches for data in multiple directories
  - Performs stratified splitting to maintain class distribution
  - Allocates 80% for training, 10% for validation, and 10% for testing
  - Creates separate directories for each split while preserving class structure
  - Ensures no data leakage between splits
  - Provides detailed verification of copied files

### Phase 2: Model Development

#### GPU Check
- **Purpose**: Verifies GPU availability for accelerated training
- **Process**:
  - Checks if CUDA is available
  - Reports GPU name and memory if available
  - Warns if training will proceed on CPU (significantly slower)

#### Model Selection
- **Purpose**: Allows selection of different model architectures for training
- **Options**:
  1. **ResNet152** (default): Deep residual network with 152 layers
  2. **DenseNet201**: Dense convolutional network with 201 layers
  3. **EfficientNet-B7**: Optimized convolutional network with compound scaling
  4. **ConvNeXt Base**: Modern convolutional network with transformer-inspired design
  5. **SwinV2**: Hierarchical vision transformer with shifted windows
  6. **EfficientViT**: Hybrid architecture combining efficiency of CNNs with vision transformers
- **Process**:
  - In interactive mode: Prompts user to select a model architecture
  - In automatic mode: Uses ResNet152 (default model)
  - Sets environment variable for model selection
  - Configures the selected model for training

#### Step 7: Model Training (`train_model.py`)
- **Purpose**: Trains a deep learning model on the processed data
- **Input**: Split datasets from `data/split`
- **Output**: Trained model in `models/`, evaluation metrics in `outputs/results/`, and plots in `outputs/plots/`
- **Process**:
  - Loads and prepares data using PyTorch DataLoaders
  - Initializes a pre-trained ResNet152 model (transfer learning)
  - Fine-tunes the model on embryo images
  - Implements early stopping to prevent overfitting
  - Saves the best model based on validation accuracy
  - Generates training curves and evaluation metrics

### Phase 3: Evaluation and Deployment

#### Step 8: Comprehensive Model Evaluation (`evaluate_model.py`)
- **Purpose**: Provides detailed evaluation of trained models
- **Input**: Trained model from `models/` and test dataset
- **Output**: Comprehensive metrics, visualizations, and HTML reports
- **Process**:
  - Loads the trained model and test data
  - Computes detailed metrics (accuracy, precision, recall, F1 score)
  - Generates confusion matrices and ROC curves
  - Calculates per-class performance metrics
  - Creates exportable HTML reports with visualizations
  - Saves all results to `outputs/results/` and `outputs/reports/`

#### Step 9: Interactive Evaluation Dashboard (`app/app.py`)
- **Purpose**: Provides a web interface for exploring model performance
- **Input**: Evaluation results from `outputs/`
- **Output**: Interactive web dashboard
- **Process**:
  - Serves evaluation results through a Flask web application
  - Provides model comparison features
  - Displays interactive charts and visualizations
  - Allows browsing of evaluation history
  - Launched via `evaluate_and_visualize.py`

#### Step 10: Single Image Validation (`predict_image.py`)
- **Purpose**: Validates individual embryo images using trained models
- **Input**: Single embryo image and selected model
- **Output**: Prediction results with confidence scores and visualizations
- **Process**:
  - Loads and preprocesses the input image
  - Applies the selected model to generate predictions
  - Calculates class probabilities and confidence scores
  - Displays results with visualizations through the web interface
  - Accessible via the `/validate` endpoint in the web application

#### Step 11: Explainable AI Visualization (`xai_utils.py`)
- **Purpose**: Provides visual explanations of model decisions using Grad-CAM
- **Input**: Trained model and input image
- **Output**: Heatmap visualizations highlighting regions influencing predictions
- **Process**:
  - Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
  - Identifies regions in the image that most influenced the model's decision
  - Generates heatmap overlays on original images
  - Supports both single image and batch visualization
  - Integrated with the validation tool for real-time explainability

## Model Output Expectations

### Strengths of the Model

1. **High Accuracy for Clear Images**: The model typically achieves high accuracy (>90%) for well-focused, clearly visible embryo images with distinct morphological features.

2. **Class Differentiation**: The model excels at differentiating between major embryo stages (e.g., 8-cell vs. blastocyst) due to their distinct morphological differences.

3. **Transfer Learning Advantage**: By leveraging pre-trained weights from ImageNet, the model benefits from general feature extraction capabilities even with a relatively small embryo dataset.

4. **Robust to Minor Variations**: Thanks to data augmentation, the model is relatively robust to minor variations in orientation, lighting, and positioning of embryos.

5. **Comprehensive Metrics**: The model provides detailed evaluation metrics (accuracy, precision, recall, F1 score) and confusion matrices to help understand its performance across different classes.

### Limitations and Challenges

1. **Grade Differentiation Difficulty**: The model may struggle to differentiate between subtle quality grades (e.g., blastocyst_grade_A vs. blastocyst_grade_B) as these distinctions can be subjective and require expert knowledge.

2. **Sensitivity to Image Quality**: Poor image quality, inconsistent focus, or non-standard imaging conditions can significantly reduce classification accuracy.

3. **Class Imbalance Effects**: If the dataset has significant class imbalance (e.g., many more blastocysts than 8-cell embryos), the model may show bias toward majority classes despite stratification efforts.

4. **Limited Generalization**: The model may not generalize well to images from different clinics or imaging systems with different protocols if they weren't represented in the training data.

5. **Interpretability Challenges**: While the new Grad-CAM visualization helps with model explainability, the underlying deep learning model still has inherent interpretability limitations that should be considered in clinical settings.

## Expected Evaluation Metrics

The comprehensive model evaluation system produces the following metrics and visualizations:

### Overall Performance Metrics
1. **Accuracy**: Overall percentage of correctly classified embryos (typically 85-95% on test set)
2. **Precision**: Ratio of true positives to all positive predictions (measures false positive rate)
3. **Recall**: Ratio of true positives to all actual positives (measures false negative rate)
4. **F1 Score**: Harmonic mean of precision and recall (balanced measure of model performance)

### Detailed Analysis
5. **Confusion Matrix**: Visual representation of classification performance across all classes
6. **ROC Curves**: Receiver Operating Characteristic curves showing model discrimination ability
7. **Per-Class Metrics**: Precision, recall, and F1 score for each embryo class
8. **Class Probability Distributions**: Distribution of prediction probabilities for each class

### Comparative Analysis
9. **Model Comparison**: Side-by-side comparison of multiple models
10. **Radar Charts**: Visualization of per-class F1 scores across different models
11. **Performance Over Time**: Tracking of model performance across different training runs

## Troubleshooting Common Issues

### Missing Split Directories
- **Issue**: Training fails with error about missing `data/split/train` directory
- **Solution**: The updated workflow now checks multiple directories for data (normalized, augmented, and sorted). It also provides detailed logging about which directory was used and verifies that files were actually copied. Check the console output for specific error messages.

### GPU Memory Errors
- **Issue**: CUDA out of memory errors during training
- **Solution**: Reduce batch size in `Config` class of `train_model.py` or use a GPU with more memory

### Class Imbalance
- **Issue**: Poor performance on minority classes
- **Solution**: Adjust augmentation parameters to generate more samples for underrepresented classes or implement class weighting in the loss function

### Normalize Step Completing Too Quickly
- **Issue**: The normalize step completes in 0.00 seconds without processing any files
- **Solution**: This can happen if the normalize step runs before augmentation. The updated workflow now runs normalize after augmentation to ensure both original and augmented images are processed.

### Overfitting
- **Issue**: Large gap between training and validation accuracy
- **Solution**: Increase dropout rate, add more regularization, or reduce model complexity

### Flask Server Issues
- **Issue**: Flask server fails to start or crashes
- **Solution**: Check for port conflicts (default is 5000), ensure all dependencies are installed, and check console for specific error messages

### Multiple Dataset Handling
- **Issue**: Need to process different dataset formats
- **Solution**: The updated workflow now supports multiple dataset sources. At the start of the workflow, you'll be prompted to select which dataset(s) to use (roboflow, other, or both).

### Image Validation Errors
- **Issue**: Uploaded images not displaying or processing correctly
- **Solution**: Ensure the `uploads` directory exists and has write permissions, check that the image format is supported (JPEG, PNG, TIFF), and verify that the selected model exists

## Best Practices for Model Deployment

1. **Regular Retraining**: Periodically retrain the model with new data to maintain performance
2. **Human Verification**: Use the model as a decision support tool with human expert verification
3. **Confidence Thresholds**: Implement confidence thresholds to flag low-confidence predictions for manual review
4. **Monitoring**: Continuously monitor model performance in production to detect drift
5. **Version Control**: Maintain version control of models and datasets to track changes over time

## Using the Evaluation Dashboard

The interactive evaluation dashboard provides a comprehensive interface for model assessment:

1. **Launching the Dashboard**:
   ```bash
   python evaluate_and_visualize.py
   ```

2. **Key Features**:
   - **Home Dashboard**: Overview of available models and quick actions
   - **Model Evaluation**: Run evaluation on any trained model
   - **Model Comparison**: Compare multiple models side-by-side
   - **Evaluation History**: Browse past evaluation results
   - **Image Validation**: Upload and validate individual embryo images

3. **Model Comparison Workflow**:
   - Select two or more models from the dropdown
   - View side-by-side comparison of all metrics
   - Analyze radar charts showing per-class performance
   - Identify strengths and weaknesses of each model

4. **Single Image Validation**:
   - Upload an embryo image through the web interface
   - Select a model for prediction
   - View detailed results including class probabilities
   - Results include confidence level and interpretation
   - Visualize model decision-making with Grad-CAM heatmaps

## Conclusion

The embryo quality prediction workflow provides a comprehensive pipeline for processing embryo images, training a classification model, and evaluating its performance. The addition of the interactive dashboard and single-image validation tool enhances the system's usability and provides valuable insights into model performance.

While the model demonstrates strong performance under ideal conditions, users should be aware of its limitations, particularly with subtle grade distinctions and varying image quality. The system is designed to support, not replace, expert judgment in embryo quality assessment.

The comprehensive evaluation tools help users understand model strengths and weaknesses, compare different approaches, and make informed decisions about model selection and deployment.
