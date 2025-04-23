# Embryo Quality Prediction: Complete Workflow Documentation

## Overview

This document provides a comprehensive explanation of the embryo quality prediction workflow, including each processing step, model training, evaluation metrics, and expected outputs. The workflow is designed to classify embryo images into different quality grades using deep learning techniques.

## Workflow Architecture

The embryo classification workflow consists of two main phases:

1. **Data Preparation Phase**: Processes raw embryo images through multiple stages to prepare them for model training
2. **Model Development Phase**: Trains, evaluates, and saves a deep learning model for embryo classification

## Complete Workflow Steps

### Phase 1: Data Preparation

#### Step 1: Data Labeling (`label.py`)
- **Purpose**: Organizes and labels raw embryo images into appropriate class folders
- **Input**: Raw embryo images in `data/raw`
- **Output**: Labeled images in `data/sorted` organized by class
- **Process**:
  - Identifies embryo types and quality grades from image metadata or manual annotation
  - Creates class directories (e.g., `8cell_grade_A`, `blastocyst_grade_A`, etc.)
  - Copies images to their respective class directories

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

#### Step 4: Image Normalization (`normalize.py`)
- **Purpose**: Standardizes images for consistent processing
- **Input**: Cleaned images from `data/sorted`
- **Output**: Normalized images in `data/normalized`
- **Process**:
  - Resizes images to a standard dimension (224×224 pixels)
  - Applies color normalization to account for staining variations
  - Enhances contrast and reduces noise
  - Standardizes pixel values to improve model convergence

#### Step 5: Image Augmentation (`imgaug.py`)
- **Purpose**: Expands the dataset through various transformations
- **Input**: Normalized images from `data/normalized`
- **Output**: Augmented images in `data/augmented`
- **Process**:
  - Applies random transformations (rotations, flips, color adjustments)
  - Generates multiple variations of each original image
  - Balances class distribution by generating more samples for underrepresented classes
  - Increases dataset size to improve model generalization

#### Step 6: Dataset Splitting (`split_dataset.py`)
- **Purpose**: Divides data into training, validation, and test sets
- **Input**: Augmented images from `data/augmented`
- **Output**: Split datasets in `data/split/train`, `data/split/val`, and `data/split/test`
- **Process**:
  - Performs stratified splitting to maintain class distribution
  - Allocates approximately 70% for training, 15% for validation, and 15% for testing
  - Creates separate directories for each split while preserving class structure
  - Ensures no data leakage between splits

### Phase 2: Model Development

#### GPU Check
- **Purpose**: Verifies GPU availability for accelerated training
- **Process**:
  - Checks if CUDA is available
  - Reports GPU name and memory if available
  - Warns if training will proceed on CPU (significantly slower)

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

5. **Black Box Nature**: As with most deep learning models, the decision-making process is not easily interpretable, which can be a limitation in clinical settings where explainability is important.

## Expected Evaluation Metrics

The model evaluation produces the following metrics:

1. **Accuracy**: Overall percentage of correctly classified embryos (typically 85-95% on test set)
2. **Precision**: Ratio of true positives to all positive predictions (measures false positive rate)
3. **Recall**: Ratio of true positives to all actual positives (measures false negative rate)
4. **F1 Score**: Harmonic mean of precision and recall (balanced measure of model performance)
5. **Confusion Matrix**: Visual representation of classification performance across all classes
6. **Per-Class Metrics**: Precision, recall, and F1 score for each embryo class

## Troubleshooting Common Issues

### Missing Split Directories
- **Issue**: Training fails with error about missing `data/split/train` directory
- **Solution**: Ensure the `split_dataset.py` script has run successfully and that the `data/augmented` directory contains images. The workflow will automatically attempt to recreate split directories if they're missing.

### GPU Memory Errors
- **Issue**: CUDA out of memory errors during training
- **Solution**: Reduce batch size in `Config` class of `train_model.py` or use a GPU with more memory

### Class Imbalance
- **Issue**: Poor performance on minority classes
- **Solution**: Adjust augmentation parameters to generate more samples for underrepresented classes or implement class weighting in the loss function

### Overfitting
- **Issue**: Large gap between training and validation accuracy
- **Solution**: Increase dropout rate, add more regularization, or reduce model complexity

## Best Practices for Model Deployment

1. **Regular Retraining**: Periodically retrain the model with new data to maintain performance
2. **Human Verification**: Use the model as a decision support tool with human expert verification
3. **Confidence Thresholds**: Implement confidence thresholds to flag low-confidence predictions for manual review
4. **Monitoring**: Continuously monitor model performance in production to detect drift
5. **Version Control**: Maintain version control of models and datasets to track changes over time

## Conclusion

The embryo quality prediction workflow provides a comprehensive pipeline for processing embryo images and training a classification model. While the model demonstrates strong performance under ideal conditions, users should be aware of its limitations, particularly with subtle grade distinctions and varying image quality. The system is designed to support, not replace, expert judgment in embryo quality assessment.
