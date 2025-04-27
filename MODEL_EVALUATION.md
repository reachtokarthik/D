# Model Evaluation Dashboard for Embryo Quality Prediction

This document provides instructions on how to use the comprehensive model evaluation system with an interactive HTML dashboard and single-image validation tool.

## Overview

The model evaluation system consists of:

1. A Python script (`evaluate_model.py`) for evaluating trained models
2. A Flask web application for visualizing evaluation results
3. A command-line utility (`evaluate_and_visualize.py`) to run both components

## Features

- **Comprehensive Metrics**: Accuracy, precision, recall, F1 score, per-class metrics, and confusion matrices
- **Interactive Dashboard**: View evaluation results in a user-friendly web interface
- **Model Comparison**: Compare multiple models side-by-side with radar charts and metrics tables
- **Visualizations**: Charts and graphs for easy interpretation of results
- **Explainable AI (XAI)**: Grad-CAM visualizations to understand model decisions
- **Exportable Reports**: HTML reports that can be saved and shared
- **Single-Image Validation**: Upload and validate individual embryo images with detailed results
- **User Experience Enhancements**: Loading spinners, responsive design, and intuitive navigation

## Installation

All required packages are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Using the Command-Line Utility

The easiest way to evaluate models and launch the dashboard is using the provided utility script:

```bash
python evaluate_and_visualize.py
```

This will:
1. Find the latest trained model
2. Evaluate it on the test dataset
3. Generate an HTML report
4. Launch the web dashboard

### Option 2: Evaluating a Specific Model

To evaluate a specific model:

```bash
python evaluate_and_visualize.py --model path/to/your/model.pth
```

### Option 3: Dashboard Only

To launch the dashboard without running evaluation:

```bash
python evaluate_and_visualize.py --dashboard_only
```

### Option 4: Evaluation Only

To run evaluation without launching the dashboard:

```bash
python evaluate_and_visualize.py --evaluate_only
```

## Dashboard Features

### Home Page
- Quick actions for model evaluation and image validation
- Overview of available models
- Recent evaluation reports
- Performance metrics visualization
- Intuitive navigation with loading indicators

### Model Comparison
- Side-by-side comparison of multiple models
- Comparative charts for all metrics
- Per-class performance comparison with radar charts
- Detailed metrics tables for quantitative analysis
- Recommendations based on model performance

### Detailed Reports
- Comprehensive metrics for each model
- Confusion matrices with color-coded visualization
- ROC curves for classification performance
- Per-class performance breakdown
- Exportable HTML format for sharing

### Image Validation Tool
- Upload embryo images via drag-and-drop or file selection
- Immediate display of image details (filename, size, dimensions)
- Model selection for prediction
- Real-time validation with loading indicators
- Detailed results showing:
  - Predicted class with confidence level
  - Class probability distribution (chart and table)
  - Interpretation based on confidence level
  - Technical details of the prediction
  - Grad-CAM heatmap visualization highlighting regions that influenced the prediction

## Programmatic Usage

You can also use the evaluation module in your own scripts:

```python
from src.evaluate_model import ModelEvaluator

# Initialize evaluator with a model path
evaluator = ModelEvaluator('path/to/model.pth')

# Run evaluation
results = evaluator.evaluate()

# Save results and generate HTML report
evaluator.save_results()
html_path = evaluator.generate_html_report()

print(f"Report generated at: {html_path}")
```

## Interpreting Results

### Key Metrics

- **Accuracy**: Overall percentage of correctly classified embryos
- **Precision**: Ratio of true positives to all positive predictions (measures false positive rate)
- **Recall**: Ratio of true positives to all actual positives (measures false negative rate)
- **F1 Score**: Harmonic mean of precision and recall (balanced measure)
- **AUC**: Area Under the ROC Curve (discrimination ability)

### Grad-CAM Visualizations

Gradient-weighted Class Activation Mapping (Grad-CAM) provides visual explanations of model decisions by highlighting regions in the image that most influenced the prediction:

- **Red/Yellow Areas**: Regions that strongly influenced the model's decision for the predicted class
- **Blue/Green Areas**: Regions with less influence on the prediction
- **Interpretation**: Focus on where the model is looking to understand if it's using relevant embryo features
- **Clinical Value**: Helps verify if the model is focusing on biologically relevant structures

### Confusion Matrix

The confusion matrix shows:
- True positives (diagonal)
- False positives (columns)
- False negatives (rows)

This helps identify which classes are being confused with each other.

### ROC Curves

ROC curves plot the true positive rate against the false positive rate at various threshold settings. The area under the curve (AUC) provides a measure of the model's ability to discriminate between classes.

### Image Validation Results

When validating individual images, the results include:

- **Predicted Class**: The most likely embryo quality class
- **Confidence Score**: How confident the model is in its prediction (0-100%)
- **Class Probabilities**: Distribution of probabilities across all classes
- **Interpretation**:
  - **High Confidence** (≥90%): Prediction can be considered reliable
  - **Medium Confidence** (70-89%): Consider reviewing the image or getting a second opinion
  - **Low Confidence** (<70%): Prediction is uncertain and should be verified by an expert

## Best Practices

1. **Regular Evaluation**: Re-evaluate models after significant dataset changes
2. **Compare Multiple Models**: Try different architectures and hyperparameters
3. **Focus on F1 Score**: For imbalanced datasets, F1 score is often more informative than accuracy
4. **Per-Class Analysis**: Look at performance for individual classes, not just overall metrics
5. **Error Analysis**: Examine misclassified examples to understand model weaknesses
6. **Image Quality**: For validation, use clear, well-focused images for best results
7. **Confidence Thresholds**: Pay attention to confidence scores and be cautious with low-confidence predictions
8. **Expert Verification**: Always have critical predictions verified by domain experts

## Troubleshooting

### Common Issues

- **Missing Model File**: Ensure model files are in the `models` directory
- **Flask Port Conflict**: If port 5000 is in use, specify a different port with `--port`
- **Memory Errors**: For large models, reduce batch size in the Config class
- **Image Upload Issues**: Ensure the `uploads` directory exists and has proper permissions
- **Loading Spinners Not Appearing**: Check browser console for JavaScript errors
- **Image Preview Not Showing**: Verify that the image format is supported (JPEG, PNG, TIFF)

### Getting Help

If you encounter issues, check:
1. The console output for error messages
2. The Flask server logs for web application errors
3. Browser developer tools (F12) for frontend issues

## Recent Improvements

### User Interface Enhancements
- **Loading Spinners**: Added for all long-running operations
- **Image Preview Details**: Filename, size, and dimensions shown immediately after selection
- **Responsive Design**: Improved mobile and tablet compatibility
- **Error Handling**: Better error messages and validation

### Functionality Improvements
- **Explainable AI Integration**: Added Grad-CAM visualizations for model interpretability
- **Image Validation Tool**: New feature for validating individual embryo images
- **Model Comparison**: Enhanced visualization with radar charts
- **Dashboard Navigation**: Streamlined user experience
- **Performance Optimization**: Faster page loading and response times

### XAI Implementation
- **Grad-CAM Algorithm**: Implemented for CNN model visualization
- **Heatmap Generation**: Real-time generation of attention heatmaps
- **Batch Processing**: Support for processing multiple images
- **Integration with Validation**: Seamless integration with the image validation workflow
- **Combined Visualizations**: Side-by-side display of original images and heatmaps
