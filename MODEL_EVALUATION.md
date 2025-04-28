# Embryo Quality Prediction Project

## Project Overview

This comprehensive system employs deep learning to classify embryo images based on quality, aiding in IVF success prediction. The project provides an end-to-end solution from data preprocessing to model evaluation, complete with an interactive visualization dashboard.

## System Architecture

The project consists of:

1. **Data Pipeline**: Processes raw embryo images through labeling, verification, normalization, and augmentation
2. **Training Module**: Trains various CNN architectures on preprocessed data
3. **Evaluation Engine**: Comprehensive metrics and visualizations for model performance assessment
4. **Web Dashboard**: Flask-based visualization and interaction system
5. **Explainable AI**: Grad-CAM visualizations to interpret model decisions
6. **Workflow Automation**: Integrated scripts to run the entire pipeline

## Key Workflows

### Complete Workflow

The `run_workflow.py` script orchestrates the entire process:

1. **Data Preparation**:
   - Labeling embryo images
   - Verifying image integrity
   - Normalizing images for consistent input
   - Augmenting dataset for improved training
   - Splitting dataset into train/validation/test sets

2. **Model Training**:
   - Multiple architecture options (ResNet152, etc.)
   - Hyperparameter optimization
   - Transfer learning capabilities
   - Training progress visualization

3. **Model Evaluation**:
   - Comprehensive metrics calculation
   - Generation of visual reports
   - Comparison with previous models

4. **Dashboard Deployment**:
   - Launch interactive web interface
   - Visualize model performance
   - Perform single-image validation

### Model Evaluation System

The evaluation system (`evaluate_and_visualize.py`) provides:

1. Command-line utility to evaluate models and launch visualization dashboard
2. Options for evaluation-only or dashboard-only operation
3. Customizable port configuration for the web server

## Web Dashboard Features

### Home Page
- Quick actions for model evaluation and image validation
- Overview of available models and evaluation reports
- Recent evaluation history
- Streamlined navigation

### Dashboard Page
- Comprehensive metrics visualization
- Model comparison charts
- Per-class performance analysis
- Interactive elements with loading indicators

### Model Comparison
- Side-by-side comparison of multiple models
- Comparative metrics charts
- Class-specific performance visualization with radar charts
- Sortable metrics tables

### Image Validation Tool
- Upload individual embryo images via drag-and-drop
- Real-time validation with confidence scores
- Probability distribution visualization
- Grad-CAM heatmaps for model interpretability
- Detailed analysis of prediction confidence

## Evaluation Metrics

The system calculates and visualizes:

- **Accuracy**: Overall classification accuracy
- **Precision**: Measure of false positive rate (class-specific and macro-average)
- **Recall**: Measure of false negative rate (class-specific and macro-average)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC Curves**: True positive vs. false positive rates with AUC
- **Confusion Matrix**: Visual representation of class predictions vs. actual values
- **Per-Class Metrics**: Detailed breakdown of performance by embryo quality class

## Explainable AI Implementation

The XAI module provides:

- **Grad-CAM Visualizations**: Highlights regions influencing model decisions
- **Attention Maps**: Overlay heatmaps on original images
- **Batch Processing**: Generate visualizations for multiple images
- **Integration with Validation**: Built into the image validation workflow

## Installation and Setup

### Requirements

All required packages are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

### Project Structure

The project follows a modular structure:

- `/app`: Flask web application
- `/data`: Data storage and processing
- `/models`: Trained model storage
- `/src`: Core functionality modules
- `/outputs`: Results, plots, and reports
- `/uploads`: Temporary storage for uploaded images

## Usage Options

### Option 1: Complete Workflow

Run the entire embryo classification workflow:

```bash
python run_workflow.py
```

This script offers three modes:
- **Automatic**: Uses default values for all steps
- **Interactive**: Prompts for selections at each step
- **Step-by-Step**: Runs each workflow component individually

### Option 2: Model Evaluation Dashboard

Launch the model evaluation dashboard:

```bash
python evaluate_and_visualize.py
```

Additional options:
- `--model`: Specify a particular model to evaluate
- `--evaluate_only`: Run evaluation without launching dashboard
- `--dashboard_only`: Launch dashboard without running evaluation
- `--port`: Specify custom port (default: 5000)

### Option 3: Direct Flask App Usage

Run the Flask application directly:

```bash
python app/app.py
```

## Interpreting Results

### Dashboard Components

The dashboard presents:

1. **Metric Cards**: Summary statistics for quick assessment
2. **Confusion Matrix**: Color-coded visualization of prediction accuracy
3. **ROC Curves**: Performance across different threshold settings
4. **Class Distribution**: Analysis of performance across embryo quality classes
5. **Model Comparison**: Side-by-side evaluation of different models
6. **XAI Visualizations**: Grad-CAM heatmaps for model decisions

### Image Validation Results

When validating individual embryo images:

- **Predicted Class**: Highest probability embryo quality class
- **Confidence Score**: Prediction certainty (0-100%)
- **Class Probabilities**: Distribution across all possible classes
- **Confidence Interpretation**:
  - **High** (≥90%): Reliable prediction
  - **Medium** (70-89%): Consider expert review
  - **Low** (<70%): Low reliability, requires expert verification
- **Grad-CAM Visualization**: Highlighting regions influencing the prediction

## Best Practices

1. **Regular Evaluation**: Reassess models after dataset changes
2. **Multiple Model Comparison**: Train various architectures for best results
3. **Balanced Metrics**: Consider F1 score for imbalanced datasets
4. **Class-Specific Analysis**: Review per-class metrics for targeted improvements
5. **Attention to Gradients**: Review Grad-CAM visualizations to ensure model focuses on embryo features
6. **Confidence Thresholds**: Consider prediction confidence for clinical applications
7. **Expert Verification**: Always have embryologists verify critical predictions

## Troubleshooting

### Common Issues

- **CUDA/GPU Errors**: Check GPU compatibility and CUDA installation
- **Image Loading Errors**: Verify image format and integrity
- **Model Loading Failures**: Ensure model architecture matches saved weights
- **Memory Issues**: Reduce batch size for large models
- **Dashboard Access Problems**: Check port availability and firewall settings
- **Slow Predictions**: Optimize image preprocessing and model inference

### Getting Help

For issues:
1. Check console output for detailed error messages
2. Review Flask server logs for web application errors
3. Examine browser console (F12) for frontend issues
4. Check GPU availability with `python check_gpu.py`

## Future Enhancements

Planned improvements include:

1. Integration with external embryology databases
2. Video sequence analysis for time-lapse embryo development
3. Ensemble learning for improved prediction accuracy
4. Advanced explainability with feature importance analysis
5. Mobile application for on-the-go embryo assessment
6. API endpoints for integration with laboratory information systems
7. Additional visualization tools for deeper model insights
