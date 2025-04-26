import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import base64
from io import BytesIO
from PIL import Image

# Get the absolute path to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

# Import the Config class from train_model
from src.train_model import Config, get_transforms

# Import XAI utilities
from src.xai_utils import generate_xai_visualization, generate_batch_xai_visualization, create_combined_visualization

class ModelEvaluator:
    def __init__(self, model_path, test_data_dir=None):
        """
        Initialize the model evaluator.
        
        Args:
            model_path (str): Path to the saved model
            test_data_dir (str, optional): Path to test data directory. 
                                          Defaults to Config.test_dir.
        """
        self.model_path = model_path
        self.test_data_dir = test_data_dir or Config.test_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        self.model = self._load_model()
        
        # Load test data
        self.test_loader, self.class_names = self._load_test_data()
        
        # Results storage
        self.results = {}
        self.figures = {}

    def _load_model(self):
        """Load the trained model."""
        print(f"Loading model from {self.model_path}...")
        
        try:
            # Load the saved model
            model_data = torch.load(self.model_path, map_location=self.device)
            
            # Print model_data keys for debugging
            print(f"Model data keys: {model_data.keys() if isinstance(model_data, dict) else 'Not a dictionary'}")
            
            # Handle different model saving formats
            if isinstance(model_data, dict):
                # PyTorch standard format with state_dict
                if 'state_dict' in model_data:
                    state_dict = model_data['state_dict']
                    print("Using 'state_dict' key")
                elif 'model_state_dict' in model_data:
                    state_dict = model_data['model_state_dict']
                    print("Using 'model_state_dict' key")
                else:
                    # The model_data itself might be the state_dict
                    state_dict = model_data
                    print("Using model_data as state_dict")
                    
                # Extract model architecture
                model_name = model_data.get('model_name', Config.model_name)
            else:
                # The loaded object might be the model itself
                print("Loaded object is not a dictionary, might be the model itself")
                return model_data.to(self.device).eval()
            
            # Print some state_dict keys for debugging
            print(f"State dict keys sample: {list(state_dict.keys())[:5]}")
            
            # Determine number of classes from the state_dict
            if 'fc.weight' in state_dict:
                num_classes = state_dict['fc.weight'].size(0)
                print(f"Found fc.weight with shape: {state_dict['fc.weight'].shape}")
            elif 'module.fc.weight' in state_dict:
                num_classes = state_dict['module.fc.weight'].size(0)
                print(f"Found module.fc.weight with shape: {state_dict['module.fc.weight'].shape}")
                # Adjust keys to remove 'module.' prefix if needed
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            else:
                # Try to find any key that might contain fc.weight
                fc_weight_keys = [k for k in state_dict.keys() if 'fc.weight' in k]
                if fc_weight_keys:
                    key = fc_weight_keys[0]
                    num_classes = state_dict[key].size(0)
                    print(f"Found {key} with shape: {state_dict[key].shape}")
                else:
                    num_classes = model_data.get('num_classes', Config.num_classes)
                    print(f"Using num_classes from config: {num_classes}")
            
            print(f"Detected {num_classes} classes in the saved model")
            
            # Initialize the model architecture
            if model_name == "resnet152":
                model = models.resnet152(weights=None)
                num_ftrs = model.fc.in_features
                # Replace the final fully connected layer
                model.fc = nn.Linear(num_ftrs, num_classes)
                print(f"Created ResNet152 with {num_classes} output classes")
            # Add other model architectures as needed
            
            # Try to load the state dictionary
            try:
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading state dict directly: {e}")
                print("Trying to load with strict=False...")
                model.load_state_dict(state_dict, strict=False)
                
            model = model.to(self.device)
            model.eval()
            
            print(f"Model loaded successfully with {num_classes} classes")
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # As a fallback, try to load the model directly without state_dict
            try:
                print("Attempting to load model directly...")
                model = torch.load(self.model_path, map_location=self.device)
                if isinstance(model, torch.nn.Module):
                    model = model.to(self.device).eval()
                    print("Model loaded directly successfully")
                    return model
            except Exception as e2:
                print(f"Failed to load model directly: {e2}")
                raise e  # Re-raise the original error

    def _load_test_data(self):
        """Load test data."""
        print(f"Loading test data from {self.test_data_dir}...")
        
        # Get transforms
        _, test_transform = get_transforms()
        
        # Load test dataset
        test_dataset = ImageFolder(
            root=self.test_data_dir, 
            transform=test_transform
        )
        
        # Create test dataloader
        test_loader = DataLoader(
            test_dataset, 
            batch_size=Config.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        print(f"Test data loaded with {len(test_dataset)} samples across {len(test_dataset.classes)} classes")
        return test_loader, test_dataset.classes

    def evaluate(self):
        """Evaluate the model and compute all metrics."""
        print("Starting model evaluation...")
        
        # Collect predictions
        all_preds = []
        all_labels = []
        all_probs = []
        all_images = []
        all_image_paths = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Evaluating"):
                try:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass with error checking
                    outputs = self.model(inputs)
                    if outputs is None:
                        print(f"Warning: Model returned None output for batch")
                        continue
                        
                    # Check output shape
                    if len(outputs.shape) != 2:
                        print(f"Warning: Unexpected output shape: {outputs.shape}. Expected shape: [batch_size, num_classes]")
                        continue
                    
                    # Calculate probabilities with error checking
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # Store original images for XAI visualization
                    all_images.append(inputs.cpu())
                    if probabilities is None or len(probabilities) == 0:
                        print(f"Warning: Failed to compute probabilities for batch")
                        continue
                    
                    # Get predictions with error handling
                    max_result = torch.max(outputs, 1)
                    if max_result is None:
                        print(f"Warning: torch.max returned None for batch")
                        continue
                        
                    # Safely unpack the result
                    try:
                        values, preds = max_result
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Failed to unpack torch.max result: {e}. Result was: {max_result}")
                        continue
                    
                    # Store results
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probabilities.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate basic metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Store results
        self.results = {
            'model_name': os.path.basename(self.model_path),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'class_names': self.class_names
        }
        
        # Generate detailed class report
        class_report = classification_report(
            all_labels, all_preds, 
            target_names=self.class_names, 
            zero_division=0, 
            output_dict=True
        )
        self.results['class_report'] = class_report
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self.results['confusion_matrix'] = cm.tolist()
        
        # Find misclassified examples
        misclassified_indices = np.where(all_preds != all_labels)[0]
        self.results['num_misclassified'] = len(misclassified_indices)
        
        # Calculate per-class metrics
        per_class_metrics = []
        for i, class_name in enumerate(self.class_names):
            class_metrics = {
                'class_name': class_name,
                'precision': float(class_report[class_name]['precision']),
                'recall': float(class_report[class_name]['recall']),
                'f1_score': float(class_report[class_name]['f1-score']),
                'support': int(class_report[class_name]['support'])
            }
            per_class_metrics.append(class_metrics)
        self.results['per_class_metrics'] = per_class_metrics
        
        # Generate ROC curves for each class (one-vs-rest)
        self._generate_roc_curves(all_labels, all_probs)
        
        # Generate confusion matrix plot
        self._generate_confusion_matrix_plot(cm)
        
        # Generate per-class metrics plot
        self._generate_per_class_metrics_plot(per_class_metrics)
        
        # Generate XAI visualizations
        self.generate_xai_visualizations()
        
        print("Evaluation completed successfully!")
        return self.results

    def _generate_roc_curves(self, all_labels, all_probs):
        """Generate ROC curves for each class (one-vs-rest)."""
        plt.figure(figsize=(10, 8))
        
        # Store AUC values
        auc_values = []
        
        # Check if the number of classes in probabilities matches expected classes
        num_prob_classes = all_probs.shape[1]
        if num_prob_classes < len(self.class_names):
            print(f"Warning: Model output has {num_prob_classes} classes but {len(self.class_names)} class names provided.")
            print(f"Only generating ROC curves for the first {num_prob_classes} classes.")
            class_names_to_use = self.class_names[:num_prob_classes]
        else:
            class_names_to_use = self.class_names
        
        # Generate ROC curve for each class
        for i, class_name in enumerate(class_names_to_use):
            # Convert to one-vs-rest
            y_true = (all_labels == i).astype(int)
            y_score = all_probs[:, i]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            auc_values.append(float(roc_auc))
            
            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Add diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        
        # Save figure to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64 for HTML embedding
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        self.figures['roc_curves'] = img_str
        
        # Store AUC values in results
        self.results['auc_values'] = dict(zip(self.class_names, auc_values))
        
        # Save to file
        plt.savefig(os.path.join(Config.plots_dir, 'roc_curves.png'))
        plt.close()

    def _generate_confusion_matrix_plot(self, cm):
        """Generate confusion matrix plot."""
        plt.figure(figsize=(10, 8))
        
        # Check if confusion matrix dimensions match the number of class names
        n_classes = cm.shape[0]
        if n_classes != len(self.class_names):
            print(f"Warning: Confusion matrix has {n_classes} classes but {len(self.class_names)} class names provided.")
            print(f"Using only the first {n_classes} class names for the confusion matrix plot.")
            class_names_to_use = self.class_names[:n_classes]
        else:
            class_names_to_use = self.class_names
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_norm, annot=cm, fmt='d', cmap='Blues',
            xticklabels=class_names_to_use,
            yticklabels=class_names_to_use
        )
        
        # Set plot properties
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save figure to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64 for HTML embedding
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        self.figures['confusion_matrix'] = img_str
        
        # Save to file
        plt.savefig(os.path.join(Config.plots_dir, 'confusion_matrix.png'))
        plt.close()

    def _generate_per_class_metrics_plot(self, per_class_metrics):
        """Generate per-class metrics plot."""
        # Extract data
        class_names = [m['class_name'] for m in per_class_metrics]
        precision = [m['precision'] for m in per_class_metrics]
        recall = [m['recall'] for m in per_class_metrics]
        f1_score = [m['f1_score'] for m in per_class_metrics]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set width of bars
        bar_width = 0.25
        index = np.arange(len(class_names))
        
        # Plot bars
        ax.bar(index - bar_width, precision, bar_width, label='Precision', color='#3498db')
        ax.bar(index, recall, bar_width, label='Recall', color='#2ecc71')
        ax.bar(index + bar_width, f1_score, bar_width, label='F1 Score', color='#e74c3c')
        
        # Set plot properties
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(index)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64 for HTML embedding
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        self.figures['per_class_metrics'] = img_str
        
        # Save to file
        plt.savefig(os.path.join(Config.plots_dir, 'per_class_metrics.png'))
        plt.close()
        
    def generate_xai_visualizations(self, num_samples=6):
        """Generate XAI visualizations for a subset of test images."""
        print(f"Generating XAI visualizations for {num_samples} test images...")
        
        # Create directory for XAI visualizations
        xai_dir = os.path.join(Config.results_dir, "xai_visualizations")
        os.makedirs(xai_dir, exist_ok=True)
        
        # Get a subset of test images
        dataset = self.test_loader.dataset
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        # Get image paths and labels
        image_paths = [dataset.samples[i][0] for i in indices]
        labels = [dataset.samples[i][1] for i in indices]
        
        # Get transforms for prediction
        _, transform = get_transforms()
        
        # Generate XAI visualizations
        xai_results = generate_batch_xai_visualization(
            model=self.model,
            image_paths=image_paths,
            transform=transform,
            class_names=self.class_names,
            device=self.device,
            output_dir=xai_dir
        )
        
        # Create combined visualization
        combined_path = os.path.join(xai_dir, "combined_visualization.png")
        create_combined_visualization(xai_results, output_path=combined_path)
        
        # Save combined visualization as base64 for HTML embedding
        with open(combined_path, 'rb') as f:
            img_data = f.read()
            img_str = base64.b64encode(img_data).decode('utf-8')
            self.figures['xai_visualization'] = img_str
        
        # Save XAI results
        self.results['xai_visualizations'] = {
            'combined_path': combined_path,
            'image_paths': image_paths,
            'labels': labels
        }
        
        print(f"XAI visualizations saved to {xai_dir}")
        return xai_dir

    def save_results(self):
        """Save evaluation results to files."""
        # Create output directories if they don't exist
        os.makedirs(Config.results_dir, exist_ok=True)
        os.makedirs(Config.plots_dir, exist_ok=True)
        
        # Save results to JSON
        results_file = os.path.join(Config.results_dir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            # Create a copy of results without figures (they're saved separately)
            results_copy = self.results.copy()
            json.dump(results_copy, f, indent=4)
        
        # Save figures dictionary
        figures_file = os.path.join(Config.results_dir, f"figures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(figures_file, 'w') as f:
            json.dump(self.figures, f)
        
        # Update or create CSV record
        csv_file = os.path.join(Config.results_dir, "model_evaluations.csv")
        
        # Basic metrics for CSV
        csv_data = {
            'model_name': self.results['model_name'],
            'accuracy': self.results['accuracy'],
            'precision': self.results['precision'],
            'recall': self.results['recall'],
            'f1_score': self.results['f1_score'],
            'timestamp': self.results['timestamp'],
            'results_file': os.path.basename(results_file)
        }
        
        # Add per-class F1 scores
        for i, class_name in enumerate(self.class_names):
            csv_data[f'f1_{class_name}'] = self.results['per_class_metrics'][i]['f1_score']
        
        # Create or append to DataFrame
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df = pd.concat([df, pd.DataFrame([csv_data])], ignore_index=True)
        else:
            df = pd.DataFrame([csv_data])
        
        # Save CSV
        df.to_csv(csv_file, index=False)
        
        print(f"Results saved to {results_file}")
        print(f"Figures saved to {figures_file}")
        print(f"CSV record updated at {csv_file}")
        
        return results_file, figures_file

    def generate_html_report(self, output_path=None):
        """Generate an HTML report of the evaluation results."""
        if output_path is None:
            output_path = os.path.join(Config.results_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        # HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Model Evaluation Report</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                .card {{
                    margin-bottom: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .card-header {{
                    background-color: #4a6fa5;
                    color: white;
                    font-weight: bold;
                    border-radius: 10px 10px 0 0 !important;
                }}
                .metric-value {{
                    font-size: 2.5rem;
                    font-weight: bold;
                }}
                .metric-label {{
                    font-size: 0.9rem;
                    color: #6c757d;
                }}
                .chart-container {{
                    position: relative;
                    height: 300px;
                    width: 100%;
                }}
                .table-responsive {{
                    max-height: 400px;
                    overflow-y: auto;
                }}
                .bg-success-light {{
                    background-color: rgba(40, 167, 69, 0.1);
                }}
                .bg-danger-light {{
                    background-color: rgba(220, 53, 69, 0.1);
                }}
            </style>
        </head>
        <body>
            <div class="container-fluid">
                <div class="row mb-4">
                    <div class="col">
                        <h1 class="display-4">Embryo Quality Prediction</h1>
                        <h2 class="text-muted">Model Evaluation Report</h2>
                        <p class="lead">Model: {self.results['model_name']} | Generated: {self.results['timestamp']}</p>
                    </div>
                </div>
                
                <div class="row mb-4">
                    <!-- Overall Metrics -->
                    <div class="col-md-3">
                        <div class="card h-100">
                            <div class="card-header">Accuracy</div>
                            <div class="card-body text-center">
                                <div class="metric-value">{self.results['accuracy']:.2%}</div>
                                <div class="metric-label">Overall classification accuracy</div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card h-100">
                            <div class="card-header">Precision</div>
                            <div class="card-body text-center">
                                <div class="metric-value">{self.results['precision']:.2%}</div>
                                <div class="metric-label">Weighted average precision</div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card h-100">
                            <div class="card-header">Recall</div>
                            <div class="card-body text-center">
                                <div class="metric-value">{self.results['recall']:.2%}</div>
                                <div class="metric-label">Weighted average recall</div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card h-100">
                            <div class="card-header">F1 Score</div>
                            <div class="card-body text-center">
                                <div class="metric-value">{self.results['f1_score']:.2%}</div>
                                <div class="metric-label">Weighted average F1 score</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mb-4">
                    <!-- Confusion Matrix -->
                    <div class="col-md-6">
                        <div class="card h-100">
                            <div class="card-header">Confusion Matrix</div>
                            <div class="card-body text-center">
                                <img src="data:image/png;base64,{self.figures['confusion_matrix']}" 
                                     class="img-fluid" alt="Confusion Matrix">
                            </div>
                        </div>
                    </div>
                    
                    <!-- Per-Class Metrics -->
                    <div class="col-md-6">
                        <div class="card h-100">
                            <div class="card-header">Per-Class Performance</div>
                            <div class="card-body text-center">
                                <img src="data:image/png;base64,{self.figures['per_class_metrics']}" 
                                     class="img-fluid" alt="Per-Class Metrics">
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mb-4">
                    <!-- ROC Curves -->
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">ROC Curves</div>
                            <div class="card-body text-center">
                                <img src="data:image/png;base64,{self.figures['roc_curves']}" 
                                     class="img-fluid" alt="ROC Curves">
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mb-4">
                    <!-- XAI Visualizations -->
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">Explainable AI (XAI) Visualizations</div>
                            <div class="card-body">
                                <p class="mb-3">The visualizations below show both the original embryo images and their corresponding XAI heatmaps, which highlight the regions that influenced the model's predictions. Warmer colors (red/yellow) indicate areas that strongly influenced the classification decision.</p>
                                <div class="text-center">
                                    <img src="data:image/png;base64,{self.figures.get('xai_visualization', '')}" 
                                         class="img-fluid" alt="XAI Visualizations">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mb-4">
                    <!-- Detailed Class Report -->
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">Detailed Classification Report</div>
                            <div class="card-body">
                                <div class="table-responsive">
                                    <table class="table table-striped table-hover">
                                        <thead>
                                            <tr>
                                                <th>Class</th>
                                                <th>Precision</th>
                                                <th>Recall</th>
                                                <th>F1 Score</th>
                                                <th>Support</th>
                                                <th>AUC</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {''.join([f"""
                                            <tr>
                                                <td>{m['class_name']}</td>
                                                <td>{m['precision']:.2%}</td>
                                                <td>{m['recall']:.2%}</td>
                                                <td>{m['f1_score']:.2%}</td>
                                                <td>{m['support']}</td>
                                                <td>{self.results['auc_values'].get(m['class_name'], 0):.3f}</td>
                                            </tr>
                                            """ for m in self.results['per_class_metrics']])}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mb-4">
                    <!-- Model Information -->
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">Model Information</div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <p><strong>Model Name:</strong> {self.results['model_name']}</p>
                                        <p><strong>Evaluation Date:</strong> {self.results['timestamp']}</p>
                                        <p><strong>Number of Classes:</strong> {len(self.class_names)}</p>
                                    </div>
                                    <div class="col-md-6">
                                        <p><strong>Device Used:</strong> {self.device}</p>
                                        <p><strong>Misclassified Examples:</strong> {self.results['num_misclassified']}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <footer class="text-center text-muted my-4">
                    <p>Embryo Quality Prediction Project | Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </footer>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report generated at {output_path}")
        return output_path


def find_latest_model(models_dir=None):
    """Find the latest model file in the models directory."""
    if models_dir is None:
        models_dir = os.path.join(PROJECT_ROOT, "models")
    
    # List all model files
    model_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) 
                  if f.endswith('.pth') or f.endswith('.pt')]
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    # Sort by modification time (newest first)
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Found latest model: {latest_model}")
    return latest_model


def main():
    """Main function to evaluate a model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--model', type=str, help='Path to the model file')
    parser.add_argument('--test_dir', type=str, help='Path to test data directory')
    parser.add_argument('--output', type=str, help='Path to save the HTML report')
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(Config.results_dir, exist_ok=True)
    os.makedirs(Config.plots_dir, exist_ok=True)
    
    # Find latest model if not specified
    model_path = args.model
    if model_path is None:
        try:
            model_path = find_latest_model()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, args.test_dir)
    
    # Evaluate model
    evaluator.evaluate()
    
    # Save results
    evaluator.save_results()
    
    # Generate HTML report
    html_path = args.output or os.path.join(Config.results_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    evaluator.generate_html_report(html_path)
    
    print(f"\nEvaluation completed successfully!")
    print(f"HTML report saved to: {html_path}")


if __name__ == "__main__":
    main()
