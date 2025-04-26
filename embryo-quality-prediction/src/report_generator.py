import os
import json
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO
import seaborn as sns
from PIL import Image

class ReportGenerator:
    """
    Generates HTML reports for machine learning model training and evaluation.
    Updates the report incrementally as each step of the workflow completes.
    """
    
    def __init__(self, project_root, model_name="model"):
        """Initialize the report generator with project paths."""
        self.project_root = project_root
        self.model_name = model_name
        
        # Create timestamp for this report
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model-specific directories
        self.model_dir = os.path.join(project_root, "outputs", "results", f"{model_name}_{self.timestamp}")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Set report path with model name and timestamp
        self.report_path = os.path.join(self.model_dir, f"{model_name}_report.html")
        
        # Create model-specific plots directory
        self.plots_dir = os.path.join(self.model_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Check if running in Colab
        self.in_colab = 'google.colab' in str(globals())
        if self.in_colab:
            print("ReportGenerator: Running in Google Colab environment")
        
        # Initialize report data structure
        self.report_data = {
            "model_name": model_name,
            "class_distribution": {},
            "normalization": {},
            "split_info": {},
            "training_params": {},
            "training_metrics": {},
            "evaluation_metrics": {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Load or create HTML template
        self._initialize_report()
    
    def _initialize_report(self):
        """Initialize the HTML report with a template."""
        with open(self.report_path, "w") as f:
            f.write(self._get_html_template())
    
    def _get_html_template(self):
        """Return the HTML template for the report."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embryo Quality Prediction - {self.model_name} - Model Report</title>
    <style>
        :root {
            --primary-color: #4a6fa5;
            --secondary-color: #6b8cbe;
            --accent-color: #ff7e67;
            --background-color: #f9f9f9;
            --card-color: #ffffff;
            --text-color: #333333;
            --border-color: #e0e0e0;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: var(--primary-color);
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        h2 {
            font-size: 1.8rem;
            margin: 25px 0 15px;
            color: var(--primary-color);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }
        
        h3 {
            font-size: 1.4rem;
            margin: 20px 0 10px;
            color: var(--secondary-color);
        }
        
        .card {
            background-color: var(--card-color);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 25px;
            margin-bottom: 30px;
        }
        
        .flex-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: space-between;
        }
        
        .flex-item {
            flex: 1 1 300px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        
        table, th, td {
            border: 1px solid var(--border-color);
        }
        
        th, td {
            padding: 12px 15px;
            text-align: left;
        }
        
        th {
            background-color: var(--secondary-color);
            color: white;
        }
        
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        
        .metric-box {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 15px;
            border-radius: 8px;
            background-color: #f0f5ff;
            margin: 10px 0;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .metric-label {
            font-size: 1rem;
            color: var(--text-color);
        }
        
        .chart-container {
            width: 100%;
            height: auto;
            margin: 20px 0;
            position: relative;
        }
        
        img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
            margin: 0 auto;
        }
        
        .parameter-list {
            list-style-type: none;
        }
        
        .parameter-list li {
            padding: 8px 0;
            border-bottom: 1px dashed var(--border-color);
        }
        
        .parameter-name {
            font-weight: bold;
            color: var(--secondary-color);
        }
        
        footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            color: #777;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Embryo Quality Prediction</h1>
            <p>Model: <span id="model_name">{self.model_name}</span> - Training and Evaluation Report</p>
            <p>Generated on: <span id="date">TIMESTAMP</span></p>
        </header>
        
        <!-- Class Distribution Section -->
        <section id="class_distribution" class="card">
            <h2>1. Class Distribution</h2>
            <div class="flex-container">
                <div class="flex-item">
                    <h3>Sorted Class Counts</h3>
                    <div id="class_distribution_table">
                        <p>Waiting for class distribution data...</p>
                    </div>
                </div>
                <div class="flex-item">
                    <h3>Class Distribution Visualization</h3>
                    <div class="chart-container">
                        <img id="class_distribution_chart" src="" alt="Class Distribution Chart">
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Normalization Details Section -->
        <section id="normalization" class="card">
            <h2>2. Normalization Details</h2>
            <div class="flex-container">
                <div class="flex-item">
                    <h3>Image Preprocessing</h3>
                    <div id="normalization_details">
                        <p>Waiting for normalization data...</p>
                    </div>
                </div>
                <div class="flex-item">
                    <h3>Sample Images Before/After Normalization</h3>
                    <div class="chart-container">
                        <img id="normalization_samples" src="" alt="Normalization Samples">
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Dataset Split Section -->
        <section id="split_info" class="card">
            <h2>3. Dataset Split</h2>
            <div class="flex-container">
                <div class="flex-item">
                    <h3>Split Ratios</h3>
                    <div id="split_table">
                        <p>Waiting for split data...</p>
                    </div>
                </div>
                <div class="flex-item">
                    <h3>Split Strategy</h3>
                    <div id="split_strategy">
                        <p>Waiting for split strategy data...</p>
                    </div>
                    <div class="chart-container">
                        <img id="split_distribution" src="" alt="Split Distribution">
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Training Details Section -->
        <section id="training_params" class="card">
            <h2>4. Training Details</h2>
            
            <div class="flex-container">
                <div class="flex-item">
                    <h3>Model Architecture</h3>
                    <div id="model_architecture">
                        <p>Waiting for model architecture data...</p>
                    </div>
                </div>
                <div class="flex-item">
                    <h3>Training Parameters</h3>
                    <div id="training_parameters">
                        <p>Waiting for training parameters...</p>
                    </div>
                </div>
            </div>
            
            <h3>Data Augmentation</h3>
            <div class="flex-container">
                <div class="flex-item">
                    <div id="augmentation_params">
                        <p>Waiting for augmentation parameters...</p>
                    </div>
                </div>
                <div class="flex-item">
                    <div class="chart-container">
                        <img id="augmentation_samples" src="" alt="Augmentation Samples">
                    </div>
                </div>
            </div>
            
            <h3>Training Progress</h3>
            <div class="flex-container">
                <div class="flex-item">
                    <div class="chart-container">
                        <img id="training_accuracy" src="" alt="Training Accuracy">
                    </div>
                </div>
                <div class="flex-item">
                    <div class="chart-container">
                        <img id="training_loss" src="" alt="Training Loss">
                    </div>
                </div>
            </div>
            
            <h3>Training Metrics</h3>
            <div id="training_metrics" class="flex-container">
                <p>Waiting for training metrics...</p>
            </div>
        </section>
        
        <!-- Evaluation Results Section -->
        <section id="evaluation_metrics" class="card">
            <h2>5. Evaluation Results</h2>
            
            <h3>Test Set Performance</h3>
            <div id="test_metrics" class="flex-container">
                <p>Waiting for evaluation metrics...</p>
            </div>
            
            <h3>Confusion Matrix</h3>
            <div class="chart-container">
                <img id="confusion_matrix" src="" alt="Confusion Matrix">
            </div>
            
            <h3>Per-Class Metrics</h3>
            <div id="per_class_metrics">
                <p>Waiting for per-class metrics...</p>
            </div>
        </section>
        
        <footer>
            <p>Embryo Quality Prediction Model Report</p>
            <p>© 2025 - Generated with Cascade AI</p>
        </footer>
    </div>
    
    <script>
        // Set current date
        document.getElementById('date').textContent = "TIMESTAMP";
    </script>
</body>
</html>"""
    
    def update_class_distribution(self, class_counts):
        """Update the class distribution section of the report."""
        self.report_data["class_distribution"] = class_counts
        
        # Generate class distribution chart
        plt.figure(figsize=(10, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        # Sort by count (descending)
        sorted_indices = np.argsort(counts)[::-1]
        classes = [classes[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        # Create bar chart
        plt.bar(classes, counts, color='#4a6fa5')
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(self.plots_dir, 'class_distribution.png')
        plt.savefig(chart_path)
        plt.close()
        
        # Update HTML
        self._update_html_element('class_distribution_table', self._format_class_table(class_counts))
        self._update_html_image('class_distribution_chart', chart_path)
    
    def update_normalization(self, norm_params):
        """Update the normalization section of the report."""
        self.report_data["normalization"] = norm_params
        
        # Update HTML
        self._update_html_element('normalization_details', self._format_normalization_details(norm_params))
    
    def update_split_info(self, split_info):
        """Update the dataset split section of the report."""
        self.report_data["split_info"] = split_info
        
        # Generate split distribution chart
        plt.figure(figsize=(8, 6))
        splits = list(split_info.keys())
        counts = list(split_info.values())
        
        # Filter out non-numeric values
        numeric_splits = []
        numeric_counts = []
        for i, count in enumerate(counts):
            if isinstance(count, (int, float)) and splits[i] in ['train', 'validation', 'test']:
                numeric_splits.append(splits[i])
                numeric_counts.append(count)
        
        # Create pie chart
        plt.pie(numeric_counts, labels=numeric_splits, autopct='%1.1f%%', 
                colors=['#4a6fa5', '#6b8cbe', '#ff7e67'])
        plt.title('Dataset Split Distribution')
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(self.plots_dir, 'split_distribution.png')
        plt.savefig(chart_path)
        plt.close()
        
        # Update HTML
        self._update_html_element('split_table', self._format_split_table(split_info))
        self._update_html_element('split_strategy', self._format_split_strategy(split_info))
        self._update_html_image('split_distribution', chart_path)
    
    def update_training_params(self, params):
        """Update the training parameters section of the report."""
        self.report_data["training_params"] = params
        
        # Update HTML
        self._update_html_element('model_architecture', self._format_model_architecture(params))
        self._update_html_element('training_parameters', self._format_training_parameters(params))
        if 'augmentation' in params:
            self._update_html_element('augmentation_params', self._format_augmentation_params(params['augmentation']))
    
    def update_training_metrics(self, metrics, history=None):
        """Update the training metrics section of the report."""
        self.report_data["training_metrics"] = metrics
        
        # Generate training plots if history is provided
        if history:
            # Accuracy plot
            plt.figure(figsize=(10, 6))
            plt.plot(history['accuracy'], label='Training Accuracy')
            plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save accuracy plot
            acc_path = os.path.join(self.plots_dir, 'training_accuracy.png')
            plt.savefig(acc_path)
            plt.close()
            
            # Loss plot
            plt.figure(figsize=(10, 6))
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save loss plot
            loss_path = os.path.join(self.plots_dir, 'training_loss.png')
            plt.savefig(loss_path)
            plt.close()
            
            # Update HTML images
            self._update_html_image('training_accuracy', acc_path)
            self._update_html_image('training_loss', loss_path)
        
        # Update HTML
        self._update_html_element('training_metrics', self._format_training_metrics(metrics))
    
    def update_evaluation_metrics(self, metrics, confusion_matrix=None, class_names=None):
        """Update the evaluation metrics section of the report."""
        self.report_data["evaluation_metrics"] = metrics
        
        # Generate confusion matrix plot if provided
        if confusion_matrix is not None and class_names is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            
            # Save confusion matrix plot
            cm_path = os.path.join(self.plots_dir, 'confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()
            
            # Update HTML image
            self._update_html_image('confusion_matrix', cm_path)
        
        # Update HTML
        self._update_html_element('test_metrics', self._format_test_metrics(metrics))
        if 'class_metrics' in metrics:
            self._update_html_element('per_class_metrics', self._format_per_class_metrics(metrics['class_metrics']))
    
    def _format_class_table(self, class_counts):
        """Format class distribution as an HTML table."""
        total = sum(class_counts.values())
        
        # Sort classes by count (descending)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        
        html = """<table>
            <thead>
                <tr>
                    <th>Class</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>"""
        
        for cls, count in sorted_classes:
            percentage = (count / total) * 100 if total > 0 else 0
            html += f"""
                <tr>
                    <td>{cls}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                </tr>"""
        
        html += """
            </tbody>
        </table>"""
        
        return html
    
    def _format_normalization_details(self, norm_params):
        """Format normalization details as HTML."""
        html = """<ul class="parameter-list">"""
        
        for param, value in norm_params.items():
            param_name = param.replace('_', ' ').title()
            html += f"""
                <li><span class="parameter-name">{param_name}:</span> {value}</li>"""
        
        html += """</ul>"""
        
        return html
    
    def _format_split_table(self, split_info):
        """Format dataset split as an HTML table."""
        # Extract split counts
        train_count = split_info.get('train', 0)
        val_count = split_info.get('validation', 0)
        test_count = split_info.get('test', 0)
        
        total = train_count + val_count + test_count
        
        html = """<table>
            <thead>
                <tr>
                    <th>Dataset</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>"""
        
        if total > 0:
            html += f"""
                <tr>
                    <td>Training Set</td>
                    <td>{train_count}</td>
                    <td>{(train_count / total) * 100:.2f}%</td>
                </tr>
                <tr>
                    <td>Validation Set</td>
                    <td>{val_count}</td>
                    <td>{(val_count / total) * 100:.2f}%</td>
                </tr>
                <tr>
                    <td>Test Set</td>
                    <td>{test_count}</td>
                    <td>{(test_count / total) * 100:.2f}%</td>
                </tr>"""
        
        html += """
            </tbody>
        </table>"""
        
        return html
    
    def _format_split_strategy(self, split_info):
        """Format split strategy details as HTML."""
        html = """<ul class="parameter-list">"""
        
        # Filter out count values
        strategy_info = {k: v for k, v in split_info.items() 
                         if k not in ['train', 'validation', 'test'] and not isinstance(v, (int, float))}
        
        for param, value in strategy_info.items():
            param_name = param.replace('_', ' ').title()
            html += f"""
                <li><span class="parameter-name">{param_name}:</span> {value}</li>"""
        
        html += """</ul>"""
        
        return html
    
    def _format_model_architecture(self, params):
        """Format model architecture details as HTML."""
        html = """<ul class="parameter-list">"""
        
        # Extract architecture-related parameters
        arch_params = {
            'Base Model': params.get('model_name', 'Unknown'),
            'Pretrained': 'Yes (ImageNet weights)' if params.get('pretrained', True) else 'No',
            'Fine-tuned': 'Yes' if params.get('fine_tuned', True) else 'No',
            'Output Classes': params.get('num_classes', 'Unknown')
        }
        
        for param, value in arch_params.items():
            html += f"""
                <li><span class="parameter-name">{param}:</span> {value}</li>"""
        
        html += """</ul>"""
        
        return html
    
    def _format_training_parameters(self, params):
        """Format training parameters as HTML."""
        html = """<ul class="parameter-list">"""
        
        # Extract training-related parameters
        train_params = {
            'Optimizer': params.get('optimizer', 'Unknown'),
            'Learning Rate': params.get('learning_rate', 'Unknown'),
            'Batch Size': params.get('batch_size', 'Unknown'),
            'Epochs': params.get('epochs', 'Unknown'),
            'Loss Function': params.get('loss', 'Categorical Cross-Entropy'),
            'Early Stopping': f"Yes (patience = {params.get('early_stopping_patience', 'Unknown')})" 
                             if params.get('early_stopping_patience') else 'No',
            'Learning Rate Scheduler': params.get('lr_scheduler', 'None')
        }
        
        for param, value in train_params.items():
            html += f"""
                <li><span class="parameter-name">{param}:</span> {value}</li>"""
        
        html += """</ul>"""
        
        return html
    
    def _format_augmentation_params(self, aug_params):
        """Format augmentation parameters as HTML."""
        html = """<ul class="parameter-list">"""
        
        for param, value in aug_params.items():
            param_name = param.replace('_', ' ').title()
            html += f"""
                <li><span class="parameter-name">{param_name}:</span> {value}</li>"""
        
        html += """</ul>"""
        
        return html
    
    def _format_training_metrics(self, metrics):
        """Format training metrics as HTML."""
        html = """<div class="flex-container">"""
        
        # Create metric boxes for key metrics
        metric_boxes = [
            ('Best Training Accuracy', metrics.get('best_train_acc', 0), '{:.2%}'),
            ('Best Validation Accuracy', metrics.get('best_val_acc', 0), '{:.2%}'),
            ('Final Training Loss', metrics.get('final_train_loss', 0), '{:.4f}'),
            ('Final Validation Loss', metrics.get('final_val_loss', 0), '{:.4f}')
        ]
        
        for label, value, format_str in metric_boxes:
            formatted_value = format_str.format(value)
            html += f"""
                <div class="flex-item">
                    <div class="metric-box">
                        <span class="metric-value">{formatted_value}</span>
                        <span class="metric-label">{label}</span>
                    </div>
                </div>"""
        
        html += """</div>"""
        
        return html
    
    def _format_test_metrics(self, metrics):
        """Format test metrics as HTML."""
        html = """<div class="flex-container">"""
        
        # Create metric boxes for key metrics
        metric_boxes = [
            ('Test Accuracy', metrics.get('test_accuracy', 0), '{:.2%}'),
            ('Precision (Weighted)', metrics.get('precision', 0), '{:.2f}'),
            ('Recall (Weighted)', metrics.get('recall', 0), '{:.2f}'),
            ('F1 Score (Weighted)', metrics.get('f1_score', 0), '{:.2f}')
        ]
        
        for label, value, format_str in metric_boxes:
            formatted_value = format_str.format(value)
            html += f"""
                <div class="flex-item">
                    <div class="metric-box">
                        <span class="metric-value">{formatted_value}</span>
                        <span class="metric-label">{label}</span>
                    </div>
                </div>"""
        
        html += """</div>"""
        
        return html
    
    def _format_per_class_metrics(self, class_metrics):
        """Format per-class metrics as an HTML table."""
        html = """<table>
            <thead>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Support</th>
                </tr>
            </thead>
            <tbody>"""
        
        for cls, metrics in class_metrics.items():
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                html += f"""
                    <tr>
                        <td>{cls}</td>
                        <td>{metrics.get('precision', 0):.2f}</td>
                        <td>{metrics.get('recall', 0):.2f}</td>
                        <td>{metrics.get('f1-score', 0):.2f}</td>
                        <td>{metrics.get('support', 0)}</td>
                    </tr>"""
        
        html += """
            </tbody>
        </table>"""
        
        return html
        
    def _update_html_element(self, element_id, content):
        """Update an HTML element in the report."""
        try:
            # Try different encodings to handle potential encoding issues
            encodings = ['utf-8', 'latin-1', 'cp1252']
            html = None
            
    except Exception as e:
        print(f"Error updating HTML element: {e}")

def _update_html_image(self, image_id, image_path):
    """Update an image in the HTML report."""
    try:
        # Read the current HTML content
        with open(self.report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Find the image by ID
        img_tag = f'<img id="{image_id}"'
        
        img_pos = html_content.find(img_tag)
        if img_pos == -1:
            print(f"Image with ID '{image_id}' not found in HTML")
            return
        
        # Find src attribute
        src_start = html_content.find('src="', img_pos) + 5
        src_end = html_content.find('"', src_start)
        
        # In Colab, use data URLs instead of file paths
        if self.in_colab and os.path.exists(image_path):
            try:
                with open(image_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                img_type = image_path.split('.')[-1].lower()
                if img_type == 'jpg':
                    img_type = 'jpeg'
                data_url = f'data:image/{img_type};base64,{img_data}'
                new_src = data_url
                print(f"Embedded image {image_id} as data URL for Colab compatibility")
            except Exception as img_err:
                print(f"Could not embed image as data URL: {img_err}")
                # Fallback to relative path
    def _update_html_image(self, image_id, image_path):
        """Update an image in the HTML report."""
        try:
            # Try different encodings to handle potential encoding issues
            encodings = ['utf-8', 'latin-1', 'cp1252']
            html = None
            
            for encoding in encodings:
                try:
                    with open(self.report_path, 'r', encoding=encoding) as f:
                        html = f.read()
                    break  # If successful, break the loop
                except UnicodeDecodeError:
                    continue
            
            if html is None:
                print(f"Could not read HTML file with any of the attempted encodings")
                return
            
            # Find the image by ID
            img_tag = f'<img id="{image_id}"'
            
            img_pos = html.find(img_tag)
            if img_pos == -1:
                print(f"Image with ID '{image_id}' not found in HTML")
                return
            
            # Find src attribute
            src_start = html.find('src="', img_pos) + 5
            src_end = html.find('"', src_start)
            
            # In Colab, use data URLs instead of file paths
            if self.in_colab and os.path.exists(image_path):
                try:
                    with open(image_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    img_type = image_path.split('.')[-1].lower()
                    if img_type == 'jpg':
                        img_type = 'jpeg'
                    data_url = f'data:image/{img_type};base64,{img_data}'
                    new_src = data_url
                    print(f"Embedded image {image_id} as data URL for Colab compatibility")
                except Exception as img_err:
                    print(f"Could not embed image as data URL: {img_err}")
                    # Fallback to relative path
                    rel_path = os.path.relpath(image_path, os.path.dirname(self.report_path))
                    rel_path = rel_path.replace('\\', '/')
                    new_src = rel_path
            else:
                # Get relative path for HTML in non-Colab environments
                rel_path = os.path.relpath(image_path, os.path.dirname(self.report_path))
                rel_path = rel_path.replace('\\', '/')
                new_src = rel_path
            
            # Replace the src
            new_html = html[:src_start] + new_src + html[src_end:]
            
            with open(self.report_path, 'w', encoding='utf-8') as f:
                f.write(new_html)
                
        except Exception as e:
            print(f"Error updating HTML image: {e}")
    
    def save_report_data(self):
        """Save report data to JSON file for later reference."""
        try:
            # Save report data as JSON with model name and timestamp
            report_data_path = os.path.join(self.model_dir, f"{self.model_name}_report_data.json")
            with open(report_data_path, 'w') as f:
                json.dump(self.report_data, f, indent=4)
            print(f"Report data saved to {report_data_path}")
            
            # Also save a summary file with key metrics
            summary_data = {
                "model_name": self.model_name,
                "timestamp": self.timestamp,
                "formatted_timestamp": self.report_data["timestamp"],
                "training_time": self.report_data.get("training_metrics", {}).get("training_time", "N/A"),
                "accuracy": self.report_data.get("evaluation_metrics", {}).get("accuracy", "N/A"),
                "precision": self.report_data.get("evaluation_metrics", {}).get("precision", "N/A"),
                "recall": self.report_data.get("evaluation_metrics", {}).get("recall", "N/A"),
                "f1_score": self.report_data.get("evaluation_metrics", {}).get("f1_score", "N/A"),
                "report_path": self.report_path
            }
            
            # Save summary to model directory
            summary_path = os.path.join(self.model_dir, f"{self.model_name}_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=4)
                
            # Also save to a master summary file that contains all model runs
            master_summary_path = os.path.join(self.project_root, "outputs", "results", "all_models_summary.json")
            
            # Load existing summary if it exists
            all_models = []
            if os.path.exists(master_summary_path):
                try:
                    with open(master_summary_path, 'r') as f:
                        all_models = json.load(f)
                except:
                    all_models = []
            
            # Append this model's summary
            all_models.append(summary_data)
            
            # Save updated summary
            with open(master_summary_path, 'w') as f:
                json.dump(all_models, f, indent=4)
                
            print(f"Model summary saved to {summary_path} and added to master summary")
        except Exception as e:
            print(f"Error saving report data: {str(e)}")
