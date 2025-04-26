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

class SimpleReportGenerator:
    """
    A simplified report generator that avoids any potential variable reference issues.
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
        
        # Create a basic HTML report
        self._create_basic_report()
    
    def _create_basic_report(self):
        """Create a basic HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.model_name} - Training Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #4a6fa5; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4a6fa5; color: white; }}
        .metric-box {{ 
            border: 1px solid #ddd; 
            padding: 15px; 
            margin: 10px; 
            text-align: center;
            border-radius: 5px;
            background-color: #f9f9f9;
        }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #4a6fa5; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>{self.model_name} - Training Report</h1>
    <p>Generated on: {self.report_data["timestamp"]}</p>
    
    <h2>1. Class Distribution</h2>
    <div style="display: flex; flex-wrap: wrap;">
        <div style="flex: 1 1 300px;">
            <h3>Class Counts</h3>
            <div id="class_distribution_table">
                <p>Waiting for class distribution data...</p>
            </div>
        </div>
        <div style="flex: 1 1 300px;">
            <h3>Class Distribution Visualization</h3>
            <div>
                <img id="class_distribution_chart" src="" alt="Class Distribution Chart">
            </div>
        </div>
    </div>
    
    <h2>2. Normalization Details</h2>
    <div style="display: flex; flex-wrap: wrap;">
        <div style="flex: 1 1 300px;">
            <h3>Image Preprocessing</h3>
            <div id="normalization_details">
                <p>Waiting for normalization data...</p>
            </div>
        </div>
        <div style="flex: 1 1 300px;">
            <h3>Sample Images</h3>
            <div>
                <img id="normalization_samples" src="" alt="Normalization Samples">
            </div>
        </div>
    </div>
    
    <h2>3. Dataset Split</h2>
    <div style="display: flex; flex-wrap: wrap;">
        <div style="flex: 1 1 300px;">
            <h3>Split Ratios</h3>
            <div id="split_table">
                <p>Waiting for split data...</p>
            </div>
        </div>
        <div style="flex: 1 1 300px;">
            <h3>Split Strategy</h3>
            <div id="split_strategy">
                <p>Waiting for split strategy data...</p>
            </div>
            <div>
                <img id="split_distribution" src="" alt="Split Distribution">
            </div>
        </div>
    </div>
    
    <h2>4. Training Parameters</h2>
    <div id="training_parameters">
        <p>Waiting for training parameters...</p>
    </div>
    
    <h2>Training Metrics</h2>
    <div id="training_metrics">
        <p>Waiting for training metrics...</p>
    </div>
    
    <h2>Training Progress</h2>
    <div>
        <img id="training_accuracy" src="" alt="Training Accuracy">
        <img id="training_loss" src="" alt="Training Loss">
    </div>
    
    <h2>Evaluation Results</h2>
    <div id="test_metrics">
        <p>Waiting for evaluation metrics...</p>
    </div>
    
    <h2>Confusion Matrix</h2>
    <div>
        <img id="confusion_matrix" src="" alt="Confusion Matrix">
    </div>
    
    <h2>Per-Class Metrics</h2>
    <div id="per_class_metrics">
        <p>Waiting for per-class metrics...</p>
    </div>
    
    <footer>
        <p>Embryo Quality Prediction - {self.timestamp}</p>
    </footer>
</body>
</html>"""
        
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(html)
    
    def update_normalization(self, norm_params):
        """Update the normalization section of the report."""
        self.report_data["normalization"] = norm_params
        
        # Format as HTML
        html = "<table>"
        html += "<tr><th>Parameter</th><th>Value</th></tr>"
        
        for key, value in norm_params.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html += "</table>"
        
        self._update_element("normalization_details", html)
    
    def update_class_distribution(self, class_counts):
        """Update the class distribution section of the report."""
        self.report_data["class_distribution"] = class_counts
        
        # Format as HTML
        html = "<table>"
        html += "<tr><th>Class</th><th>Count</th><th>Percentage</th></tr>"
        
        total = sum(class_counts.values())
        for class_name, count in class_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            html += f"<tr><td>{class_name}</td><td>{count}</td><td>{percentage:.2f}%</td></tr>"
        
        html += "</table>"
        
        self._update_element("class_distribution_table", html)
        
        # Create a bar chart for class distribution
        try:
            plt.figure(figsize=(10, 6))
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            # Sort by count (descending)
            sorted_indices = np.argsort(counts)[::-1]
            classes = [classes[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
            
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
            
            # Update image in report
            self._update_image("class_distribution_chart", chart_path)
        except Exception as e:
            print(f"Error creating class distribution chart: {e}")
    
    def update_split_info(self, split_info):
        """Update the dataset split section of the report."""
        self.report_data["split_info"] = split_info
        
        # Format as HTML
        html = "<table>"
        html += "<tr><th>Set</th><th>Count</th><th>Percentage</th></tr>"
        
        total = sum(split_info.get("counts", {}).values())
        for set_name, count in split_info.get("counts", {}).items():
            percentage = (count / total) * 100 if total > 0 else 0
            html += f"<tr><td>{set_name}</td><td>{count}</td><td>{percentage:.2f}%</td></tr>"
        
        html += "</table>"
        
        self._update_element("split_table", html)
        
        # Format split strategy
        strategy_html = f"<p><strong>Strategy:</strong> {split_info.get('strategy', 'Unknown')}</p>"
        if "ratios" in split_info:
            strategy_html += "<p><strong>Ratios:</strong></p>"
            strategy_html += "<ul>"
            for set_name, ratio in split_info["ratios"].items():
                strategy_html += f"<li>{set_name}: {ratio}</li>"
            strategy_html += "</ul>"
        
        self._update_element("split_strategy", strategy_html)
    
    def update_training_params(self, params):
        """Update the training parameters section of the report."""
        self.report_data["training_params"] = params
        
        # Format as HTML
        html = "<table>"
        html += "<tr><th>Parameter</th><th>Value</th></tr>"
        
        for key, value in params.items():
            if key == "augmentation" and isinstance(value, dict):
                aug_str = "<ul>"
                for aug_key, aug_val in value.items():
                    aug_str += f"<li>{aug_key}: {aug_val}</li>"
                aug_str += "</ul>"
                value = aug_str
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html += "</table>"
        
        self._update_element("training_parameters", html)
    
    def update_training_metrics(self, metrics, history=None):
        """Update the training metrics section of the report."""
        self.report_data["training_metrics"] = metrics
        
        # Format metrics as HTML
        html = "<div style='display: flex; flex-wrap: wrap;'>"
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                html += f"""
                <div class="metric-box">
                    <div class="metric-value">{value:.4f}</div>
                    <div class="metric-label">{key}</div>
                </div>"""
        
        html += "</div>"
        
        self._update_element("training_metrics", html)
        
        # Plot training history if provided
        if history and all(k in history for k in ["accuracy", "val_accuracy", "loss", "val_loss"]):
            # Plot accuracy
            plt.figure(figsize=(10, 6))
            plt.plot(history["accuracy"], label="Training Accuracy")
            plt.plot(history["val_accuracy"], label="Validation Accuracy")
            plt.title("Training and Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            
            # Save accuracy plot
            acc_path = os.path.join(self.plots_dir, "training_accuracy.png")
            plt.savefig(acc_path)
            plt.close()
            
            # Plot loss
            plt.figure(figsize=(10, 6))
            plt.plot(history["loss"], label="Training Loss")
            plt.plot(history["val_loss"], label="Validation Loss")
            plt.title("Training and Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            
            # Save loss plot
            loss_path = os.path.join(self.plots_dir, "training_loss.png")
            plt.savefig(loss_path)
            plt.close()
            
            # Update images in report
            self._update_image("training_accuracy", acc_path)
            self._update_image("training_loss", loss_path)
    
    def update_evaluation_metrics(self, metrics, confusion_matrix=None, class_names=None):
        """Update the evaluation metrics section of the report."""
        self.report_data["evaluation_metrics"] = metrics
        
        # Format metrics as HTML
        html = "<div style='display: flex; flex-wrap: wrap;'>"
        
        for key, value in metrics.items():
            if key != "class_metrics" and isinstance(value, (int, float)):
                html += f"""
                <div class="metric-box">
                    <div class="metric-value">{value:.4f}</div>
                    <div class="metric-label">{key}</div>
                </div>"""
        
        html += "</div>"
        
        self._update_element("test_metrics", html)
        
        # Format per-class metrics if available
        if "class_metrics" in metrics and class_names:
            class_html = "<table>"
            class_html += "<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Support</th></tr>"
            
            for class_name in class_names:
                if class_name in metrics["class_metrics"]:
                    class_data = metrics["class_metrics"][class_name]
                    class_html += f"""<tr>
                        <td>{class_name}</td>
                        <td>{class_data['precision']:.4f}</td>
                        <td>{class_data['recall']:.4f}</td>
                        <td>{class_data['f1-score']:.4f}</td>
                        <td>{class_data['support']}</td>
                    </tr>"""
            
            class_html += "</table>"
            
            self._update_element("per_class_metrics", class_html)
        
        # Plot confusion matrix if provided
        if confusion_matrix is not None and class_names:
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names)
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            
            # Save confusion matrix plot
            cm_path = os.path.join(self.plots_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            plt.close()
            
            # Update image in report
            self._update_image("confusion_matrix", cm_path)
    
    def _update_element(self, element_id, content):
        """Update an HTML element in the report by its ID."""
        try:
            with open(self.report_path, "r", encoding="utf-8") as f:
                html = f.read()
            
            # Find the element by ID
            start_tag = f'id="{element_id}"'
            start_pos = html.find(start_tag)
            
            if start_pos == -1:
                print(f"Element with ID '{element_id}' not found in the report.")
                return
            
            # Find the div that contains this ID
            div_start = html.rfind("<div", 0, start_pos)
            if div_start == -1:
                print(f"Could not find div start for element '{element_id}'.")
                return
            
            # Find the end of the opening div tag
            div_end = html.find(">", start_pos)
            if div_end == -1:
                print(f"Could not find div end for element '{element_id}'.")
                return
            
            # Find the closing div tag
            close_div = html.find("</div>", div_end)
            if close_div == -1:
                print(f"Could not find closing div for element '{element_id}'.")
                return
            
            # Replace the content
            new_html = html[:div_end+1] + "\n" + content + "\n" + html[close_div:]
            
            with open(self.report_path, "w", encoding="utf-8") as f:
                f.write(new_html)
                
        except Exception as e:
            print(f"Error updating element '{element_id}': {e}")
    
    def _update_image(self, image_id, image_path):
        """Update an image in the report by its ID."""
        try:
            with open(self.report_path, "r", encoding="utf-8") as f:
                html = f.read()
            
            # Find the image by ID
            img_tag = f'id="{image_id}"'
            img_pos = html.find(img_tag)
            
            if img_pos == -1:
                print(f"Image with ID '{image_id}' not found in the report.")
                return
            
            # Find the img tag that contains this ID
            img_start = html.rfind("<img", 0, img_pos)
            if img_start == -1:
                print(f"Could not find img tag for image '{image_id}'.")
                return
            
            # Find the end of the img tag
            img_end = html.find(">", img_pos)
            if img_end == -1:
                print(f"Could not find end of img tag for image '{image_id}'.")
                return
            
            # Get the relative path to the image
            if os.path.isabs(image_path):
                report_dir = os.path.dirname(os.path.abspath(self.report_path))
                rel_path = os.path.relpath(image_path, report_dir)
                image_path = rel_path.replace("\\", "/")
            
            # Create a new img tag with the updated src
            img_tag_content = html[img_start:img_end]
            if "src=" in img_tag_content:
                # Replace existing src
                new_img_tag = img_tag_content.split("src=")[0] + f'src="{image_path}"'
                if "alt=" in img_tag_content:
                    new_img_tag += ' ' + img_tag_content.split("alt=")[1]
            else:
                # Add src attribute
                new_img_tag = img_tag_content + f' src="{image_path}"'
            
            # Replace the img tag
            new_html = html[:img_start] + new_img_tag + html[img_end:]
            
            with open(self.report_path, "w", encoding="utf-8") as f:
                f.write(new_html)
                
        except Exception as e:
            print(f"Error updating image '{image_id}': {e}")
    
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
                "accuracy": self.report_data.get("evaluation_metrics", {}).get("test_accuracy", "N/A"),
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
