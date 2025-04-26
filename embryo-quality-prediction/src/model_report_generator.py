#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Report Generator
This script generates comprehensive reports for all trained models, organized by model name with timestamps.
It can be used to:
1. Generate a new report for a model
2. List all available model reports
3. Compare multiple models
"""

import os
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser
from pathlib import Path

# Try to import tabulate, but provide a fallback if not available
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    # Define a simple fallback for tabulate
    def tabulate(data, headers='keys', tablefmt='pretty', showindex=False):
        """Simple fallback for tabulate function"""
        if isinstance(data, pd.DataFrame):
            return data.to_string(index=showindex)
        else:
            # Simple string representation for lists of lists
            result = []
            if headers == 'keys' and isinstance(data[0], dict):
                headers = list(data[0].keys())
                result.append('  '.join(str(h) for h in headers))
                result.append('-' * len(result[0]))
                for row in data:
                    result.append('  '.join(str(row.get(h, '')) for h in headers))
            else:
                if headers and headers != 'keys':
                    result.append('  '.join(str(h) for h in headers))
                    result.append('-' * len(result[0]))
                for row in data:
                    result.append('  '.join(str(cell) for cell in row))
            return '\n'.join(result)

class ModelReportManager:
    """
    Manages model reports and provides utilities for comparing models and generating summaries.
    """
    
    def __init__(self, project_root):
        """Initialize the model report manager with project paths."""
        self.project_root = project_root
        self.results_dir = os.path.join(project_root, "outputs", "results")
        self.summary_file = os.path.join(self.results_dir, "all_models_summary.json")
        
        # Ensure directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load existing summary if available
        self.models_summary = self._load_summary()
    
    def _load_summary(self):
        """Load the summary of all models if it exists."""
        if os.path.exists(self.summary_file):
            try:
                with open(self.summary_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading model summary: {str(e)}")
                return []
        return []
    
    def list_models(self, sort_by="timestamp", ascending=False):
        """
        List all available models with their key metrics.
        
        Parameters:
        -----------
        sort_by : str
            Field to sort by (timestamp, accuracy, precision, recall, f1_score)
        ascending : bool
            Sort in ascending order if True, descending if False
        """
        if not self.models_summary:
            print("No models found in the summary. Train a model first.")
            return None
        
        # Create DataFrame from models summary
        df = pd.DataFrame(self.models_summary)
        
        # Convert metrics to numeric values for proper sorting
        numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort the DataFrame
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)
        
        # Format for display
        display_cols = ['model_name', 'formatted_timestamp', 'accuracy', 'precision', 'recall', 'f1_score']
        display_df = df[display_cols].copy()
        
        # Rename columns for better display
        display_df.columns = ['Model', 'Timestamp', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Format numeric columns
        for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        
        print("\n=== MODEL REPORTS ===")
        print(tabulate(display_df, headers='keys', tablefmt='pretty', showindex=True))
        print(f"\nTotal models: {len(df)}")
        
        return df
    
    def compare_models(self, model_indices=None, model_names=None, metrics=None):
        """
        Compare multiple models based on selected metrics.
        
        Parameters:
        -----------
        model_indices : list
            List of indices to compare (from list_models output)
        model_names : list
            List of model names to compare
        metrics : list
            List of metrics to compare (default: accuracy, precision, recall, f1_score)
        """
        if not self.models_summary:
            print("No models found in the summary. Train a model first.")
            return
        
        # Default metrics to compare
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create DataFrame from models summary
        df = pd.DataFrame(self.models_summary)
        
        # Convert metrics to numeric values
        for metric in metrics:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors='coerce')
        
        # Filter models to compare
        if model_indices is not None:
            if max(model_indices) >= len(df):
                print(f"Error: Index {max(model_indices)} is out of range. Max index is {len(df)-1}.")
                return
            compare_df = df.iloc[model_indices].copy()
        elif model_names is not None:
            compare_df = df[df['model_name'].isin(model_names)].copy()
            if len(compare_df) == 0:
                print(f"Error: No models found with the specified names.")
                return
        else:
            # Compare all models
            compare_df = df.copy()
        
        # Create comparison plots
        self._plot_model_comparison(compare_df, metrics)
        
        # Print comparison table
        display_cols = ['model_name', 'formatted_timestamp'] + metrics
        display_df = compare_df[display_cols].copy()
        
        # Rename columns for better display
        column_mapping = {
            'model_name': 'Model',
            'formatted_timestamp': 'Timestamp',
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1 Score'
        }
        display_df.columns = [column_mapping.get(col, col) for col in display_df.columns]
        
        # Format numeric columns
        for col in metrics:
            display_col = column_mapping.get(col, col)
            if display_col in display_df.columns:
                display_df[display_col] = display_df[display_col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        
        print("\n=== MODEL COMPARISON ===")
        print(tabulate(display_df, headers='keys', tablefmt='pretty', showindex=True))
    
    def _plot_model_comparison(self, df, metrics):
        """Create comparison plots for the selected metrics."""
        # Set up the plot
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        if len(metrics) == 1:
            axes = [axes]
        
        # Create a bar plot for each metric
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                # Sort by the current metric for better visualization
                plot_df = df.sort_values(by=metric, ascending=False)
                
                # Create labels that combine model name and timestamp
                labels = [f"{row['model_name']}\n{row['formatted_timestamp'].split()[0]}" 
                          for _, row in plot_df.iterrows()]
                
                # Plot the metric
                ax = axes[i]
                bars = ax.bar(labels, plot_df[metric], color='skyblue')
                ax.set_title(f"{metric.capitalize()}")
                ax.set_ylim(0, 1.0)  # Metrics are typically between 0 and 1
                ax.set_ylabel(metric.capitalize())
                ax.set_xticklabels(labels, rotation=45, ha='right')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        
        # Save the comparison plot
        comparison_dir = os.path.join(self.results_dir, "comparisons")
        os.makedirs(comparison_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(comparison_dir, f"model_comparison_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Comparison plot saved to: {plot_path}")
    
    def open_report(self, model_index=None, model_name=None, timestamp=None):
        """
        Open a specific model report in the web browser.
        
        Parameters:
        -----------
        model_index : int
            Index of the model from list_models output
        model_name : str
            Name of the model
        timestamp : str
            Timestamp of the model run (to distinguish between runs of the same model)
        """
        if not self.models_summary:
            print("No models found in the summary. Train a model first.")
            return
        
        df = pd.DataFrame(self.models_summary)
        
        # Select the model to open
        if model_index is not None:
            if model_index >= len(df):
                print(f"Error: Index {model_index} is out of range. Max index is {len(df)-1}.")
                return
            report_path = df.iloc[model_index]['report_path']
        elif model_name is not None and timestamp is not None:
            # Find the model with matching name and timestamp
            matching_models = df[(df['model_name'] == model_name) & 
                                 (df['timestamp'] == timestamp)]
            if len(matching_models) == 0:
                print(f"Error: No model found with name '{model_name}' and timestamp '{timestamp}'.")
                return
            report_path = matching_models.iloc[0]['report_path']
        elif model_name is not None:
            # Find the latest model with matching name
            matching_models = df[df['model_name'] == model_name]
            if len(matching_models) == 0:
                print(f"Error: No model found with name '{model_name}'.")
                return
            # Sort by timestamp (descending) and take the first one
            matching_models = matching_models.sort_values(by='timestamp', ascending=False)
            report_path = matching_models.iloc[0]['report_path']
        else:
            print("Please specify either model_index or model_name.")
            return
        
        # Check if the report file exists
        if not os.path.exists(report_path):
            print(f"Error: Report file not found at {report_path}")
            return
        
        # Open the report in the default web browser
        print(f"Opening report: {report_path}")
        webbrowser.open(f"file://{os.path.abspath(report_path)}")

def generate_model_summary_report():
    """Generate a summary report of all models in HTML format."""
    # Get the absolute path to the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Initialize the model report manager
    manager = ModelReportManager(project_root)
    
    # Get all models
    df = manager.list_models(sort_by="accuracy", ascending=False)
    
    if df is None or len(df) == 0:
        print("No models found. Train a model first.")
        return
    
    # Create output directory
    report_dir = os.path.join(project_root, "outputs", "results")
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate HTML report
    report_path = os.path.join(report_dir, "all_models_summary.html")
    
    # Convert metrics to numeric
    numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embryo Quality Prediction - All Models Summary</title>
    <style>
        :root {{
            --primary-color: #4a6fa5;
            --secondary-color: #6b8cbe;
            --accent-color: #ff7e67;
            --background-color: #f9f9f9;
            --card-color: #ffffff;
            --text-color: #333333;
            --border-color: #e0e0e0;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: var(--primary-color);
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        h2 {{
            font-size: 1.8rem;
            margin: 25px 0 15px;
            color: var(--primary-color);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
        }}
        
        h3 {{
            font-size: 1.4rem;
            margin: 20px 0 10px;
            color: var(--secondary-color);
        }}
        
        .card {{
            background-color: var(--card-color);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 25px;
            margin-bottom: 30px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        
        table, th, td {{
            border: 1px solid var(--border-color);
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
        }}
        
        th {{
            background-color: var(--secondary-color);
            color: white;
        }}
        
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        
        .metric-box {{
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 15px;
            border-radius: 8px;
            background-color: #f0f5ff;
            margin: 10px 0;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary-color);
        }}
        
        .metric-label {{
            font-size: 1rem;
            color: var(--text-color);
        }}
        
        .chart-container {{
            width: 100%;
            height: auto;
            margin: 20px 0;
            position: relative;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
            margin: 0 auto;
        }}
        
        footer {{
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            color: #777;
            font-size: 0.9rem;
        }}
        
        .model-link {{
            color: var(--primary-color);
            text-decoration: none;
            font-weight: bold;
        }}
        
        .model-link:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Embryo Quality Prediction</h1>
            <p>All Models Summary Report</p>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </header>
        
        <section class="card">
            <h2>Models Overview</h2>
            <p>Total models: {len(df)}</p>
            
            <h3>Best Performing Models</h3>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Timestamp</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>Report</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add rows for each model
    for i, (_, row) in enumerate(df.iterrows()):
        # Create a link to the model report
        report_path = row.get('report_path', '')
        report_link = f'<a href="file://{os.path.abspath(report_path)}" class="model-link">View Report</a>' if os.path.exists(report_path) else 'N/A'
        
        html_content += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{row['model_name']}</td>
                        <td>{row['formatted_timestamp']}</td>
                        <td>{row['accuracy']:.4f}</td>
                        <td>{row['precision']:.4f}</td>
                        <td>{row['recall']:.4f}</td>
                        <td>{row['f1_score']:.4f}</td>
                        <td>{report_link}</td>
                    </tr>"""
    
    # Create metric comparison charts
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Generate plots
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Sort by the metric for better visualization
        plot_df = df.sort_values(by=metric, ascending=False).head(10)  # Show top 10 for clarity
        
        # Create labels that combine model name and timestamp
        labels = [f"{row['model_name']}\n{row['formatted_timestamp'].split()[0]}" 
                  for _, row in plot_df.iterrows()]
        
        # Plot
        bars = plt.bar(labels, plot_df[metric], color='skyblue')
        plt.title(f"Top Models by {metric.capitalize()}")
        plt.ylim(0, 1.0)
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        
        # Save the plot
        plot_dir = os.path.join(report_dir, "comparison_plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"top_models_by_{metric}.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Add the plot to the HTML
        html_content += f"""
                </tbody>
            </table>
            
            <h3>Top Models by {metric.capitalize()}</h3>
            <div class="chart-container">
                <img src="{os.path.relpath(plot_path, report_dir)}" alt="Top Models by {metric.capitalize()}">
            </div>
        """
    
    # Complete the HTML
    html_content += """
        </section>
        
        <footer>
            <p>Embryo Quality Prediction - All Models Summary</p>
            <p>© 2025 - Generated with Cascade AI</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # Write the HTML to file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nAll models summary report generated: {report_path}")
    
    # Open the report in the default web browser
    webbrowser.open(f"file://{os.path.abspath(report_path)}")

def main():
    """Main function to run the model report generator."""
    # Get the absolute path to the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Initialize the model report manager
    manager = ModelReportManager(project_root)
    
    # Print available commands
    print("\n===== MODEL REPORT GENERATOR =====")
    print("1. List all models")
    print("2. Compare models")
    print("3. Open a specific model report")
    print("4. Generate summary report of all models")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == "1":
                # List all models
                print("\nSort by:")
                print("1. Timestamp (newest first)")
                print("2. Accuracy (highest first)")
                print("3. F1 Score (highest first)")
                
                sort_choice = input("Enter sort option (1-3): ")
                sort_mapping = {
                    "1": ("timestamp", False),
                    "2": ("accuracy", False),
                    "3": ("f1_score", False)
                }
                
                sort_by, ascending = sort_mapping.get(sort_choice, ("timestamp", False))
                manager.list_models(sort_by=sort_by, ascending=ascending)
                
            elif choice == "2":
                # Compare models
                df = manager.list_models()
                if df is not None and len(df) > 0:
                    indices_input = input("\nEnter model indices to compare (comma-separated, or 'all' for all models): ")
                    
                    if indices_input.lower() == 'all':
                        manager.compare_models()
                    else:
                        try:
                            indices = [int(idx.strip()) for idx in indices_input.split(',')]
                            manager.compare_models(model_indices=indices)
                        except ValueError:
                            print("Invalid input. Please enter comma-separated numbers.")
                
            elif choice == "3":
                # Open a specific model report
                df = manager.list_models()
                if df is not None and len(df) > 0:
                    index_input = input("\nEnter the index of the model report to open: ")
                    try:
                        index = int(index_input)
                        manager.open_report(model_index=index)
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                
            elif choice == "4":
                # Generate summary report
                generate_model_summary_report()
                
            elif choice == "5":
                # Exit
                print("Exiting Model Report Generator.")
                break
                
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
