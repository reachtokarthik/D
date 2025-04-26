import os
import sys
import json
import glob
import pandas as pd
import uuid
import base64
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from datetime import datetime

# Get the absolute path to the project root directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.append(PROJECT_ROOT)

# Import evaluation and prediction modules
from src.evaluate_model import ModelEvaluator, find_latest_model
from src.predict_image import EmbryoPredictor

app = Flask(__name__)
app.secret_key = 'embryo_quality_prediction_app'

# Configuration
class AppConfig:
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "results")
    PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
    UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
    PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
    
    # Allowed file extensions for image uploads
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist."""
        for dir_path in [cls.MODELS_DIR, cls.RESULTS_DIR, cls.PLOTS_DIR, cls.UPLOAD_DIR, cls.PREDICTIONS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
            
    @classmethod
    def allowed_file(cls, filename):
        """Check if file has an allowed extension."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Home page showing model evaluation dashboard."""
    AppConfig.ensure_dirs()
    
    # Get list of available models
    models = []
    for model_file in glob.glob(os.path.join(AppConfig.MODELS_DIR, "*.pth")):
        model_name = os.path.basename(model_file)
        models.append({
            'name': model_name,
            'path': model_file,
            'modified': datetime.fromtimestamp(os.path.getmtime(model_file)).strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Sort models by modification time (newest first)
    models.sort(key=lambda x: x['modified'], reverse=True)
    
    # Get list of evaluation reports
    reports = []
    for report_file in glob.glob(os.path.join(AppConfig.RESULTS_DIR, "report_*.html")):
        report_name = os.path.basename(report_file)
        reports.append({
            'name': report_name,
            'path': report_file,
            'modified': datetime.fromtimestamp(os.path.getmtime(report_file)).strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Sort reports by modification time (newest first)
    reports.sort(key=lambda x: x['modified'], reverse=True)
    
    # Load evaluation history if available
    csv_file = os.path.join(AppConfig.RESULTS_DIR, "model_evaluations.csv")
    history = None
    if os.path.exists(csv_file):
        try:
            history = pd.read_csv(csv_file)
            # Convert to list of dicts for template
            history = history.to_dict('records')
        except Exception as e:
            flash(f"Error loading evaluation history: {e}", "danger")
    
    return render_template('index.html', 
                          models=models, 
                          reports=reports, 
                          history=history)


@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    """Evaluate a model and generate a report."""
    model_path = request.form.get('model_path')
    
    if not model_path:
        flash("No model selected for evaluation", "danger")
        return redirect(url_for('index'))
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(model_path)
        
        # Evaluate model
        evaluator.evaluate()
        
        # Save results
        evaluator.save_results()
        
        # Generate HTML report
        html_path = os.path.join(AppConfig.RESULTS_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        report_path = evaluator.generate_html_report(html_path)
        
        flash(f"Model evaluated successfully. Report generated at {os.path.basename(report_path)}", "success")
        
        # Redirect to the report
        return redirect(url_for('view_report', report_path=os.path.basename(report_path)))
    
    except Exception as e:
        flash(f"Error evaluating model: {e}", "danger")
        return redirect(url_for('index'))


@app.route('/report/<path:report_path>')
def view_report(report_path):
    """View a specific evaluation report."""
    # Check if the path is a directory or file
    full_path = os.path.join(AppConfig.RESULTS_DIR, report_path)
    
    # If it's a directory, look for the report file
    if os.path.isdir(full_path):
        # Find HTML files in the directory
        html_files = glob.glob(os.path.join(full_path, "*.html"))
        if html_files:
            full_path = html_files[0]  # Use the first HTML file found
        else:
            flash(f"No report found in {report_path}", "danger")
            return redirect(url_for('index'))
    
    # If it's not a directory and doesn't exist, report error
    if not os.path.exists(full_path):
        flash(f"Report {report_path} not found", "danger")
        return redirect(url_for('index'))
    
    # Read and return the report content with proper content type
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        return report_content, {"Content-Type": "text/html"}
    except Exception as e:
        flash(f"Error reading report: {e}", "danger")
        return redirect(url_for('index'))


@app.route('/compare', methods=['GET', 'POST'])
def compare_models():
    """Compare multiple model evaluations."""
    AppConfig.ensure_dirs()
    
    if request.method == 'POST':
        selected_models = request.form.getlist('selected_models')
        
        if not selected_models:
            flash("No models selected for comparison", "danger")
            return redirect(url_for('compare_models'))
        
        # Load evaluation history
        csv_file = os.path.join(AppConfig.RESULTS_DIR, "model_evaluations.csv")
        if not os.path.exists(csv_file):
            flash("No evaluation history found", "danger")
            return redirect(url_for('index'))
        
        try:
            history = pd.read_csv(csv_file)
            # Filter by selected models
            comparison_data = history[history['model_name'].isin(selected_models)]
            
            if comparison_data.empty:
                flash("No evaluation data found for selected models", "danger")
                return redirect(url_for('compare_models'))
            
            # Convert to list of dicts for template
            comparison_data = comparison_data.to_dict('records')
            
            return render_template('compare.html', 
                                  comparison_data=comparison_data,
                                  selected_models=selected_models)
        
        except Exception as e:
            flash(f"Error loading comparison data: {e}", "danger")
            return redirect(url_for('index'))
    
    # GET request - show selection form
    # Load evaluation history
    csv_file = os.path.join(AppConfig.RESULTS_DIR, "model_evaluations.csv")
    models_to_compare = []
    
    if os.path.exists(csv_file):
        try:
            history = pd.read_csv(csv_file)
            # Get unique model names
            models_to_compare = history['model_name'].unique().tolist()
        except Exception as e:
            flash(f"Error loading evaluation history: {e}", "danger")
    
    return render_template('select_compare.html', models=models_to_compare)


@app.route('/api/model_metrics')
def api_model_metrics():
    """API endpoint to get model metrics for charts."""
    csv_file = os.path.join(AppConfig.RESULTS_DIR, "model_evaluations.csv")
    
    if not os.path.exists(csv_file):
        return jsonify({'error': 'No evaluation history found'})
    
    try:
        history = pd.read_csv(csv_file)
        # Sort by timestamp
        history = history.sort_values('timestamp')
        
        # Prepare data for charts
        data = {
            'labels': history['model_name'].tolist(),
            'timestamps': history['timestamp'].tolist(),
            'accuracy': history['accuracy'].tolist(),
            'precision': history['precision'].tolist(),
            'recall': history['recall'].tolist(),
            'f1_score': history['f1_score'].tolist(),
        }
        
        return jsonify(data)
    
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/validate', methods=['GET', 'POST'])
def validate_image():
    """Validate embryo images using the trained model."""
    AppConfig.ensure_dirs()
    
    # Get list of available models
    models = []
    for model_file in glob.glob(os.path.join(AppConfig.MODELS_DIR, "*.pth")):
        model_name = os.path.basename(model_file)
        models.append({
            'name': model_name,
            'path': model_file,
            'modified': datetime.fromtimestamp(os.path.getmtime(model_file)).strftime('%Y-%m-%d %H:%M:%S')
        })
    
    # Sort models by modification time (newest first)
    models.sort(key=lambda x: x['modified'], reverse=True)
    
    # Handle file upload
    if request.method == 'POST':
        # Check if model is selected
        model_path = request.form.get('model_path')
        if not model_path and models:
            model_path = models[0]['path']
        
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and AppConfig.allowed_file(file.filename):
            # Generate a unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(AppConfig.UPLOAD_DIR, unique_filename)
            
            # Save the file
            file.save(file_path)
            
            try:
                # Initialize predictor with selected model
                try:
                    predictor = EmbryoPredictor(model_path)
                except Exception as e:
                    flash(f"Error initializing predictor: {e}", "danger")
                    import traceback
                    traceback.print_exc()
                    return redirect(request.url)
                
                # Make prediction with detailed error handling
                try:
                    result = predictor.predict(file_path)
                    if result is None:
                        raise ValueError("Prediction returned None result")
                except Exception as e:
                    flash(f"Error during prediction: {e}", "danger")
                    import traceback
                    traceback.print_exc()
                    return redirect(request.url)
                
                # Save prediction
                try:
                    predictor.save_prediction(result)
                except Exception as e:
                    # Non-critical error, just log it
                    print(f"Warning: Could not save prediction: {e}")
                
                # Convert image to base64 for display
                try:
                    with open(file_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                except Exception as e:
                    flash(f"Error encoding image: {e}", "danger")
                    return redirect(request.url)
                
                # Render result template
                return render_template('validation_result.html', 
                                      result=result,
                                      image_data=encoded_image,
                                      image_name=file.filename)
            
            except Exception as e:
                flash(f"Error in validation process: {e}", "danger")
                import traceback
                traceback.print_exc()
                return redirect(request.url)
        else:
            flash(f"Invalid file type. Allowed types: {', '.join(AppConfig.ALLOWED_EXTENSIONS)}", "danger")
            return redirect(request.url)
    
    # GET request - show upload form
    return render_template('validate.html', models=models)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(AppConfig.UPLOAD_DIR, filename)


@app.route('/results/<path:filepath>')
def serve_results(filepath):
    """Serve files from the results directory (images, plots, etc.)."""
    # Extract the directory part from the filepath
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    
    # Construct the full directory path
    full_dir_path = os.path.join(AppConfig.RESULTS_DIR, directory)
    
    return send_from_directory(full_dir_path, filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
