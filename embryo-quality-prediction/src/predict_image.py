import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import transforms, models
from datetime import datetime

# Get the absolute path to the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

# Import the Config class from train_model
from src.train_model import Config, get_transforms

class EmbryoPredictor:
    def __init__(self, model_path=None):
        """
        Initialize the embryo image predictor.
        
        Args:
            model_path (str, optional): Path to the model file. If None, will use the latest model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Find latest model if not specified
        if model_path is None:
            model_path = self._find_latest_model()
        
        self.model_path = model_path
        
        # Load the model
        self.model, self.class_names = self._load_model()
        
        # Get transforms for prediction
        _, self.transform = get_transforms()
    
    def _find_latest_model(self):
        """Find the latest model file in the models directory."""
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
                    
                # Extract model architecture and class names
                model_name = model_data.get('model_name', Config.model_name)
                class_names = model_data.get('class_names', None)
            else:
                # The loaded object might be the model itself
                print("Loaded object is not a dictionary, might be the model itself")
                return model_data.to(self.device).eval(), [f"Class_{i}" for i in range(10)]  # Assume 10 classes
            
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
            
            # If class_names is not provided, generate default class names
            if class_names is None:
                class_names = [f"Class_{i}" for i in range(num_classes)]
                
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
            
            print(f"Model loaded successfully with {num_classes} classes: {class_names}")
            return model, class_names
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # As a fallback, try to load the model directly without state_dict
            try:
                print("Attempting to load model directly...")
                model = torch.load(self.model_path, map_location=self.device)
                if isinstance(model, torch.nn.Module):
                    model = model.to(self.device).eval()
                    print("Model loaded directly successfully")
                    # Assume 10 classes if we can't determine
                    return model, [f"Class_{i}" for i in range(10)]
            except Exception as e2:
                print(f"Failed to load model directly: {e2}")
                raise e  # Re-raise the original error
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for prediction.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tensor: Preprocessed image tensor
        """
        try:
            # Open and convert image
            img = Image.open(image_path).convert('RGB')
            
            # Apply transformations
            img_tensor = self.transform(img)
            
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)
            
            return img_tensor.to(self.device)
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {e}")
    
    def predict(self, image_path):
        """
        Predict the class of an embryo image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Prediction results including class, confidence, and all class probabilities
        """
        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image_path)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # Get predicted class and confidence
                confidence, predicted_class = torch.max(probabilities, 0)
                
                # Convert to Python types
                predicted_class = predicted_class.item()
                confidence = confidence.item()
                probabilities = probabilities.cpu().numpy().tolist()
                
                # Create result dictionary
                result = {
                    'image_path': image_path,
                    'predicted_class_index': predicted_class,
                    'predicted_class': self.class_names[predicted_class],
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'class_names': self.class_names,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                return result
        except Exception as e:
            raise ValueError(f"Error predicting image: {e}")
    
    def predict_batch(self, image_paths):
        """
        Predict classes for multiple embryo images.
        
        Args:
            image_paths (list): List of paths to image files
            
        Returns:
            list: List of prediction results
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error predicting {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def save_prediction(self, prediction, output_dir=None):
        """
        Save prediction results to a JSON file.
        
        Args:
            prediction (dict): Prediction result
            output_dir (str, optional): Directory to save results. Defaults to outputs/predictions.
            
        Returns:
            str: Path to the saved file
        """
        if output_dir is None:
            output_dir = os.path.join(PROJECT_ROOT, "outputs", "predictions")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename from image name and timestamp
        image_name = os.path.basename(prediction['image_path'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"pred_{image_name}_{timestamp}.json")
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(prediction, f, indent=4)
        
        return output_file


def main():
    """Main function to predict embryo images."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict embryo image quality')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--output', type=str, help='Output directory for prediction results')
    args = parser.parse_args()
    
    if args.image is None:
        parser.error("Please provide an image path with --image")
    
    # Initialize predictor
    predictor = EmbryoPredictor(args.model)
    
    # Make prediction
    result = predictor.predict(args.image)
    
    # Save prediction
    output_file = predictor.save_prediction(result, args.output)
    
    # Print results
    print("\n===== Prediction Results =====")
    print(f"Image: {result['image_path']}")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
