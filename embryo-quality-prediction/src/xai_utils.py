import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import cv2
from torchvision import transforms
import base64
from io import BytesIO

class GradCAM:
    """
    Grad-CAM implementation for CNN model visualization
    """
    def __init__(self, model, target_layer_name='layer4'):
        """
        Initialize GradCAM with a model and target layer
        
        Args:
            model: PyTorch model
            target_layer_name: Name of the target layer for visualization
        """
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        
        # Get the target layer
        if hasattr(model, 'layer4'):  # ResNet
            self.target_layer = model.layer4
        elif hasattr(model, 'features'):  # DenseNet, VGG
            self.target_layer = model.features
        else:
            raise ValueError(f"Could not find target layer: {target_layer_name}")
            
        # Register hooks
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)
        
    def _forward_hook(self, module, input, output):
        """Store activations during forward pass"""
        self.activations = output
        
    def _backward_hook(self, module, grad_input, grad_output):
        """Store gradients during backward pass"""
        self.gradients = grad_output[0]
        
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Class Activation Map
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            target_class: Target class for visualization, if None uses predicted class
            
        Returns:
            cam: Class activation map
            probs: Class probabilities
            predicted_class: Predicted class index
        """
        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        # Get predicted class if target_class is None
        if target_class is None:
            target_class = torch.argmax(probs, dim=1).item()
        
        # Get predicted probability
        predicted_prob = probs[0, target_class].item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get weights
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # Take average of gradients over each channel
        weights = np.mean(gradients, axis=(1, 2))
        
        # Create weighted sum of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to focus on positive contributions
        cam = np.maximum(cam, 0)
        
        # Normalize CAM
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        return cam, probs.detach().cpu().numpy()[0], target_class
    
    def overlay_cam_on_image(self, img, cam, use_rgb=True, colormap=cv2.COLORMAP_JET, alpha=0.5):
        """
        Overlay CAM on input image
        
        Args:
            img: Input image (PIL Image or numpy array)
            cam: Class activation map
            use_rgb: Whether to use RGB (True) or BGR (False)
            colormap: OpenCV colormap for heatmap
            alpha: Transparency factor
            
        Returns:
            visualization: CAM overlaid on input image
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(img, Image.Image):
            img = np.array(img)
            
        # Resize CAM to match image size
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        # Apply colormap to create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        
        # Convert to RGB if needed
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
        # Create weighted overlay
        visualization = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
        
        return visualization
    
def generate_xai_visualization(model, image_path, transform, class_names, device):
    """
    Generate XAI visualization for an image
    
    Args:
        model: PyTorch model
        image_path: Path to input image
        transform: Preprocessing transforms
        class_names: List of class names
        device: Device to run model on
        
    Returns:
        result_dict: Dictionary containing visualization results
    """
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Create GradCAM object
    grad_cam = GradCAM(model)
    
    # Generate CAM
    cam, probs, predicted_class = grad_cam.generate_cam(input_tensor)
    
    # Get original image as numpy array
    orig_img = np.array(img)
    
    # Generate heatmap visualization
    xai_visualization = grad_cam.overlay_cam_on_image(orig_img, cam)
    
    # Convert images to base64 for display
    orig_img_pil = Image.fromarray(orig_img)
    xai_img_pil = Image.fromarray(xai_visualization)
    
    # Save to buffer
    orig_buffer = BytesIO()
    xai_buffer = BytesIO()
    
    orig_img_pil.save(orig_buffer, format="PNG")
    xai_img_pil.save(xai_buffer, format="PNG")
    
    # Convert to base64
    orig_img_base64 = base64.b64encode(orig_buffer.getvalue()).decode('utf-8')
    xai_img_base64 = base64.b64encode(xai_buffer.getvalue()).decode('utf-8')
    
    # Create result dictionary
    result_dict = {
        'original_image': orig_img_base64,
        'xai_image': xai_img_base64,
        'predicted_class': class_names[predicted_class],
        'predicted_class_index': predicted_class,
        'confidence': probs[predicted_class],
        'probabilities': probs.tolist(),
        'class_names': class_names
    }
    
    return result_dict

def generate_batch_xai_visualization(model, image_paths, transform, class_names, device, output_dir=None):
    """
    Generate XAI visualizations for multiple images
    
    Args:
        model: PyTorch model
        image_paths: List of paths to input images
        transform: Preprocessing transforms
        class_names: List of class names
        device: Device to run model on
        output_dir: Directory to save visualizations (optional)
        
    Returns:
        results: List of dictionaries containing visualization results
    """
    results = []
    
    for image_path in image_paths:
        result = generate_xai_visualization(model, image_path, transform, class_names, device)
        results.append(result)
        
        # Save visualization if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename
            base_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(base_name)[0]
            
            # Convert base64 back to image
            xai_img_data = base64.b64decode(result['xai_image'])
            xai_img = Image.open(BytesIO(xai_img_data))
            
            # Save image
            output_path = os.path.join(output_dir, f"{name_without_ext}_xai.png")
            xai_img.save(output_path)
            
    return results

def create_combined_visualization(results, output_path=None, max_images=6, figsize=(15, 10)):
    """
    Create a combined visualization of original images and XAI heatmaps
    
    Args:
        results: List of dictionaries containing visualization results
        output_path: Path to save the combined visualization (optional)
        max_images: Maximum number of images to include
        figsize: Figure size
        
    Returns:
        fig: Matplotlib figure
    """
    # Limit number of images
    results = results[:max_images]
    num_images = len(results)
    
    # Calculate grid dimensions
    cols = min(3, num_images)
    rows = (num_images + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows * 2, cols, figsize=figsize)
    if rows * cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows * 2, cols)
    
    # Flatten axes for easy indexing
    axes = axes.flatten()
    
    # Plot images
    for i, result in enumerate(results):
        # Get images
        orig_img_data = base64.b64decode(result['original_image'])
        xai_img_data = base64.b64decode(result['xai_image'])
        
        orig_img = Image.open(BytesIO(orig_img_data))
        xai_img = Image.open(BytesIO(xai_img_data))
        
        # Plot original image
        axes[i * 2].imshow(orig_img)
        if i == 0:
            axes[i * 2].set_title('Original Image')
        axes[i * 2].axis('off')
        
        # Plot XAI image
        axes[i * 2 + 1].imshow(xai_img)
        if i == 0:
            axes[i * 2 + 1].set_title('XAI')
        axes[i * 2 + 1].axis('off')
        
        # Add text box with prediction info
        pred_class = result['predicted_class']
        confidence = result['confidence'] * 100
        
        # Create text box
        textstr = f'Predicted Class: {pred_class}\nConfidence Score: {confidence:.2f}%'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        
        # Add text box to XAI image
        axes[i * 2 + 1].text(0.05, 0.95, textstr, transform=axes[i * 2 + 1].transAxes,
                            fontsize=10, verticalalignment='top', bbox=props)
    
    # Hide unused axes
    for i in range(num_images * 2, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
