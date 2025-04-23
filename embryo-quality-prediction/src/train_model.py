import os
from os import path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os.path as path
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

print("Starting embryo quality prediction training script...")

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Get the absolute path to the project root directory
SCRIPT_DIR = path.dirname(path.abspath(__file__))
PROJECT_ROOT = path.dirname(SCRIPT_DIR)

# Configuration
class Config:
    # Data paths
    data_dir = path.join(PROJECT_ROOT, "data", "split")
    train_dir = path.join(data_dir, "train")
    val_dir = path.join(data_dir, "val")
    test_dir = path.join(data_dir, "test")
    
    # Output paths
    output_dir = path.join(PROJECT_ROOT, "models")
    plots_dir = path.join(PROJECT_ROOT, "outputs", "plots")
    results_dir = path.join(PROJECT_ROOT, "outputs", "results")
    
    # Model parameters
    model_name = "resnet152"  # Options: convnext_base, efficientv2t, swinv2, densenet201, efficientnet_b7, resnet152
    pretrained = True
    num_classes = 5  # Will be updated based on dataset
    
    # Training parameters
    batch_size = 32  # As specified in the screenshot
    num_epochs = 30
    learning_rate = 1e-4
    weight_decay = 1e-5
    optimizer_name = "Lion"  # Using Lion optimizer as specified in the screenshot
    scheduler_patience = 3
    early_stopping_patience = 7  # Early stopping as specified in the screenshot
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Image parameters
    img_size = 224
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]   # ImageNet std
    
    # Random seed for reproducibility
    seed = 42

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data transformations
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.mean, std=Config.std)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((Config.img_size, Config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.mean, std=Config.std)
    ])
    
    return train_transform, val_test_transform

# Load datasets
def load_datasets():
    train_transform, val_test_transform = get_transforms()
    
    print("\nChecking for valid class directories...")
    # Get list of class directories that actually contain images
    valid_classes = []
    for class_dir in os.listdir(Config.train_dir):
        class_path = os.path.join(Config.train_dir, class_dir)
        if os.path.isdir(class_path):
            # Check if directory contains any valid image files
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'))]
            if image_files:
                valid_classes.append(class_dir)
                print(f"  ✓ {class_dir}: {len(image_files)} images")
            else:
                print(f"  ✗ {class_dir}: No valid images found (skipping)")
    
    print(f"\nFound {len(valid_classes)} valid classes with images")
    
    # Load datasets from the split directory with only valid classes
    try:
        train_dataset = ImageFolder(root=Config.train_dir, transform=train_transform, is_valid_file=lambda x: x.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')))
        val_dataset = ImageFolder(root=Config.val_dir, transform=val_test_transform, is_valid_file=lambda x: x.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')))
        test_dataset = ImageFolder(root=Config.test_dir, transform=val_test_transform, is_valid_file=lambda x: x.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')))
    except Exception as e:
        print(f"Error loading datasets: {e}")
        # If we still have issues, try a more restrictive approach by only including valid classes
        print("\nTrying alternative approach with only valid classes...")
        
        # Create temporary symbolic links for valid classes only
        import shutil
        temp_train_dir = os.path.join(PROJECT_ROOT, "data", "temp_train")
        temp_val_dir = os.path.join(PROJECT_ROOT, "data", "temp_val")
        temp_test_dir = os.path.join(PROJECT_ROOT, "data", "temp_test")
        
        # Clean up any existing temp directories
        for temp_dir in [temp_train_dir, temp_val_dir, temp_test_dir]:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
        
        # Copy only valid classes
        for class_name in valid_classes:
            # Copy train data
            src_dir = os.path.join(Config.train_dir, class_name)
            dst_dir = os.path.join(temp_train_dir, class_name)
            if os.path.exists(src_dir) and os.listdir(src_dir):
                shutil.copytree(src_dir, dst_dir)
            
            # Copy validation data
            src_dir = os.path.join(Config.val_dir, class_name)
            dst_dir = os.path.join(temp_val_dir, class_name)
            if os.path.exists(src_dir) and os.listdir(src_dir):
                shutil.copytree(src_dir, dst_dir)
            
            # Copy test data
            src_dir = os.path.join(Config.test_dir, class_name)
            dst_dir = os.path.join(temp_test_dir, class_name)
            if os.path.exists(src_dir) and os.listdir(src_dir):
                shutil.copytree(src_dir, dst_dir)
        
        # Load datasets from temporary directories
        train_dataset = ImageFolder(root=temp_train_dir, transform=train_transform)
        val_dataset = ImageFolder(root=temp_val_dir, transform=val_test_transform)
        test_dataset = ImageFolder(root=temp_test_dir, transform=val_test_transform)
    
    # Update number of classes in Config
    Config.num_classes = len(train_dataset.classes)
    Config.class_names = train_dataset.classes
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)
    
    print(f"\nud83dudcca Dataset loaded:")
    print(f"   - Number of classes: {Config.num_classes}")
    print(f"   - Class names: {Config.class_names}")
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(val_dataset)}")
    print(f"   - Test samples: {len(test_dataset)}")
    
    # Print class distribution
    print("\nud83dudcca Class distribution:")
    class_counts = {}
    for dataset_name, dataset in [("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
        counts = {Config.class_names[i]: 0 for i in range(Config.num_classes)}
        for _, label in dataset.samples:
            counts[Config.class_names[label]] += 1
        class_counts[dataset_name] = counts
        
        print(f"   - {dataset_name} set:")
        for class_name, count in counts.items():
            print(f"      - {class_name}: {count} samples ({count/len(dataset)*100:.1f}%)")
    
    return train_loader, val_loader, test_loader, class_counts

# Initialize model
def get_model(model_name, num_classes, pretrained=True):
    if model_name == "resnet152":
        model = models.resnet152(weights='DEFAULT' if pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "densenet201":
        model = models.densenet201(weights='DEFAULT' if pretrained else None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "efficientnet_b7":
        model = models.efficientnet_b7(weights='DEFAULT' if pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "convnext_base":
        model = models.convnext_base(weights='DEFAULT' if pretrained else None)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    
    # Add more models as needed
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    # Create directories for saving results
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.plots_dir, exist_ok=True)
    os.makedirs(Config.results_dir, exist_ok=True)
    
    # Initialize variables
    best_val_acc = 0.0
    best_model_wts = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping_counter = 0
    
    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Training")
        for inputs, labels in train_pbar:
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update progress bar
            train_pbar.set_postfix({"loss": loss.item(), "accuracy": torch.sum(preds == labels.data).item() / inputs.size(0)})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        
        print(f"Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Validation")
        for inputs, labels in val_pbar:
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update progress bar
            val_pbar.set_postfix({"loss": loss.item(), "accuracy": torch.sum(preds == labels.data).item() / inputs.size(0)})
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc.item())
        
        print(f"Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Update learning rate scheduler
        scheduler.step(epoch_loss)
        
        # Save the best model
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            best_model_wts = model.state_dict().copy()
            early_stopping_counter = 0
            
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'class_names': Config.class_names
            }, os.path.join(Config.output_dir, f"{Config.model_name}_best.pth"))
            
            print(f"\u2705 Saved new best model with validation accuracy: {best_val_acc:.4f}")
        else:
            early_stopping_counter += 1
            print(f"\u23f3 Early stopping counter: {early_stopping_counter}/{Config.early_stopping_patience}")
        
        # Early stopping
        if early_stopping_counter >= Config.early_stopping_patience:
            print(f"\u26d4 Early stopping triggered after {epoch+1} epochs")
            break
    
    # Calculate training time
    time_elapsed = time.time() - start_time
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot training and validation loss/accuracy
    plot_training_results(train_losses, val_losses, train_accs, val_accs)
    
    return model

# Plot training results
def plot_training_results(train_losses, val_losses, train_accs, val_accs):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Save the plot
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(Config.plots_dir, f"{Config.model_name}_training_plot_{timestamp}.png"))
    plt.close()

# Evaluate model on test set
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    # Collect predictions
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Print metrics
    print(f"\nud83dudcca Test Set Evaluation:")
    print(f"   - Accuracy: {accuracy:.4f}")
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall: {recall:.4f}")
    print(f"   - F1 Score: {f1:.4f}")
    
    # Generate classification report
    class_report = classification_report(all_labels, all_preds, target_names=Config.class_names, zero_division=0)
    print("\nClassification Report:")
    print(class_report)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, Config.class_names)
    
    # Save results to CSV
    results = {
        'model_name': Config.model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create or append to results CSV
    results_file = os.path.join(Config.results_dir, "model_results.csv")
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
    else:
        results_df = pd.DataFrame([results])
    
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    return accuracy, precision, recall, f1, cm

# Plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the confusion matrix plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(os.path.join(Config.plots_dir, f"{Config.model_name}_confusion_matrix_{timestamp}.png"))
    plt.close()

# Main function
def main():
    print("\n Setting up directories...")
    # Set random seed for reproducibility
    set_seed(Config.seed)
    
    # Create output directories
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.plots_dir, exist_ok=True)
    os.makedirs(Config.results_dir, exist_ok=True)
    print(f"Directories created: {Config.output_dir}, {Config.plots_dir}, {Config.results_dir}")
        
    print(f"\nStarting model training for embryo quality prediction")
    print(f"   - Model: {Config.model_name}")
    print(f"   - Device: {Config.device}")
    print(f"   - Batch size: {Config.batch_size}")
    print(f"   - Learning rate: {Config.learning_rate}")
    print(f"   - Optimizer: {Config.optimizer_name}")
        
    # Load datasets
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, class_names = load_datasets()
    print(f"Datasets loaded with {len(class_names)} classes: {class_names}")
        
    # Initialize model
    print(f"\nInitializing {Config.model_name} model...")
    model = get_model(Config.model_name, Config.num_classes, Config.pretrained)
    model = model.to(Config.device)
    print(f"Model initialized and moved to {Config.device}")
        
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
        
    if Config.optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        print(f"\nUsing AdamW optimizer with learning rate {Config.learning_rate}")
    elif Config.optimizer_name == "Lion":
        # Using Lion optimizer as specified in the screenshot
        try:
            from lion_pytorch import Lion
            optimizer = Lion(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
            print(f"\nUsing Lion optimizer with learning rate {Config.learning_rate}")
        except ImportError:
            print("Lion optimizer not available. Using AdamW instead.")
            optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=Config.scheduler_patience, verbose=True)
    
    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, Config.num_epochs)
    
    # Evaluate the model on test set
    accuracy, precision, recall, f1, _ = evaluate_model(model, test_loader)
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'class_names': Config.class_names,
        'config': {
            'model_name': Config.model_name,
            'img_size': Config.img_size,
            'mean': Config.mean,
            'std': Config.std,
            'num_classes': Config.num_classes
        }
    }, os.path.join(Config.output_dir, f"{Config.model_name}_final.pth"))
    
    print(f"\n\u2705 Model training and evaluation complete!")
    print(f"   - Final test accuracy: {accuracy:.4f}")
    print(f"   - Model saved to {os.path.join(Config.output_dir, f'{Config.model_name}_final.pth')}")

# Main execution
if __name__ == "__main__":
    try:
        print("Starting main function...")
        main()
        print("Training completed successfully!")
    except Exception as e:
        import traceback
        print(f"\n Error during training: {e}")
        traceback.print_exc()
        print("\nStack trace printed above.")
