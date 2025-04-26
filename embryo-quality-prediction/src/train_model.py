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

# Import the report generator
from src.report_generator import ReportGenerator

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
    # Initialize report generator
    report_generator = ReportGenerator(PROJECT_ROOT)
    
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
    
    print(f"\n📊 Dataset loaded:")
    print(f"   - Number of classes: {Config.num_classes}")
    print(f"   - Class names: {Config.class_names}")
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(val_dataset)}")
    print(f"   - Test samples: {len(test_dataset)}")
    
    # Print class distribution
    print("\n📊 Class distribution:")
    class_counts = {}
    for dataset_name, dataset in [("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
        counts = {Config.class_names[i]: 0 for i in range(Config.num_classes)}
        for _, label in dataset.samples:
            counts[Config.class_names[label]] += 1
        class_counts[dataset_name] = counts
        
        print(f"   - {dataset_name} set:")
        for class_name, count in counts.items():
            print(f"      - {class_name}: {count} samples ({count/len(dataset)*100:.1f}%)")
    
    # Update report with normalization details
    norm_params = {
        "rescaling": "1/255 (ToTensor)",
        "image_size": f"{Config.img_size}x{Config.img_size}",
        "mean": str(Config.mean),
        "std": str(Config.std)
    }
    report_generator.update_normalization(norm_params)
    
    # Update report with class distribution
    report_generator.update_class_distribution(class_counts["Train"])
    
    # Update report with split information
    split_info = {
        "train": len(train_dataset),
        "validation": len(val_dataset),
        "test": len(test_dataset),
        "strategy": "Stratified split",
        "seed": Config.seed
    }
    report_generator.update_split_info(split_info)
    
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
    
    elif model_name == "swinv2":
        try:
            model = models.swin_v2_b(weights='DEFAULT' if pretrained else None)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes)
        except AttributeError:
            # Fallback for older torchvision versions
            print("Warning: swinv2_b not available in this torchvision version. Using swin_b instead.")
            model = models.swin_b(weights='DEFAULT' if pretrained else None)
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, num_classes)
    
    elif model_name == "efficientvit":
        try:
            # Try to import EfficientViT from torchvision
            model = models.efficientvit_b1(weights='DEFAULT' if pretrained else None)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        except (AttributeError, ImportError):
            # Fallback to EfficientNet if EfficientViT is not available
            print("Warning: EfficientViT not available in this torchvision version. Using EfficientNet-B0 instead.")
            model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    # Initialize report generator
    report_generator = ReportGenerator(PROJECT_ROOT)
    
    # Check if running in Colab
    in_colab = 'google.colab' in str(globals())
    if in_colab:
        print("\n📊 Running in Google Colab - will display training progress in notebook")
        try:
            from IPython.display import clear_output
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            # Function to display training progress in Colab
            def display_progress(train_losses, val_losses, train_accs, val_accs):
                clear_output(wait=True)
                
                fig = plt.figure(figsize=(12, 10))
                gs = GridSpec(2, 2, figure=fig)
                
                # Loss plot
                ax1 = fig.add_subplot(gs[0, :])
                ax1.plot(train_losses, label='Training Loss')
                ax1.plot(val_losses, label='Validation Loss')
                ax1.set_title('Training and Validation Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True)
                
                # Accuracy plot
                ax2 = fig.add_subplot(gs[1, 0])
                ax2.plot(train_accs, label='Training Accuracy')
                ax2.plot(val_accs, label='Validation Accuracy')
                ax2.set_title('Training and Validation Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()
                ax2.grid(True)
                
                # Metrics table
                ax3 = fig.add_subplot(gs[1, 1])
                ax3.axis('tight')
                ax3.axis('off')
                
                # Create metrics table
                last_epoch = len(train_losses) - 1
                best_val_acc_epoch = val_accs.index(max(val_accs))
                best_val_loss_epoch = val_losses.index(min(val_losses))
                
                table_data = [
                    ['Metric', 'Value', 'Epoch'],
                    ['Current Train Loss', f'{train_losses[-1]:.4f}', last_epoch + 1],
                    ['Current Val Loss', f'{val_losses[-1]:.4f}', last_epoch + 1],
                    ['Current Train Acc', f'{train_accs[-1]:.4f}', last_epoch + 1],
                    ['Current Val Acc', f'{val_accs[-1]:.4f}', last_epoch + 1],
                    ['Best Val Acc', f'{max(val_accs):.4f}', best_val_acc_epoch + 1],
                    ['Best Val Loss', f'{min(val_losses):.4f}', best_val_loss_epoch + 1],
                ]
                
                ax3.table(cellText=table_data, loc='center', cellLoc='center')
                ax3.set_title('Training Metrics Summary')
                
                plt.tight_layout()
                plt.show()
                
                # Print current status
                print(f"Epoch {last_epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f} - "
                      f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")
                
                if last_epoch > 0:
                    if val_losses[-1] < val_losses[-2]:
                        print(f"✅ Validation loss improved from {val_losses[-2]:.4f} to {val_losses[-1]:.4f}")
                    else:
                        print(f"⚠️ Validation loss did not improve from {val_losses[-2]:.4f}")
        except Exception as e:
            print(f"Could not initialize Colab visualization: {e}")
            display_progress = None
    else:
        display_progress = None
    
    # Detect hardware and optimize accordingly
    device = Config.device
    
    # Check if CUDA is available and optimize for GPU
    if device.type == 'cuda':
        print(f"\n🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"Optimizing for GPU training...")
        
        # Enable cuDNN benchmark for faster training
        torch.backends.cudnn.benchmark = True
        
        # Use mixed precision training if available (PyTorch >= 1.6)
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            use_amp = True
            print("✅ Using mixed precision training (faster GPU training)")
        except ImportError:
            use_amp = False
            print("⚠️ Mixed precision training not available (PyTorch < 1.6)")
    else:
        print("\n💻 Training on CPU")
        print("Optimizing for CPU training...")
        use_amp = False
        
        # For CPU: Set number of threads for better performance
        import multiprocessing
        num_workers = min(multiprocessing.cpu_count(), 8)  # Use at most 8 workers
        torch.set_num_threads(num_workers)
        print(f"✅ Using {num_workers} CPU threads for computation")
    
    # Move model to device
    model = model.to(device)
    
    # Update report with training parameters
    training_params = {
        "model_name": Config.model_name,
        "optimizer": Config.optimizer_name,
        "learning_rate": Config.learning_rate,
        "batch_size": Config.batch_size,
        "epochs": Config.num_epochs,
        "early_stopping_patience": Config.early_stopping_patience,
        "lr_scheduler": f"ReduceLROnPlateau (patience={Config.scheduler_patience})",
        "loss": "CrossEntropyLoss",
        "pretrained": Config.pretrained,
        "fine_tuned": True,
        "num_classes": Config.num_classes,
        "augmentation": {
            "rotation_range": "15°",
            "horizontal_flip": "True",
            "vertical_flip": "True",
            "color_jitter": "brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1"
        }
    }
    report_generator.update_training_params(training_params)
    
    # Initialize lists to track metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    print("\n🏋️ Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, labels in train_pbar:
            # Move data to device (non-blocking for faster data transfer on GPU)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Zero the parameter gradients - use set_to_none for better performance
            optimizer.zero_grad(set_to_none=True)
            
            # Use mixed precision for faster GPU training if available
            if use_amp and device.type == 'cuda':
                with autocast():
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Standard backward pass
                loss.backward()
                optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar (less frequently for better performance)
            if train_pbar.n % 5 == 0 or train_pbar.n == len(train_loader) - 1:
                train_pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100 * correct / total
                })
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # No gradients needed for validation
        with torch.no_grad():
            # Progress bar for validation
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            
            for inputs, labels in val_pbar:
                # Move data to device (non-blocking for faster data transfer on GPU)
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Use mixed precision for faster inference if available
                if use_amp and device.type == 'cuda':
                    with autocast():
                        # Forward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    # Standard forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar (less frequently for better performance)
                if val_pbar.n % 5 == 0 or val_pbar.n == len(val_loader) - 1:
                    val_pbar.set_postfix({
                        'loss': loss.item(),
                        'acc': 100 * correct / total
                    })
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Print epoch summary or display progress in Colab
        if display_progress:
            display_progress(train_losses, val_losses, train_accs, val_accs)
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
                  f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # Learning rate scheduler step
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()
        
        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"\u2728 New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{Config.early_stopping_patience}")
            
            if patience_counter >= Config.early_stopping_patience:
                print(f"\n⛔ Early stopping triggered after {epoch+1} epochs")
                break
    
    # Training complete
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    
    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("Loaded best model state from validation")
    
    # Save the model
    os.makedirs(Config.output_dir, exist_ok=True)
    model_path = os.path.join(Config.output_dir, f"{Config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    torch.save({
        'model_name': Config.model_name,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_classes': Config.num_classes,
        'class_names': Config.class_names,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Update report with training metrics
    training_metrics = {
        "best_train_acc": max(train_accs),
        "best_val_acc": max(val_accs),
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1]
    }
    
    # Create history dict for plotting
    history = {
        "accuracy": train_accs,
        "val_accuracy": val_accs,
        "loss": train_losses,
        "val_loss": val_losses
    }
    
    report_generator.update_training_metrics(training_metrics, history)
    
    return model, train_losses, val_losses, train_accs, val_accs

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
    # Initialize report generator
    report_generator = ReportGenerator(PROJECT_ROOT)
    
    device = Config.device
    model = model.to(device)
    model.eval()
    
    # Initialize variables
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    # No gradients needed for evaluation
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Statistics
            running_corrects += torch.sum(preds == labels.data)
            
            # Store predictions and labels for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = running_corrects.double() / len(test_loader.dataset)
    print(f"\n📈 Test Accuracy: {accuracy:.4f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Generate classification report
    print("\n📋 Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=Config.class_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=Config.class_names))
    
    # Calculate precision, recall, and f1 score (weighted)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    
    # Print additional metrics
    print(f"📈 Precision (weighted): {precision:.4f}")
    print(f"📈 Recall (weighted): {recall:.4f}")
    print(f"📈 F1 Score (weighted): {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=Config.class_names, yticklabels=Config.class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save confusion matrix plot
    os.makedirs(Config.plots_dir, exist_ok=True)
    cm_path = os.path.join(Config.plots_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save results to CSV
    results = {
        'model_name': Config.model_name,
        'accuracy': accuracy.item(),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(Config.class_names):
        results[f"{class_name}_precision"] = report[class_name]['precision']
        results[f"{class_name}_recall"] = report[class_name]['recall']
        results[f"{class_name}_f1"] = report[class_name]['f1-score']
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    os.makedirs(Config.results_dir, exist_ok=True)
    results_path = os.path.join(Config.results_dir, f"{Config.model_name}_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Update report with evaluation metrics
    evaluation_metrics = {
        "test_accuracy": accuracy.item(),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "class_metrics": report
    }
    report_generator.update_evaluation_metrics(evaluation_metrics, cm, Config.class_names)
    
    return accuracy.item(), cm, report

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

# Function to prompt user for model selection
def select_model():
    # Check if model name was passed as an environment variable
    model_name = os.environ.get("EMBRYO_MODEL_SELECTION", "")
    
    # If no environment variable is set, prompt the user
    if not model_name:
        print("\n🔍 Please select a model architecture:")
        print("  1. ResNet152 (default)")
        print("  2. DenseNet201")
        print("  3. EfficientNet-B7")
        print("  4. ConvNeXt Base")
        print("  5. SwinV2")
        print("  6. EfficientViT")
        
        choice = input("\nEnter your choice (1-6) or press Enter for default: ")
        
        # Map choice to model name
        model_mapping = {
            "1": "resnet152",
            "2": "densenet201",
            "3": "efficientnet_b7",
            "4": "convnext_base",
            "5": "swinv2",
            "6": "efficientvit",
            "": "resnet152"  # Default
        }
        
        model_name = model_mapping.get(choice, "resnet152")
    
    return model_name

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
    
    # Select model architecture
    Config.model_name = select_model()
        
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
    model, train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, Config.num_epochs)
    
    # Plot training results
    plot_training_results(train_losses, val_losses, train_accs, val_accs)
    
    # Evaluate the model on test set
    accuracy, cm, report = evaluate_model(model, test_loader)
    
    # Extract metrics from the report
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    
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
