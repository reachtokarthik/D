import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB7, ResNet152, DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os.path as path
import pandas as pd

# Import the report generator
from src.report_generator import ReportGenerator

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
    logs_dir = path.join(PROJECT_ROOT, "outputs", "logs")
    
    # Model parameters
    model_name = "efficientnet_b7"  # Options: resnet152, densenet201, efficientnet_b7
    
    # Training parameters
    batch_size = 32
    img_size = (224, 224)
    epochs = 30
    learning_rate = 1e-4
    early_stopping_patience = 7
    
    # Random seed for reproducibility
    seed = 42

# Set random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Create data generators
def create_data_generators():
    # Initialize report generator
    report_generator = ReportGenerator(PROJECT_ROOT)
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        Config.train_dir,
        target_size=Config.img_size,
        batch_size=Config.batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_test_datagen.flow_from_directory(
        Config.val_dir,
        target_size=Config.img_size,
        batch_size=Config.batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        Config.test_dir,
        target_size=Config.img_size,
        batch_size=Config.batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Print dataset information
    print(f"\n📊 Dataset loaded:")
    print(f"   - Number of classes: {len(train_generator.class_indices)}")
    print(f"   - Class mapping: {train_generator.class_indices}")
    print(f"   - Training samples: {train_generator.samples}")
    print(f"   - Validation samples: {validation_generator.samples}")
    print(f"   - Test samples: {test_generator.samples}")
    
    # Update report with class distribution
    class_counts = {}
    for class_name, idx in train_generator.class_indices.items():
        # Count samples in each class
        class_count = len([1 for _, label in train_generator.filenames if label == idx])
        class_counts[class_name] = class_count
    
    # Update normalization details in report
    norm_params = {
        "rescaling": "1/255",
        "image_size": f"{Config.img_size[0]}x{Config.img_size[1]}",
        "mean": "ImageNet mean",
        "std": "ImageNet std"
    }
    report_generator.update_normalization(norm_params)
    
    # Update class distribution in report
    report_generator.update_class_distribution(class_counts)
    
    # Update split information in report
    split_info = {
        "train": train_generator.samples,
        "validation": validation_generator.samples,
        "test": test_generator.samples,
        "strategy": "Stratified split",
        "seed": Config.seed
    }
    report_generator.update_split_info(split_info)
    
    return train_generator, validation_generator, test_generator

# Build model
def build_model(model_name, num_classes):
    input_shape = (*Config.img_size, 3)
    
    if model_name == "efficientnet_b7":
        base_model = EfficientNetB7(include_top=False, input_shape=input_shape, weights='imagenet')
    elif model_name == "resnet152":
        base_model = ResNet152(include_top=False, input_shape=input_shape, weights='imagenet')
    elif model_name == "densenet201":
        base_model = DenseNet201(include_top=False, input_shape=input_shape, weights='imagenet')
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)  # Add dropout for regularization
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model, base_model

# Train model
def train_model(model, train_generator, validation_generator, base_model=None):
    # Initialize report generator
    report_generator = ReportGenerator(PROJECT_ROOT)
    
    # Create output directories
    os.makedirs(Config.output_dir, exist_ok=True)
    os.makedirs(Config.plots_dir, exist_ok=True)
    os.makedirs(Config.results_dir, exist_ok=True)
    os.makedirs(Config.logs_dir, exist_ok=True)
    
    # Prepare model checkpoint and callbacks
    checkpoint_path = os.path.join(Config.output_dir, f"{Config.model_name}_best.h5")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=Config.early_stopping_patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-6),
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        TensorBoard(log_dir=Config.logs_dir)
    ]
    
    # Compile model
    optimizer = AdamW(learning_rate=Config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Update report with training parameters
    training_params = {
        "model_name": Config.model_name,
        "optimizer": "AdamW",
        "learning_rate": Config.learning_rate,
        "batch_size": Config.batch_size,
        "epochs": Config.epochs,
        "early_stopping_patience": Config.early_stopping_patience,
        "lr_scheduler": "ReduceLROnPlateau (factor=0.1, patience=3)",
        "loss": "categorical_crossentropy",
        "pretrained": True,
        "fine_tuned": True if base_model else False,
        "num_classes": len(train_generator.class_indices),
        "augmentation": {
            "rotation_range": "15°",
            "width_shift": "0.1",
            "height_shift": "0.1",
            "shear_range": "0.1",
            "zoom_range": "0.1",
            "horizontal_flip": "True",
            "vertical_flip": "True",
            "fill_mode": "nearest"
        }
    }
    report_generator.update_training_params(training_params)
    
    print("\n🔍 Model summary:")
    model.summary()
    
    print("\n🏋️ Training model with frozen layers...")
    # First phase: train only the top layers
    history1 = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // Config.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // Config.batch_size,
        epochs=5,  # Train for a few epochs with frozen base model
        callbacks=callbacks,
        verbose=1
    )
    
    # Second phase: fine-tune the model
    if base_model:
        print("\n🔄 Fine-tuning model...")
        # Unfreeze the base model
        for layer in base_model.layers:
            layer.trainable = True
            
        # Recompile with a lower learning rate
        optimizer = AdamW(learning_rate=Config.learning_rate / 10)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Continue training
        history2 = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // Config.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // Config.batch_size,
            epochs=Config.epochs,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=len(history1.history['loss'])
        )
        
        # Combine histories
        history = {}
        for key in history1.history.keys():
            history[key] = history1.history[key] + history2.history[key]
    else:
        # Train for full epochs
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // Config.batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // Config.batch_size,
            epochs=Config.epochs,
            callbacks=callbacks,
            verbose=1
        ).history
    
    # Save the final model
    final_model_path = os.path.join(Config.output_dir, f"{Config.model_name}_final.h5")
    model.save(final_model_path)
    print(f"\n💾 Model saved to {final_model_path}")
    
    # Update report with training metrics
    training_metrics = {
        "best_train_acc": max(history['accuracy']),
        "best_val_acc": max(history['val_accuracy']),
        "final_train_loss": history['loss'][-1],
        "final_val_loss": history['val_loss'][-1]
    }
    report_generator.update_training_metrics(training_metrics, history)
    
    return model, history

# Plot training history
def plot_training_history(history):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['accuracy'], label='Training Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
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

# Evaluate model
def evaluate_model(model, test_generator):
    # Initialize report generator
    report_generator = ReportGenerator(PROJECT_ROOT)
    
    # Get predictions
    print("\n📊 Evaluating model on test set...")
    y_pred_prob = model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Calculate metrics
    accuracy = np.mean(y_pred == y_true)
    cm = confusion_matrix(y_true, y_pred)
    class_names = list(test_generator.class_indices.keys())
    
    # Calculate precision, recall, and f1 score (weighted)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    precision = report_dict['weighted avg']['precision']
    recall = report_dict['weighted avg']['recall']
    f1 = report_dict['weighted avg']['f1-score']
    
    # Print evaluation results
    print(f"\n📈 Test Accuracy: {accuracy:.4f}")
    print(f"📈 Precision (weighted): {precision:.4f}")
    print(f"📈 Recall (weighted): {recall:.4f}")
    print(f"📈 F1 Score (weighted): {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save confusion matrix plot
    cm_path = os.path.join(Config.plots_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Save classification report
    report_path = os.path.join(Config.results_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Create results dictionary for saving
    results = {
        'model_name': Config.model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add per-class metrics
    for class_name in class_names:
        results[f"{class_name}_precision"] = report_dict[class_name]['precision']
        results[f"{class_name}_recall"] = report_dict[class_name]['recall']
        results[f"{class_name}_f1"] = report_dict[class_name]['f1-score']
    
    # Update report with evaluation metrics
    evaluation_metrics = {
        "test_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "class_metrics": report_dict
    }
    report_generator.update_evaluation_metrics(evaluation_metrics, cm, class_names)
    
    return accuracy, cm, report_dict
    
    # Create or append to results CSV
    results_file = os.path.join(Config.results_dir, "model_results.csv")
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
    else:
        results_df = pd.DataFrame([results])
    
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    return report['accuracy'], cm, report

# Main function
def main():
    # Set random seed
    set_seed(Config.seed)
    
    print(f"\n🔍 Starting embryo quality prediction model training")
    print(f"   - Model: {Config.model_name}")
    print(f"   - Batch size: {Config.batch_size}")
    print(f"   - Image size: {Config.img_size}")
    print(f"   - Learning rate: {Config.learning_rate}")
    
    # Create data generators
    train_generator, validation_generator, test_generator = create_data_generators()
    
    # Build model
    model, base_model = build_model(Config.model_name, len(train_generator.class_indices))
    
    # Train model
    model, history = train_model(model, train_generator, validation_generator, base_model)
    
    # Evaluate model
    accuracy, _, _ = evaluate_model(model, test_generator)
    
    # Save final model
    final_model_path = os.path.join(Config.output_dir, f"{Config.model_name}_final.h5")
    model.save(final_model_path)
    
    print(f"\n✅ Model training and evaluation complete!")
    print(f"   - Final test accuracy: {accuracy:.4f}")
    print(f"   - Model saved to {final_model_path}")

# Run the script
if __name__ == "__main__":
    main()
