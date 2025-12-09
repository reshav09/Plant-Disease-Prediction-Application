"""
Training Script for Plant Disease Detection Model
Handles data loading, augmentation, training, and evaluation.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
from datetime import datetime
import json

from model import get_model


def setup_data_generators(data_dir, img_size=224, batch_size=32, validation_split=0.2):
    """
    Setup data generators with augmentation for training and validation.
    
    Args:
        data_dir: Path to dataset directory
        img_size: Size to resize images
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        
    Returns:
        train_generator, val_generator, class_names
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )
    
    # Validation data (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    class_names = list(train_generator.class_indices.keys())
    
    print(f"\n📊 Data Setup Complete:")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print(f"   Number of classes: {len(class_names)}")
    print(f"   Batch size: {batch_size}")
    
    return train_generator, val_generator, class_names


def setup_callbacks(model_name, log_dir='logs'):
    """
    Setup training callbacks.
    
    Args:
        model_name: Name of the model
        log_dir: Directory for logs
        
    Returns:
        List of callbacks
    """
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=f'{log_dir}/{model_name}_{timestamp}',
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    return callbacks


def plot_training_history(history, model_name):
    """
    Plot training and validation accuracy/loss.
    
    Args:
        history: Training history object
        model_name: Name of the model
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title(f'{model_name} - Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title(f'{model_name} - Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'models/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 Training history plot saved to models/{model_name}_training_history.png")
    plt.close()


def save_class_indices(class_names, model_name):
    """
    Save class names to JSON file.
    
    Args:
        class_names: List of class names
        model_name: Name of the model
    """
    class_indices = {i: name for i, name in enumerate(class_names)}
    
    with open(f'models/{model_name}_classes.json', 'w') as f:
        json.dump(class_indices, f, indent=4)
    
    print(f"💾 Class indices saved to models/{model_name}_classes.json")


def evaluate_model(model, val_generator):
    """
    Evaluate model on validation data.
    
    Args:
        model: Trained model
        val_generator: Validation data generator
    """
    print("\n📊 Evaluating model...")
    
    results = model.evaluate(val_generator, verbose=1)
    
    print("\n✅ Evaluation Results:")
    print(f"   Validation Loss: {results[0]:.4f}")
    print(f"   Validation Accuracy: {results[1]:.4f}")
    if len(results) > 2:
        print(f"   Top-3 Accuracy: {results[2]:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Plant Disease Detection Model')
    parser.add_argument('--data_dir', type=str, default='data/test',
                        help='Path to dataset directory')
    parser.add_argument('--model', type=str, default='efficientnetb0',
                      choices=['custom_cnn', 'resnet50', 'efficientnetb0', 'mobilenetv2'],
                      help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=25,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=224,
                      help='Image size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--validation_split', type=float, default=0.2,
                      help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Dataset directory '{args.data_dir}' not found!")
        print("Please download the dataset first using 'python download_dataset.py'")
        return
    
    print("=" * 70)
    print("🌾 PLANT DISEASE DETECTION - MODEL TRAINING")
    print("=" * 70)
    print(f"\n📝 Training Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Image size: {args.img_size}x{args.img_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Validation split: {args.validation_split}")
    
    # Setup data generators
    train_gen, val_gen, class_names = setup_data_generators(
        args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )
    
    # Create model
    num_classes = len(class_names)
    model = get_model(
        model_name=args.model,
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=num_classes
    )
    
    # Print model summary
    print("\n📋 Model Summary:")
    model.summary()
    
    # Setup callbacks
    callbacks = setup_callbacks(args.model)
    
    # Train model
    print("\n" + "=" * 70)
    print("🚀 Starting Training...")
    print("=" * 70)
    
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    evaluate_model(model, val_gen)
    
    # Plot training history
    plot_training_history(history, args.model)
    
    # Save class indices
    save_class_indices(class_names, args.model)
    
    # Save final model
    model.save(f'models/{args.model}_final.h5')
    print(f"\n💾 Final model saved to models/{args.model}_final.h5")
    
    print("\n" + "=" * 70)
    print("✅ Training completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
