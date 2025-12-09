"""
GAN Training Script for Dataset Augmentation
Trains a Conditional GAN to generate synthetic plant disease images for balancing the dataset.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from datetime import datetime

from gan import ConditionalGAN, DatasetBalancer, preprocess_for_gan


def load_plantvillage_data(data_dir, img_size=(224, 224)):
    """
    Load PlantVillage dataset for GAN training.
    
    Args:
        data_dir: Path to dataset directory
        img_size: Target image size
        
    Returns:
        X_train, y_train, class_names
    """
    print(f"\n📂 Loading dataset from: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Dataset directory not found: {data_dir}")
        print("   Please download the dataset first using: python download_dataset.py")
        sys.exit(1)
    
    # Use keras image_dataset_from_directory
    train_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=None,  # Load all at once for GAN training
        shuffle=True,
        seed=42
    )
    
    class_names = train_ds.class_names
    print(f"✅ Found {len(class_names)} classes")
    
    # Convert to numpy arrays
    X_train = []
    y_train = []
    
    print("🔄 Converting to numpy arrays...")
    for image, label in train_ds:
        X_train.append(image.numpy())
        y_train.append(label.numpy())
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"✅ Dataset loaded: {X_train.shape[0]} images")
    
    return X_train, y_train, class_names


def main():
    parser = argparse.ArgumentParser(description='Train GAN for plant disease image generation')
    parser.add_argument('--data_dir', type=str, default='data/train',
                       help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, default='gan_output',
                       help='Output directory for generated images and models')
    parser.add_argument('--epochs', type=int, default=10000,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--sample_interval', type=int, default=500,
                       help='Interval for saving sample images')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size (square)')
    parser.add_argument('--latent_dim', type=int, default=100,
                       help='Latent dimension for noise vector')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze class distribution without training')
    
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║       GAN Training for Plant Disease Dataset Balancing        ║
    ║  Conditional GAN for Synthetic Diseased Leaf Image Generation ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✅ GPU Available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"   {gpu}")
    else:
        print("\n⚠️  No GPU detected. Training will be slow on CPU.")
        print("   Consider using Google Colab or a GPU-enabled environment.")
    
    # Load dataset
    try:
        X_train, y_train, class_names = load_plantvillage_data(
            args.data_dir,
            img_size=(args.img_size, args.img_size)
        )
    except Exception as e:
        print(f"\n❌ Error loading dataset: {str(e)}")
        sys.exit(1)
    
    # Analyze class distribution
    print("\n" + "="*70)
    stats = DatasetBalancer.analyze_class_distribution(y_train, class_names)
    print("="*70)
    
    # Save distribution stats
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'class_distribution.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    if args.analyze_only:
        print("\n✅ Analysis complete. Exiting (--analyze_only flag set)")
        return
    
    # Preprocess images for GAN
    print("\n🔄 Preprocessing images for GAN training...")
    X_train_gan = preprocess_for_gan(X_train)
    print(f"✅ Images preprocessed to range [{X_train_gan.min():.2f}, {X_train_gan.max():.2f}]")
    
    # Initialize GAN
    print(f"\n🤖 Initializing Conditional GAN...")
    print(f"   Image shape: {args.img_size}x{args.img_size}x3")
    print(f"   Number of classes: {len(class_names)}")
    print(f"   Latent dimension: {args.latent_dim}")
    
    gan = ConditionalGAN(
        img_shape=(args.img_size, args.img_size, 3),
        num_classes=len(class_names),
        latent_dim=args.latent_dim
    )
    
    # Print architectures
    print("\n📐 Generator Architecture:")
    gan.generator.summary()
    
    print("\n📐 Discriminator Architecture:")
    gan.discriminator.summary()
    
    # Train GAN
    print(f"\n🚀 Starting GAN training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Sample interval: {args.sample_interval}")
    print(f"   Output directory: {args.output_dir}")
    print("\n" + "="*70)
    
    start_time = datetime.now()
    
    try:
        history = gan.train(
            X_train_gan,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sample_interval=args.sample_interval,
            save_dir=args.output_dir
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("   Saving models...")
        gan.generator.save(os.path.join(args.output_dir, 'models', 'generator_interrupted.h5'))
        gan.discriminator.save(os.path.join(args.output_dir, 'models', 'discriminator_interrupted.h5'))
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print(f"✅ Training completed!")
    print(f"   Duration: {duration}")
    print(f"   Models saved to: {args.output_dir}/models/")
    print(f"   Samples saved to: {args.output_dir}/samples/")
    print("="*70)
    
    # Test generation for a few classes
    print("\n🎨 Generating test images for verification...")
    test_classes = np.random.choice(len(class_names), min(5, len(class_names)), replace=False)
    
    for class_idx in test_classes:
        print(f"   Generating 10 images for class {class_idx}: {class_names[class_idx]}")
        gen_imgs = gan.generate_images(class_idx, num_images=10)
        
        # Save a sample
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(gen_imgs[i])
            ax.axis('off')
        plt.suptitle(f'Generated Images - Class: {class_names[class_idx]}')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'test_generation_class_{class_idx}.png'))
        plt.close()
    
    print(f"\n✅ Test images saved to {args.output_dir}/")
    
    # Instructions for using generated images
    print("\n" + "="*70)
    print("📚 Next Steps:")
    print("="*70)
    print("\n1. Balance your dataset:")
    print("   from gan import ConditionalGAN, DatasetBalancer")
    print("   gan = keras.models.load_model('gan_output/models/generator_final.h5')")
    print("   X_balanced, y_balanced = DatasetBalancer.balance_dataset(X_train, y_train, gan)")
    
    print("\n2. Retrain your classifier with balanced data:")
    print("   python train.py --use_gan_augmentation --gan_model gan_output/models/generator_final.h5")
    
    print("\n3. Evaluate improvement:")
    print("   Compare metrics before and after GAN augmentation")
    print("="*70)


if __name__ == "__main__":
    main()
