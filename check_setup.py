"""
Quick verification script to check if everything is ready for training
"""

import os
import sys
from pathlib import Path

def check_setup():
    """Check if the project is ready for training/running"""
    
    print("=" * 60)
    print("🔍 Plant Disease Detection - System Check")
    print("=" * 60)
    print()
    
    all_good = True
    
    # Check 1: Dataset
    print("1. Checking dataset...")
    data_dir = Path('data/train')
    if data_dir.exists():
        classes = list(data_dir.iterdir())
        num_classes = len(classes)
        print(f"   ✅ Dataset found: {num_classes} disease classes")
        
        # Count total images
        total_images = sum(len(list(cls.glob('*.jpg')) + list(cls.glob('*.JPG')) + list(cls.glob('*.png'))) 
                          for cls in classes if cls.is_dir())
        print(f"   ✅ Total images: {total_images:,}")
    else:
        print(f"   ❌ Dataset NOT found at: {data_dir.absolute()}")
        print(f"   Please download dataset and place in: data/train/")
        all_good = False
    print()
    
    # Check 2: Required packages
    print("2. Checking required packages...")
    required_packages = [
        ('tensorflow', 'TensorFlow'),
        ('streamlit', 'Streamlit'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
        ('matplotlib', 'Matplotlib'),
        ('plotly', 'Plotly')
    ]
    
    missing_packages = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {name} installed")
        except ImportError:
            print(f"   ❌ {name} NOT installed")
            missing_packages.append(name)
            all_good = False
    print()
    
    # Check 3: Models directory
    print("3. Checking models directory...")
    models_dir = Path('models')
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created models directory")
    else:
        # Check if any trained models exist
        h5_files = list(models_dir.glob('*.h5'))
        if h5_files:
            print(f"   ✅ Found {len(h5_files)} trained model(s):")
            for h5_file in h5_files:
                size_mb = h5_file.stat().st_size / (1024*1024)
                print(f"      - {h5_file.name} ({size_mb:.1f} MB)")
        else:
            print(f"   ℹ️  No trained models found (train one with: python train.py)")
    print()
    
    # Check 4: Logs directory
    print("4. Checking logs directory...")
    logs_dir = Path('logs')
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created logs directory")
    else:
        print(f"   ✅ Logs directory exists")
    print()
    
    # Summary
    print("=" * 60)
    if all_good:
        print("✅ All checks passed! You're ready to go!")
        print()
        print("Next steps:")
        print("  1. Train a model: python train.py --model efficientnetb0 --epochs 50")
        print("  2. Or run the app: streamlit run app_advanced.py")
    else:
        print("⚠️  Some issues found. Please fix them before proceeding.")
        if missing_packages:
            print()
            print("Install missing packages:")
            print("  pip install -r requirements_full.txt")
    print("=" * 60)
    
    return all_good


if __name__ == "__main__":
    try:
        success = check_setup()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error during check: {str(e)}")
        sys.exit(1)
