"""
TensorFlow Lite Conversion Script
Converts trained models to TFLite format for mobile and edge deployment.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path


def convert_to_tflite(model_path, output_path=None, quantization='none'):
    """
    Convert Keras model to TensorFlow Lite format.
    
    Args:
        model_path: Path to saved .h5 model file
        output_path: Output path for .tflite file
        quantization: Quantization type ('none', 'dynamic', 'int8', 'float16')
        
    Returns:
        Path to converted model
    """
    print("=" * 70)
    print("🔄 TensorFlow Lite Conversion")
    print("=" * 70)
    
    # Load the model
    print(f"\n📁 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization based on quantization type
    if quantization == 'dynamic':
        print("\n⚙️  Applying dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    elif quantization == 'int8':
        print("\n⚙️  Applying int8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        
        # Representative dataset for int8 quantization
        def representative_dataset_gen():
            # Generate random samples (in real scenario, use actual data)
            for _ in range(100):
                data = np.random.rand(1, *model.input_shape[1:]).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
    elif quantization == 'float16':
        print("\n⚙️  Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    else:
        print("\n⚙️  No quantization applied (full precision)")
    
    # Convert
    print("\n🔄 Converting model...")
    try:
        tflite_model = converter.convert()
        print("✅ Conversion successful!")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None
    
    # Determine output path
    if output_path is None:
        base_name = Path(model_path).stem
        output_path = f"models/{base_name}_{quantization}.tflite"
    
    # Save the model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get file sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    tflite_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    compression_ratio = (1 - tflite_size / original_size) * 100
    
    print(f"\n💾 Model saved to: {output_path}")
    print(f"\n📊 Conversion Statistics:")
    print(f"   Original size: {original_size:.2f} MB")
    print(f"   TFLite size: {tflite_size:.2f} MB")
    print(f"   Size reduction: {compression_ratio:.1f}%")
    
    return output_path


def test_tflite_model(tflite_path, test_image_shape=(224, 224, 3)):
    """
    Test TFLite model with a random input.
    
    Args:
        tflite_path: Path to .tflite model
        test_image_shape: Shape of test image
    """
    print("\n🧪 Testing TFLite model...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    
    # Create test input
    test_image = np.random.rand(1, *test_image_shape).astype(np.float32)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_image)
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✅ Model inference successful!")
    print(f"   Output shape: {output.shape}")
    print(f"   Sample predictions: {output[0][:5]}")
    
    return True


def batch_convert_models(models_dir='models', quantization_types=['none', 'dynamic', 'float16']):
    """
    Convert all models in a directory to TFLite format.
    
    Args:
        models_dir: Directory containing .h5 models
        quantization_types: List of quantization types to apply
    """
    print("=" * 70)
    print("🔄 BATCH MODEL CONVERSION")
    print("=" * 70)
    
    models_path = Path(models_dir)
    h5_files = list(models_path.glob("*.h5"))
    
    if not h5_files:
        print(f"❌ No .h5 model files found in {models_dir}")
        return
    
    print(f"\n📁 Found {len(h5_files)} model(s) to convert")
    
    results = []
    
    for model_path in h5_files:
        print(f"\n{'=' * 70}")
        print(f"Converting: {model_path.name}")
        print("=" * 70)
        
        for quant_type in quantization_types:
            try:
                output_path = convert_to_tflite(
                    str(model_path),
                    quantization=quant_type
                )
                
                if output_path:
                    results.append({
                        'model': model_path.name,
                        'quantization': quant_type,
                        'output': output_path,
                        'status': 'success'
                    })
                else:
                    results.append({
                        'model': model_path.name,
                        'quantization': quant_type,
                        'status': 'failed'
                    })
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    'model': model_path.name,
                    'quantization': quant_type,
                    'status': 'error',
                    'error': str(e)
                })
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 CONVERSION SUMMARY")
    print("=" * 70)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n✅ Successful conversions: {success_count}/{len(results)}")
    
    for result in results:
        if result['status'] == 'success':
            print(f"   ✓ {result['model']} ({result['quantization']})")


def create_mobile_metadata(tflite_path, class_names, description="Plant Disease Detector"):
    """
    Create metadata file for mobile app.
    
    Args:
        tflite_path: Path to TFLite model
        class_names: List of class names
        description: Model description
    """
    import json
    
    metadata = {
        'model_name': Path(tflite_path).stem,
        'description': description,
        'input_shape': [1, 224, 224, 3],
        'output_shape': [1, len(class_names)],
        'classes': class_names,
        'preprocessing': {
            'normalization': 'divide_by_255',
            'resize': [224, 224]
        },
        'version': '1.0.0'
    }
    
    metadata_path = tflite_path.replace('.tflite', '_metadata.json')
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📝 Metadata saved to: {metadata_path}")
    
    return metadata_path


def main():
    parser = argparse.ArgumentParser(description='Convert Keras models to TensorFlow Lite')
    parser.add_argument('--model_path', type=str, default='models/efficientnetb0_best.h5',
                      help='Path to Keras model (.h5 file)')
    parser.add_argument('--output_path', type=str, default=None,
                      help='Output path for TFLite model')
    parser.add_argument('--quantization', type=str, default='dynamic',
                      choices=['none', 'dynamic', 'int8', 'float16'],
                      help='Quantization type')
    parser.add_argument('--batch', action='store_true',
                      help='Convert all models in models/ directory')
    parser.add_argument('--test', action='store_true',
                      help='Test the converted model')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch conversion
        batch_convert_models()
    else:
        # Single model conversion
        if not os.path.exists(args.model_path):
            print(f"❌ Error: Model file not found: {args.model_path}")
            print("Please train a model first using: python train.py")
            return
        
        # Convert model
        tflite_path = convert_to_tflite(
            args.model_path,
            args.output_path,
            args.quantization
        )
        
        # Test if requested
        if args.test and tflite_path:
            test_tflite_model(tflite_path)
        
        # Create metadata
        if tflite_path:
            print("\n📱 For mobile deployment:")
            print(f"   1. Copy {tflite_path} to your mobile app")
            print(f"   2. Use TensorFlow Lite API for inference")
            print(f"   3. Preprocess images: resize to 224x224, normalize to [0,1]")
            print(f"   4. Post-process: get argmax of output for predicted class")
    
    print("\n" + "=" * 70)
    print("✅ TensorFlow Lite conversion complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
