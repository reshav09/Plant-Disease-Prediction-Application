"""
Utility Functions
Helper functions for image preprocessing, visualization, and predictions.
"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for model prediction.
    
    Args:
        image_path: Path to the image file or uploaded file object
        target_size: Target size to resize the image
        
    Returns:
        Preprocessed image array and original image
    """
    # Load image
    if isinstance(image_path, str):
        img = Image.open(image_path)
    else:
        img = Image.open(image_path)
    
    # Save original for display
    original_img = img.copy()
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image
    img = img.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, original_img


def predict_disease(model, image, class_names):
    """
    Predict disease from an image.
    
    Args:
        model: Trained Keras model
        image: Preprocessed image array
        class_names: List of class names
        
    Returns:
        Dictionary with prediction results
    """
    # Make prediction
    predictions = model.predict(image, verbose=0)
    
    # Get top prediction
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx])
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        {
            'class': class_names[idx],
            'confidence': float(predictions[0][idx])
        }
        for idx in top_3_idx
    ]
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top_3_predictions': top_3_predictions,
        'all_predictions': predictions[0].tolist()
    }


def load_model_and_classes(model_path, classes_path):
    """
    Load trained model and class names.
    
    Args:
        model_path: Path to saved model file
        classes_path: Path to classes JSON file
        
    Returns:
        model, class_names
    """
    import json
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # Load class names
    with open(classes_path, 'r') as f:
        class_indices = json.load(f)
    
    # Convert to list (assuming indices are 0, 1, 2, ...)
    class_names = [class_indices[str(i)] for i in range(len(class_indices))]
    
    return model, class_names


def create_gradcam_heatmap(model, image, last_conv_layer_name='top_conv'):
    """
    Create Grad-CAM heatmap to visualize which parts of the image 
    the model focuses on for prediction.
    
    Args:
        model: Trained model
        image: Preprocessed image
        last_conv_layer_name: Name of last convolutional layer
        
    Returns:
        Heatmap array
    """
    try:
        # Create a model that maps the input image to the activations
        grad_model = keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            predicted_class = tf.argmax(predictions[0])
            class_channel = predictions[:, predicted_class]
        
        # Extract gradients
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    except Exception as e:
        print(f"Could not generate Grad-CAM: {e}")
        return None


def overlay_heatmap_on_image(image, heatmap, alpha=0.4):
    """
    Overlay heatmap on original image.
    
    Args:
        image: Original image (PIL Image)
        heatmap: Heatmap array
        alpha: Transparency factor
        
    Returns:
        Overlay image
    """
    # Resize heatmap to match image
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Overlay
    overlay = cv2.addWeighted(image_array, 1 - alpha, heatmap, alpha, 0)
    
    return Image.fromarray(overlay)


def get_confidence_color(confidence):
    """
    Get color based on confidence level.
    
    Args:
        confidence: Confidence value (0-1)
        
    Returns:
        Color string for Streamlit
    """
    if confidence >= 0.9:
        return "green"
    elif confidence >= 0.7:
        return "orange"
    else:
        return "red"


def format_percentage(value):
    """Format a value as percentage."""
    return f"{value * 100:.2f}%"


def parse_class_name(class_name):
    """
    Parse class name to extract crop and disease information.
    
    Args:
        class_name: Full class name (e.g., 'Tomato___Early_blight')
        
    Returns:
        Dictionary with crop and disease
    """
    parts = class_name.split('___')
    if len(parts) == 2:
        crop = parts[0].replace('_', ' ')
        disease = parts[1].replace('_', ' ')
        return {'crop': crop, 'disease': disease}
    else:
        return {'crop': 'Unknown', 'disease': class_name}


def check_image_quality(image):
    """
    Check if uploaded image meets quality requirements.
    
    Args:
        image: PIL Image
        
    Returns:
        tuple: (is_valid, message)
    """
    # Check image size
    if image.width < 50 or image.height < 50:
        return False, "Image is too small. Please upload a larger image (at least 50x50 pixels)."
    
    # Check image mode
    if image.mode not in ['RGB', 'RGBA', 'L']:
        return False, "Unsupported image format. Please upload RGB images."
    
    # Check file size (assuming we can check this)
    # This is a basic check, actual implementation may vary
    
    return True, "Image quality is acceptable."


def calculate_image_stats(image_array):
    """
    Calculate basic statistics of the image.
    
    Args:
        image_array: Numpy array of the image
        
    Returns:
        Dictionary with statistics
    """
    return {
        'mean': float(np.mean(image_array)),
        'std': float(np.std(image_array)),
        'min': float(np.min(image_array)),
        'max': float(np.max(image_array)),
        'shape': image_array.shape
    }


def create_prediction_report(prediction_result, disease_info):
    """
    Create a formatted prediction report.
    
    Args:
        prediction_result: Dictionary with prediction results
        disease_info: Dictionary with disease information
        
    Returns:
        Formatted report string
    """
    report = f"""
    🔍 PREDICTION REPORT
    {'=' * 50}
    
    Crop: {disease_info.get('crop', 'Unknown')}
    Disease: {disease_info.get('disease', 'Unknown')}
    Confidence: {format_percentage(prediction_result['confidence'])}
    Severity: {disease_info.get('severity', 'Unknown')}
    
    📋 DESCRIPTION
    {disease_info.get('description', 'No description available.')}
    
    🔴 SYMPTOMS
    {chr(10).join('• ' + s for s in disease_info.get('symptoms', ['No symptoms listed']))}
    
    🛡️ PREVENTION
    {chr(10).join('• ' + p for p in disease_info.get('prevention', ['No prevention measures listed']))}
    
    💊 TREATMENT
    {chr(10).join('• ' + t for t in disease_info.get('treatment', ['No treatment information available']))}
    """
    
    return report


def save_prediction_history(prediction_data, history_file='prediction_history.json'):
    """
    Save prediction to history file.
    
    Args:
        prediction_data: Dictionary with prediction information
        history_file: Path to history file
    """
    import json
    from datetime import datetime
    
    # Add timestamp
    prediction_data['timestamp'] = datetime.now().isoformat()
    
    # Load existing history
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except:
            history = []
    
    # Add new prediction
    history.append(prediction_data)
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test class name parsing
    test_class = "Tomato___Early_blight"
    parsed = parse_class_name(test_class)
    print(f"\nParsed class name:")
    print(f"  Crop: {parsed['crop']}")
    print(f"  Disease: {parsed['disease']}")
    
    # Test confidence color
    print(f"\nConfidence colors:")
    print(f"  0.95: {get_confidence_color(0.95)}")
    print(f"  0.75: {get_confidence_color(0.75)}")
    print(f"  0.50: {get_confidence_color(0.50)}")
    
    print("\n✅ Utility functions loaded successfully!")
