"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Explainable AI module for visualizing which parts of the leaf image 
the model focuses on for disease prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    Grad-CAM implementation for CNN visualization.
    Highlights the regions of the image that are most important for prediction.
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained Keras model
            layer_name: Name of the convolutional layer to visualize.
                       If None, uses the last convolutional layer.
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        
        # Create a model that outputs both the original prediction
        # and the activations of the target layer
        self.grad_model = keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(self.layer_name).output, model.output]
        )
    
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            # Check if it's a convolutional layer
            if len(layer.output_shape) == 4:  # Conv layers have 4D output
                return layer.name
        
        raise ValueError("Could not find a convolutional layer in the model")
    
    def compute_heatmap(self, image, pred_index=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap for an image.
        
        Args:
            image: Preprocessed image array (1, height, width, 3)
            pred_index: Index of the class to visualize. If None, uses predicted class.
            eps: Small value to prevent division by zero
            
        Returns:
            Heatmap array
        """
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Get the convolutional outputs and predictions
            conv_outputs, predictions = self.grad_model(image)
            
            # If pred_index is None, use the top predicted class
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # Extract the prediction for the target class
            class_channel = predictions[:, pred_index]
        
        # Compute gradients of the class output with respect to the feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by gradient importance and sum
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap between 0 and 1
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay the heatmap on the original image.
        
        Args:
            heatmap: Computed heatmap
            image: Original PIL Image or numpy array
            alpha: Transparency of overlay (0-1)
            colormap: OpenCV colormap to use
            
        Returns:
            PIL Image with overlay
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
        
        return Image.fromarray(overlay)
    
    def explain_prediction(self, image, original_image, pred_index=None):
        """
        Generate explanation for a prediction with visualization.
        
        Args:
            image: Preprocessed image array
            original_image: Original PIL Image
            pred_index: Class index to explain
            
        Returns:
            Dictionary with heatmap and overlay image
        """
        # Compute heatmap
        heatmap = self.compute_heatmap(image, pred_index)
        
        # Create overlay
        overlay_image = self.overlay_heatmap(heatmap, original_image)
        
        return {
            'heatmap': heatmap,
            'overlay': overlay_image,
            'layer_name': self.layer_name
        }


class GuidedGradCAM:
    """
    Guided Grad-CAM for more precise visualization.
    Combines Grad-CAM with Guided Backpropagation.
    """
    
    def __init__(self, model, layer_name=None):
        """Initialize Guided Grad-CAM."""
        self.model = model
        self.gradcam = GradCAM(model, layer_name)
    
    def compute_saliency(self, image, pred_index=None):
        """
        Compute saliency map using guided backpropagation.
        
        Args:
            image: Preprocessed image
            pred_index: Target class index
            
        Returns:
            Saliency map
        """
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            tape.watch(inputs)
            predictions = self.model(inputs)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, inputs)
        
        # Apply ReLU to get positive gradients only
        grads = tf.maximum(grads, 0)
        
        # Normalize
        grads = grads / (tf.reduce_max(grads) + 1e-8)
        
        return grads[0].numpy()
    
    def generate_visualization(self, image, original_image, pred_index=None):
        """
        Generate Guided Grad-CAM visualization.
        
        Args:
            image: Preprocessed image
            original_image: Original PIL Image
            pred_index: Target class
            
        Returns:
            Dictionary with visualizations
        """
        # Get Grad-CAM
        gradcam_result = self.gradcam.explain_prediction(image, original_image, pred_index)
        
        # Get saliency map
        saliency = self.compute_saliency(image, pred_index)
        
        # Combine Grad-CAM and saliency
        heatmap = gradcam_result['heatmap']
        heatmap_resized = cv2.resize(heatmap, (saliency.shape[1], saliency.shape[0]))
        
        # Element-wise multiplication
        guided_gradcam = heatmap_resized[..., np.newaxis] * saliency
        
        # Normalize
        guided_gradcam = guided_gradcam / (np.max(guided_gradcam) + 1e-8)
        
        return {
            'gradcam': gradcam_result,
            'saliency': saliency,
            'guided_gradcam': guided_gradcam
        }


def visualize_gradcam_grid(original_image, gradcam_overlay, heatmap, save_path=None):
    """
    Create a grid visualization of original, heatmap, and overlay.
    
    Args:
        original_image: Original PIL Image
        gradcam_overlay: Overlay image with Grad-CAM
        heatmap: Heatmap array
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(gradcam_overlay)
    axes[2].set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to {save_path}")
    
    return fig


def explain_top_predictions(model, image, original_image, class_names, top_k=3):
    """
    Explain top K predictions with Grad-CAM.
    
    Args:
        model: Trained model
        image: Preprocessed image
        original_image: Original PIL Image
        class_names: List of class names
        top_k: Number of top predictions to explain
        
    Returns:
        List of explanations for top K predictions
    """
    # Get predictions
    predictions = model.predict(image, verbose=0)[0]
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    # Create Grad-CAM
    gradcam = GradCAM(model)
    
    explanations = []
    
    for idx in top_indices:
        # Generate explanation
        explanation = gradcam.explain_prediction(image, original_image, pred_index=idx)
        
        explanations.append({
            'class_name': class_names[idx],
            'confidence': float(predictions[idx]),
            'class_index': int(idx),
            'heatmap': explanation['heatmap'],
            'overlay': explanation['overlay']
        })
    
    return explanations


if __name__ == "__main__":
    print("🔍 Grad-CAM Module - Explainable AI for Plant Disease Detection")
    print("=" * 70)
    print("\nThis module provides:")
    print("  ✓ Grad-CAM visualization")
    print("  ✓ Guided Grad-CAM")
    print("  ✓ Saliency maps")
    print("  ✓ Multi-class explanations")
    print("\nFeatures:")
    print("  • Highlights infected leaf regions")
    print("  • Shows model reasoning")
    print("  • Increases trust in AI predictions")
    print("  • Helps identify disease patterns")
