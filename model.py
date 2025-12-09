"""
Model Architecture Module
Contains implementations of different CNN architectures for plant disease classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import (
    ResNet50, ResNet101, ResNet152,
    EfficientNetB0, EfficientNetB3, EfficientNetB7,
    MobileNetV2, MobileNetV3Large,
    DenseNet121, DenseNet201,
    VGG16, VGG19,
    InceptionV3, InceptionResNetV2,
    Xception, NASNetMobile
)


def create_custom_cnn(input_shape=(224, 224, 3), num_classes=38):
    """
    Create a custom CNN model for plant disease classification.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of disease classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output Layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_resnet50_model(input_shape=(224, 224, 3), num_classes=38, trainable_layers=10):
    """
    Create a ResNet50-based transfer learning model.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of disease classes
        trainable_layers: Number of layers to unfreeze for fine-tuning
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained ResNet50 without top layers
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base model layers
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    # Add custom top layers
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=38, trainable_layers=20):
    """
    Create an EfficientNetB0-based transfer learning model.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of disease classes
        trainable_layers: Number of layers to unfreeze for fine-tuning
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained EfficientNetB0 without top layers
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base model layers
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    # Add custom top layers
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=38, trainable_layers=15):
    """
    Create a MobileNetV2-based transfer learning model.
    Lightweight and fast, ideal for deployment.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of disease classes
        trainable_layers: Number of layers to unfreeze for fine-tuning
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base model layers
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    # Add custom top layers
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=38, trainable_layers=15):
    """
    Create a MobileNetV2-based transfer learning model.
    Lightweight and fast, ideal for deployment.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of disease classes
        trainable_layers: Number of layers to unfreeze for fine-tuning
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base model layers
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    # Add custom top layers
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def create_densenet_model(input_shape=(224, 224, 3), num_classes=38, trainable_layers=20):
    """
    Create a DenseNet121-based transfer learning model.
    Dense connections enable better gradient flow.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of disease classes
        trainable_layers: Number of layers to unfreeze for fine-tuning
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained DenseNet121
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base model layers
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    # Add custom top layers
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def create_vgg16_model(input_shape=(224, 224, 3), num_classes=38, trainable_layers=5):
    """
    Create a VGG16-based transfer learning model.
    Classic architecture with deep convolutional layers.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of disease classes
        trainable_layers: Number of layers to unfreeze for fine-tuning
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained VGG16
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base model layers
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    # Add custom top layers
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def create_inceptionv3_model(input_shape=(299, 299, 3), num_classes=38, trainable_layers=30):
    """
    Create an InceptionV3-based transfer learning model.
    Multi-scale feature extraction with inception modules.
    
    Args:
        input_shape: Shape of input images (InceptionV3 requires 299x299)
        num_classes: Number of disease classes
        trainable_layers: Number of layers to unfreeze for fine-tuning
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained InceptionV3
    base_model = InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    
    # Freeze base model layers
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
    
    # Add custom top layers
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model


def get_model(model_name='efficientnetb0', input_shape=(224, 224, 3), num_classes=38):
    """
    Factory function to get the specified model.
    
    Args:
        model_name: Name of the model
        input_shape: Shape of input images
        num_classes: Number of disease classes
        
    Returns:
        Compiled Keras model
    """
    model_dict = {
        'custom_cnn': create_custom_cnn,
        'resnet50': create_resnet50_model,
        'efficientnetb0': create_efficientnet_model,
        'mobilenetv2': create_mobilenet_model,
        'densenet121': create_densenet_model,
        'vgg16': create_vgg16_model,
        'inceptionv3': create_inceptionv3_model
    }
    
    # Adjust input shape for InceptionV3
    if model_name.lower() == 'inceptionv3':
        input_shape = (299, 299, 3)
    
    if model_name.lower() not in model_dict:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(model_dict.keys())}")
    
    print(f"\n🔨 Building {model_name.upper()} model...")
    model = model_dict[model_name.lower()](input_shape, num_classes)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    print(f"✅ Model built successfully!")
    print(f"📊 Total parameters: {model.count_params():,}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...\n")
    
    models_to_test = ['custom_cnn', 'resnet50', 'efficientnetb0', 'mobilenetv2']
    
    for model_name in models_to_test:
        try:
            model = get_model(model_name, num_classes=38)
            print(f"\n{model_name.upper()} Summary:")
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
            print(f"Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Error creating {model_name}: {e}")
