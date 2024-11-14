import keras
from keras import layers
import numpy as np
from keras.applications import (
    ResNet50,
    MobileNetV3Small,
    EfficientNetV2B0,
    EfficientNetV2S, 
    EfficientNetV2M
)

def get_base_model(model_name, config, weights_path=None):
    """Get the base model architecture"""
    input_shape = (config['models'][model_name]['img_size'],
                  config['models'][model_name]['img_size'], 3)
    
    if model_name.startswith('ResNet50'):
        return ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name.startswith('MobileNetV3S'):
        return MobileNetV3Small(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name.startswith('EfficientNetV2B0'):
        return EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name.startswith('EfficientNetV2S'):
        return EfficientNetV2S(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name.startswith('EfficientNetV2M'):
        return EfficientNetV2M(weights='imagenet', include_top=False, input_shape=input_shape)
    # ... other model options ...
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def create_model(model_name, config):
    """
    Create model with proper initialization for Focal Loss
    """
    weights_path = config['models'][model_name].get('weights_path', None)    
    base_model = get_base_model(model_name, config)
    
    # Add classification head
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)  # Flatten the output to 2D
    x = keras.layers.Dense(config['models'][model_name]['num_dense_layers'], activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # Initialize the final layer with bias for Focal Loss
    num_classes = len(config['data']['class_names'])
    if num_classes == 2:
        # Binary classification
        pi = 0.01  # Initial probability for positive class
        bias_init = -np.log((1-pi)/pi)  # ≈ -2.0
        outputs = keras.layers.Dense(
            1,
            activation='sigmoid',
            bias_initializer=keras.initializers.Constant(bias_init),
            name='focal_loss_output'
        )(x)
    else:
        # Multi-class classification
        pi = 0.01
        bias_init = -np.log((1-pi)/pi)
        outputs = keras.layers.Dense(
            num_classes,
            activation='softmax',
            bias_initializer=keras.initializers.Constant(bias_init),
            name='focal_loss_output'
        )(x)
    
    model = keras.Model(inputs=base_model.input, outputs=outputs)
    
    if weights_path:
        print(f"Loading weights from {weights_path}")
        model.load_weights(weights_path, skip_mismatch=True)
    else:
        print("No pre-trained weights provided. Using imagenet weights.")
        
    # Freeze base model layers for initial training
    for layer in base_model.layers:
        layer.trainable = False
    
    # Print model summary for debugging
    print("\nModel Architecture:")
    model.summary()
    
    print(f"\nModel created with Focal Loss initialization (bias = {bias_init:.3f})")
    print(f"Number of classes: {num_classes}")
    print(f"Output activation: {'sigmoid' if num_classes == 2 else 'softmax'}")
    print(f"Output shape: {model.output_shape}")
    
    return model


def unfreeze_model(model, num_layers_to_unfreeze):
    # First, make the entire model trainable
    model.trainable = True
    for layer in model.layers:
        layer.trainable = True

    # Calculate the index of the first layer to keep trainable
    first_trainable_layer = len(model.layers) - num_layers_to_unfreeze

    # Freeze layers before the specified number of layers to unfreeze
    for layer in model.layers[:first_trainable_layer]:
        layer.trainable = False

    # Freeze BatchNormalization layers throughout the entire model
    for layer in model.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False

    return model
