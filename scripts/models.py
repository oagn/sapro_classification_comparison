import keras
from keras import layers
import numpy as np
from keras.applications import (
    ResNet50,
    MobileNetV3Small,
    EfficientNetV2B0,
    EfficientNetV2S
)

def get_base_model(model_name, config):
    """Loads a pre-trained base model architecture from Keras Applications.

    Args:
        model_name (str): Name of the model (e.g., 'ResNet50', 'EfficientNetV2S').
        config (dict): Configuration dictionary containing model parameters like img_size.

    Returns:
        keras.Model: The loaded base model instance with include_top=False.
    
    Raises:
        ValueError: If the model_name is not recognized.
    """
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
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def create_model(model_name, config):
    """
    Creates the complete Keras model for training.

    Steps:
    1. Loads the specified base model using get_base_model.
    2. Adds a classification head (GlobalAveragePooling, Dense, Dropout).
    3. Initializes the final Dense layer bias for binary classification 
       (helps with Focal Loss and imbalanced data).
    4. Optionally loads pre-trained weights if specified in config.
    5. Freezes the base model layers for initial transfer learning.

    Args:
        model_name (str): Name of the base model architecture.
        config (dict): Configuration dictionary.

    Returns:
        keras.Model: The complete, compiled Keras model ready for initial training.
    """
    weights_path = config['models'][model_name].get('weights_path', None)    
    base_model = get_base_model(model_name, config)
    
    # Add classification head
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)  # Flatten the output to 2D
    x = keras.layers.Dense(config['models'][model_name]['num_dense_layers'], activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    
    # Calculate beta for final layer bias
    num_classes = len(config['data']['class_names'])
    if num_classes == 2:  # Only for binary classification
        # Set pi to ~0.18 which gives beta ≈ -1.5
        pi = 0.01 
        beta = -np.log((1 - pi) / pi)
        print(f"\nInitializing final layer bias with beta = {beta:.3f}")
        print(f"This corresponds to a positive class prior of {pi:.1%}")
    
    # Initialize the final layer with bias for Focal Loss
    if num_classes == 2:
        # Binary classification
        outputs = keras.layers.Dense(
            2,
            activation='sigmoid',
            bias_initializer=keras.initializers.Constant(beta),
            name='output'
        )(x)
    else:
        # Multi-class classification
        outputs = keras.layers.Dense(
            num_classes,
            activation='softmax',
            bias_initializer=keras.initializers.Constant(beta),
            name='output'
        )(x)
    
    model = keras.Model(inputs=base_model.input, outputs=outputs)
    
    if weights_path:
        print(f"Loading custom weights from: {weights_path}")
        # skip_mismatch=True allows loading weights even if the head differs
        model.load_weights(weights_path, skip_mismatch=True) 
    else:
        # Defaults to imagenet weights loaded in get_base_model
        print(f"Using default 'imagenet' weights for {model_name}.") 
        
    # Freeze base model layers for initial training
    for layer in base_model.layers:
        layer.trainable = False
    
    return model


def unfreeze_model(model, num_layers_to_unfreeze):
    """Unfreezes the last N layers of a model for fine-tuning.

    It first sets the entire model as trainable, then freezes layers 
    from the beginning up to the point specified by num_layers_to_unfreeze.
    Crucially, it keeps all BatchNormalization layers frozen throughout the model,
    which is generally recommended during fine-tuning.

    Args:
        model (keras.Model): The model to modify.
        num_layers_to_unfreeze (int): The number of layers to keep trainable 
                                     (counting from the output layer backwards).

    Returns:
        keras.Model: The modified model with layers appropriately frozen/unfrozen.
    """
    # First, set the entire model to trainable
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
