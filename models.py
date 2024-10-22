import keras
from keras import layers

def create_model(model_name, config, weights_path=None):
    num_classes = len(config['data']['class_names'])
    print(f"Creating model with {num_classes} output classes")
    img_size = config['models'][model_name]['img_size']
    
    if model_name == 'MobileNetV3L':
        base_model = keras.applications.MobileNetV3Large(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
    elif model_name == 'MobileNetV3S':
        base_model = keras.applications.MobileNetV3Small(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
    elif model_name == 'EfficientNetV2B0':
        base_model = keras.applications.EfficientNetV2B0(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
    elif model_name == 'EfficientNetV2S':
        base_model = keras.applications.EfficientNetV2S(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
    elif model_name == 'EfficientNetV2M':
        base_model = keras.applications.EfficientNetV2M(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
    elif model_name == 'ResNet50':
        base_model = keras.applications.ResNet50(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Freeze the base model
    base_model.trainable = False
    
    x = layers.Dense(config['models'][model_name]['num_dense_layers'], activation='relu')(base_model.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs=base_model.input, outputs=outputs)
    
    if weights_path:
        print(f"Loading weights from {weights_path}")
        model.load_weights(weights_path, skip_mismatch=True)
    else:
        print("No pre-trained weights provided. Using imagenet weights.")
    
    print(f"Model created with output shape: {model.output_shape}")
    
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
