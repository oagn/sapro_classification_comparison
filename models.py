import keras
from keras import layers

def create_model(model_name, num_classes, config):
    print(f"Creating model: {model_name}")
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
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=base_model.input, outputs=outputs)
    return model


def unfreeze_model(model, num_layers_to_unfreeze):
    # Make the whole model trainable
    model.trainable = True
    
    # Get the total number of layers in the model
    total_layers = len(model.layers)
    
    # Determine the starting index for unfreezing
    start_unfreeze = max(0, total_layers - num_layers_to_unfreeze)
    
    # Freeze/unfreeze layers
    for i, layer in enumerate(model.layers):
        if i < start_unfreeze or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    
    return model