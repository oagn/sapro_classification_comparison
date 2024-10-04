import keras
from keras import layers

def create_model(model_name, num_classes, config):
    img_size = config['models'][model_name]['img_size']
    
    if model_name == 'MobileNetV3':
        base_model = keras.applications.MobileNetV3Large(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet'
        )
    elif model_name == 'EfficientNetV2B0':
        base_model = keras.applications.EfficientNetV2B0(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet'
        )
    elif model_name == 'EfficientNetV2S':
        base_model = keras.applications.EfficientNetV2S(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet'
        )
    elif model_name == 'ResNet50':
        base_model = keras.applications.ResNet50(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=base_model.input, outputs=outputs)
    return model