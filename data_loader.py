import tensorflow as tf
import numpy as np

def get_mild_square_root_sampler(labels, power=0.3):
    class_counts = np.bincount(labels)
    class_weights = 1. / (class_counts ** power)
    sample_weights = class_weights[labels]
    return sample_weights

def create_augmentation_layer(config):
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(config['data']['augmentation']['rotation_range']),
        tf.keras.layers.RandomZoom(config['data']['augmentation']['zoom_range']),
        tf.keras.layers.RandomContrast(config['data']['augmentation']['contrast_range']),
        tf.keras.layers.RandomBrightness(config['data']['augmentation']['brightness_range']),
    ])

def load_data(config, model_name):
    img_size = config['models'][model_name]['img_size']
    
    augmentation_layer = create_augmentation_layer(config)
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config['data']['train_dir'],
        image_size=(img_size, img_size),
        batch_size=config['data']['batch_size'],
        shuffle=False  # We'll handle shuffling ourselves
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config['data']['val_dir'],
        image_size=(img_size, img_size),
        batch_size=config['data']['batch_size']
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config['data']['test_dir'],
        image_size=(img_size, img_size),
        batch_size=config['data']['batch_size']
    )

    # Calculate sampling weights
    train_labels = np.concatenate([y for x, y in train_ds], axis=0)
    sample_weights = get_mild_square_root_sampler(train_labels, config['sampling']['power'])

    # Create a dataset of weights
    weight_ds = tf.data.Dataset.from_tensor_slices(sample_weights)

    # Zip together the training data and weights
    train_ds = tf.data.Dataset.zip((train_ds, weight_ds))

    # Apply augmentation and use the weights
    def apply_augmentation_and_weights(data, weight):
        image, label = data
        return augmentation_layer(image, training=True), label, weight

    train_ds = train_ds.map(apply_augmentation_and_weights)

    # Shuffle and repeat the dataset
    train_ds = train_ds.shuffle(buffer_size=len(train_labels)).repeat()

    # Cache and prefetch all datasets
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, len(train_labels)