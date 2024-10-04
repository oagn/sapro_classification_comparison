import tensorflow as tf
import keras_cv
import numpy as np

def get_mild_square_root_sampler(labels, power=0.3):
    class_counts = np.bincount(labels)
    class_weights = 1. / (class_counts ** power)
    sample_weights = class_weights[labels]
    return sample_weights

def load_data(config, model_name):
    img_size = config['models'][model_name]['img_size']
    
    # Create RandAugment layer
    rand_aug = keras_cv.layers.RandAugment(
        value_range=(0, 255),
        augmentations_per_image=3,
        magnitude=config['data'].get('augmentation_magnitude', 0.3)
    )

    def preprocess_and_augment(image, label):
        image = tf.cast(image, tf.float32)
        image = rand_aug(image)
        image = image / 255.0  # Normalize to [0, 1]
        return image, label

    train_ds = tf.keras.utils.image_dataset_from_directory(
        config['data']['train_dir'],
        image_size=(img_size, img_size),
        batch_size=config['data']['batch_size'],
        shuffle=True,
        seed=42
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

    # Apply augmentation to training data
    train_ds = train_ds.map(preprocess_and_augment, num_parallel_calls=tf.data.AUTOTUNE)

    # Calculate sampling weights
    train_labels = np.concatenate([y for x, y in train_ds], axis=0)
    sample_weights = get_mild_square_root_sampler(train_labels, config['sampling']['power'])

    # Add weights to the training dataset
    train_ds = train_ds.map(lambda x, y: (x, y, tf.gather(sample_weights, y)))

    # Shuffle and prefetch
    train_ds = train_ds.shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)

    # Convert to NumPy iterator for JAX compatibility
    train_iter = iter(train_ds.as_numpy_iterator())

    def get_next_batch():
        return next(train_iter)

    return get_next_batch, val_ds, test_ds, len(train_labels)