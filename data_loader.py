import tensorflow as tf
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

def get_mild_square_root_sampler(labels, power=0.3):
    class_counts = np.bincount(labels)
    class_weights = 1. / (class_counts ** power)
    sample_weights = class_weights[labels]
    return sample_weights

def apply_augmentation(image, config):
    aug_config = config['data'].get('augmentation', {})
    
    # Random flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    
    # Random rotation
    angle = tf.random.uniform((), minval=-aug_config.get('rotation_range', 0.2), maxval=aug_config.get('rotation_range', 0.2))
    image = tf.image.rotate(image, angle)
    
    # Random zoom
    zoom = tf.random.uniform((), minval=1-aug_config.get('zoom_range', 0.2), maxval=1+aug_config.get('zoom_range', 0.2))
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    crop_h = tf.cast(h / zoom, tf.int32)
    crop_w = tf.cast(w / zoom, tf.int32)
    image = tf.image.random_crop(image, (crop_h, crop_w, 3))
    image = tf.image.resize(image, (h, w))
    
    # Random contrast
    image = tf.image.random_contrast(image, 1-aug_config.get('contrast_range', 0.2), 1+aug_config.get('contrast_range', 0.2))
    
    # Random brightness
    image = tf.image.random_brightness(image, aug_config.get('brightness_range', 0.2))
    
    return tf.clip_by_value(image, 0, 1)

def load_data(config, model_name):
    img_size = config['models'][model_name]['img_size']
    
    def parse_and_augment(example):
        image = tf.image.resize(example['image'], (img_size, img_size))
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
        image = apply_augmentation(image, config)
        label = example['label']
        return image, label

    train_ds = tf.data.Dataset.load(config['data']['train_dir'])
    val_ds = tf.data.Dataset.load(config['data']['val_dir'])
    test_ds = tf.data.Dataset.load(config['data']['test_dir'])

    # Apply parsing and augmentation
    train_ds = train_ds.map(parse_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x: (tf.cast(tf.image.resize(x['image'], (img_size, img_size)), tf.float32) / 255.0, x['label']))
    test_ds = test_ds.map(lambda x: (tf.cast(tf.image.resize(x['image'], (img_size, img_size)), tf.float32) / 255.0, x['label']))

    # Calculate sampling weights
    train_labels = np.array(list(train_ds.map(lambda x, y: y).as_numpy_iterator()))
    sample_weights = get_mild_square_root_sampler(train_labels, config['sampling']['power'])

    # Add weights to the training dataset
    train_ds = train_ds.map(lambda x, y: (x, y, sample_weights[y]))

    # Batch and prefetch
    train_ds = train_ds.shuffle(10000).batch(config['data']['batch_size']).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(config['data']['batch_size']).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(config['data']['batch_size']).prefetch(tf.data.AUTOTUNE)

    # Convert to NumPy iterator for JAX compatibility
    train_iter = iter(train_ds.as_numpy_iterator())

    def get_next_batch():
        return next(train_iter)

    return get_next_batch, val_ds, test_ds, len(train_labels)