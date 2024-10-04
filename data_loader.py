import keras
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

def get_mild_square_root_sampler(labels, power=0.3):
    class_counts = np.bincount(labels)
    class_weights = 1. / (class_counts ** power)
    sample_weights = class_weights[labels]
    return sample_weights

def random_flip(key, image):
    do_flip = random.bernoulli(key)
    return jnp.where(do_flip, image[:, ::-1, :], image)

def random_rotation(key, image, max_angle=0.2):
    angle = random.uniform(key, minval=-max_angle, maxval=max_angle)
    return jax.image.rotate(image, angle, mode='nearest')

def random_zoom(key, image, max_zoom=0.2):
    zoom = random.uniform(key, minval=1-max_zoom, maxval=1+max_zoom)
    h, w = image.shape[:2]
    crop_h = int(h / zoom)
    crop_w = int(w / zoom)
    start_h = random.randint(key, 0, h - crop_h + 1)
    start_w = random.randint(key, 0, w - crop_w + 1)
    crop = jax.lax.dynamic_slice(image, (start_h, start_w, 0), (crop_h, crop_w, 3))
    return jax.image.resize(crop, (h, w, 3), method='bilinear')

def random_contrast(key, image, max_factor=0.2):
    factor = random.uniform(key, minval=1-max_factor, maxval=1+max_factor)
    mean = jnp.mean(image, axis=(0, 1), keepdims=True)
    return (image - mean) * factor + mean

def random_brightness(key, image, max_delta=0.2):
    delta = random.uniform(key, minval=-max_delta, maxval=max_delta)
    return jnp.clip(image + delta, 0, 1)

def apply_augmentation(key, image, config):
    aug_config = config['data'].get('augmentation', {})
    
    key, subkey = random.split(key)
    image = random_flip(subkey, image)
    
    key, subkey = random.split(key)
    image = random_rotation(subkey, image, aug_config.get('rotation_range', 0.2))
    
    key, subkey = random.split(key)
    image = random_zoom(subkey, image, aug_config.get('zoom_range', 0.2))
    
    key, subkey = random.split(key)
    image = random_contrast(subkey, image, aug_config.get('contrast_range', 0.2))
    
    key, subkey = random.split(key)
    image = random_brightness(subkey, image, aug_config.get('brightness_range', 0.2))
    
    return image

def load_data(config, model_name):
    img_size = config['models'][model_name]['img_size']
    
    train_ds = keras.utils.image_dataset_from_directory(
        config['data']['train_dir'],
        image_size=(img_size, img_size),
        batch_size=config['data']['batch_size'],
        shuffle=False
    )
    val_ds = keras.utils.image_dataset_from_directory(
        config['data']['val_dir'],
        image_size=(img_size, img_size),
        batch_size=config['data']['batch_size']
    )
    test_ds = keras.utils.image_dataset_from_directory(
        config['data']['test_dir'],
        image_size=(img_size, img_size),
        batch_size=config['data']['batch_size']
    )

    # Calculate sampling weights
    train_labels = np.concatenate([y.numpy() for x, y in train_ds], axis=0)
    sample_weights = get_mild_square_root_sampler(train_labels, config['sampling']['power'])

    # Convert Keras datasets to NumPy arrays
    train_images = np.concatenate([x.numpy() for x, y in train_ds], axis=0)
    train_labels = np.concatenate([y.numpy() for x, y in train_ds], axis=0)

    # Apply augmentation and use the weights
    @jax.jit
    def augment_and_weight(key, image, label, weight):
        augmented_image = apply_augmentation(key, image, config)
        return augmented_image, label, weight

    # Use vmap to apply augmentation to all images
    augment_and_weight_batch = jax.vmap(augment_and_weight, in_axes=(0, 0, 0, 0))

    # Create a dataset of images, labels, and weights
    train_ds = keras.utils.dataset_from_tensor_slices((train_images, train_labels, sample_weights))

    # Shuffle and repeat the dataset
    train_ds = train_ds.shuffle(buffer_size=len(train_labels)).repeat()

    # Function to get next batch and apply augmentation
    def get_next_batch():
        batch = next(iter(train_ds))
        images, labels, weights = batch
        keys = random.split(random.PRNGKey(0), images.shape[0])
        return augment_and_weight_batch(keys, images, labels, weights)

    return get_next_batch, val_ds, test_ds, len(train_labels)