import tensorflow as tf
import keras_cv
import keras
import numpy as np
import jax.numpy as jnp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from pathlib import Path
import os
from collections import Counter
from keras.applications import resnet


def create_fixed_train(ds_path, samples_per_class=None):
    # Selecting folder paths in dataset
    dir_ = Path(ds_path)
    ds_filepaths = list(dir_.glob(r'**/*.jpg'))
    ds_filepaths.extend(list(dir_.glob(r'**/*.JPG')))
    ds_filepaths.extend(list(dir_.glob(r'**/*.jpeg')))
    ds_filepaths.extend(list(dir_.glob(r'**/*.png')))
    ds_filepaths.extend(list(dir_.glob(r'**/*.PNG')))
    # Mapping labels...
    ds_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], ds_filepaths))
    # Data set paths & labels
    ds_filepaths = pd.Series(ds_filepaths, name='File').astype(str)
    ds_labels = pd.Series(ds_labels, name='Label')
    # Concatenating...
    ds_df = pd.concat([ds_filepaths, ds_labels], axis=1)
    ds_df = ds_df.sample(frac=1, random_state=42).reset_index(drop=True)
    if samples_per_class is not None:
        ds_df = ds_df.groupby('Label').sample(n=samples_per_class, replace=True)
    return ds_df

def create_fixed(ds_path):
    # Selecting folder paths in dataset
    dir_ = Path(ds_path)
    ds_filepaths = list(dir_.glob(r'**/*.jpg'))
    ds_filepaths.extend(list(dir_.glob(r'**/*.JPG')))
    ds_filepaths.extend(list(dir_.glob(r'**/*.jpeg')))
    ds_filepaths.extend(list(dir_.glob(r'**/*.png')))
    ds_filepaths.extend(list(dir_.glob(r'**/*.PNG')))
    # Mapping labels...
    ds_labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], ds_filepaths))
    # Data set paths & labels
    ds_filepaths = pd.Series(ds_filepaths, name='File').astype(str)
    ds_labels = pd.Series(ds_labels, name='Label')
    # Concatenating...
    ds_df = pd.concat([ds_filepaths, ds_labels], axis=1)
    return ds_df


def create_tensorset(in_df, img_size, batch_size, magnitude, ds_name="train", sample_weights=None, model_name=None):
    def load(file_path, img_size):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        return img

    def preprocess(img, model_name):
        if model_name == 'ResNet50':
            # Manual implementation of ResNet50 preprocessing
            mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
            img = img[..., ::-1]  # RGB to BGR
            img -= mean
            return img
        elif model_name in ['MobileNetV3L', 'MobileNetV3S', 'EfficientNetV2B0', 'EfficientNetV2S']:
            return img  # No preprocessing needed, it's built into the model
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    in_path = in_df['File'].values
    label_encoder = LabelEncoder()
    in_class = label_encoder.fit_transform(in_df['Label'].values)

    in_class = in_class.reshape(len(in_class), 1)
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    in_class = one_hot_encoder.fit_transform(in_class)
    
    rand_aug = keras_cv.layers.RandAugment(
        value_range=(0, 255), augmentations_per_image=3, magnitude=magnitude)

    if ds_name == "train" and sample_weights is not None:
        # Include sample weights in the dataset
        ds = tf.data.Dataset.from_tensor_slices((in_path, in_class, sample_weights))
        ds = (ds
            .map(lambda img_path, img_class, weight: (load(img_path, img_size), img_class, weight), 
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .map(lambda x, y, w: (rand_aug(tf.cast(x, tf.uint8)), y, w), 
                 num_parallel_calls=tf.data.AUTOTUNE)
            .map(lambda x, y, w: (preprocess(x, model_name), y, w),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        # Don't include sample weights for validation and test sets, or when weights are not used
        ds = tf.data.Dataset.from_tensor_slices((in_path, in_class))
        ds = (ds
            .map(lambda img_path, img_class: (load(img_path, img_size), img_class), 
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .map(lambda x, y: (preprocess(x, model_name), y),
                 num_parallel_calls=tf.data.AUTOTUNE)
        )
        if ds_name == "train":
            ds = ds.map(lambda x, y: (rand_aug(tf.cast(x, tf.uint8)), y), 
                        num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)

    if ds_name == "train":
        ds = ds.shuffle(buffer_size=len(in_df), reshuffle_each_iteration=True)
        #ds = ds.repeat()

    return ds


def get_minority_oversampling_weights(labels):
    class_counts = np.bincount(labels)
    minority_class = np.argmin(class_counts)
    weights = np.ones_like(labels, dtype=np.float32)
    weights[labels == minority_class] = class_counts.max() / class_counts[minority_class]
    return weights

def get_square_root_sampling_weights(labels, mild=False, power=0.5):
    class_counts = np.bincount(labels)
    if mild:
        weights = np.power(class_counts.max() / class_counts, power)
    else:
        weights = np.power(class_counts, power)
    return weights[labels]

def get_balanced_sampling_weights(labels):
    class_counts = np.bincount(labels)
    target_count = np.mean(class_counts)  # We'll aim for this count for each class
    weights = np.zeros_like(labels, dtype=np.float32)
    
    for class_label, count in enumerate(class_counts):
        if count > target_count:
            # Undersample
            weights[labels == class_label] = target_count / count
        else:
            # Oversample
            weights[labels == class_label] = target_count / count
    
    return weights

def apply_sampling_method(labels, sampling_config):
    sampling_method = sampling_config['method']
    sampling_power = sampling_config.get('power', 0.5)  # Default to 0.5 if not specified
    samples_per_class = sampling_config.get('samples_per_class', None)
    
    print(f"Applying sampling method: {sampling_method} with power: {sampling_power}")
    
    if sampling_method == 'minority_oversampling':
        return get_minority_oversampling_weights(labels)
    elif sampling_method == 'mild_square_root':
        return get_square_root_sampling_weights(labels, mild=True, power=sampling_power)
    elif sampling_method == 'square_root':
        return get_square_root_sampling_weights(labels, mild=False, power=sampling_power)
    elif sampling_method == 'balanced':
        return get_balanced_sampling_weights(labels)
    elif sampling_method == 'equal':
        return get_equal_sampling_weights(labels, samples_per_class)
    else:
        print(f"Unknown sampling method: {sampling_method}. No sampling applied.")
        return np.ones_like(labels, dtype=np.float32)  # No sampling

def create_counting_dataset(in_df, img_size, batch_size, sample_weights=None):
    in_path = in_df['File']
    label_encoder = LabelEncoder()
    in_class = label_encoder.fit_transform(in_df['Label'].values)

    in_class = in_class.reshape(len(in_class), 1)
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    in_class = one_hot_encoder.fit_transform(in_class)
    
    if sample_weights is None:
        sample_weights = np.ones(len(in_df), dtype=np.float32)
    
    ds = tf.data.Dataset.from_tensor_slices((in_class, sample_weights))
    ds = ds.batch(batch_size)
    
    return ds

def count_classes_from_dataset(dataset):
    class_counts = Counter()
    
    for labels, weights in dataset:
        labels = tf.argmax(labels, axis=1)  # Convert one-hot to class indices
        weights = tf.round(weights)  # Round weights to nearest integer
        for label, weight in zip(labels.numpy(), weights.numpy()):
            class_counts[label] += int(weight)
    
    return class_counts

def get_equal_sampling_weights(labels, samples_per_class=None):
    class_counts = np.bincount(labels)
    if samples_per_class is None:
        samples_per_class = class_counts.min()
    
    weights = np.zeros_like(labels, dtype=np.float32)
    for class_label, count in enumerate(class_counts):
        weights[labels == class_label] = samples_per_class / count
    
    return weights

def load_data(config, model_name):
    img_size = config['models'][model_name]['img_size']
    batch_size = config['data']['batch_size']
    augmentation_magnitude = config['data'].get('augmentation_magnitude', 0.3)
    samples_per_class = config['sampling'].get('samples_per_class', None)

    print(f"Loading data for model: {model_name}")
    print(f"Image size: {img_size}, Batch size: {batch_size}, Augmentation magnitude: {augmentation_magnitude}")

    # Load training data
    train_df = create_fixed_train(config['data']['train_dir'], samples_per_class)
    
    # Calculate sampling weights if enabled
    use_weights = config['sampling'].get('use_weights', True)
    if use_weights:
        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_df['Label'].values)
        
        print("Sampling configuration:")
        print(config['sampling'])
        
        sample_weights = apply_sampling_method(train_labels, config['sampling'])
        
        # Create dataset for counting
        counting_ds = create_counting_dataset(train_df, img_size, batch_size, sample_weights)
        
        # Count classes
        class_counts = count_classes_from_dataset(counting_ds)
        print("Class distribution in training set after sampling:")
        for class_label, count in class_counts.items():
            print(f"Class {class_label}: {count} samples")
    else:
        sample_weights = None
        class_counts = train_df['Label'].value_counts().to_dict()
        print("Class distribution in training set (no sampling):")
        for class_label, count in class_counts.items():
            print(f"Class {class_label}: {count} samples")
    
    # Create training dataset with or without sampling weights
    train_ds = create_tensorset(train_df, img_size, batch_size, augmentation_magnitude, 
                                ds_name="train", sample_weights=sample_weights, model_name=model_name)

    # Load validation data
    val_df = create_fixed(config['data']['val_dir'])
    val_ds = create_tensorset(val_df, img_size, batch_size, 0, ds_name="validation", model_name=model_name)

    # Load test data
    test_df = create_fixed(config['data']['test_dir'])
    test_ds = create_tensorset(test_df, img_size, batch_size, 0, ds_name="test", model_name=model_name)

    num_train_samples = sum(class_counts.values())
    print(f"Loaded {num_train_samples} training samples, {len(val_df)} validation samples, and {len(test_df)} test samples")

    return train_ds, val_ds, test_ds, num_train_samples, len(val_df)


