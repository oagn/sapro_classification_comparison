import tensorflow as tf
import keras_cv
import numpy as np
import jax.numpy as jnp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from pathlib import Path
import os


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


# This function takes a pandas df from create_dataframe and converts to a TensorFlow dataset
def create_tensorset(in_df, img_size, batch_size, magnitude, ds_name="train", sample_weights=None):
    def load(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.uint8)
        img = tf.image.resize(img, size=(img_size, img_size))
        return img

    in_path = in_df['File']
    label_encoder = LabelEncoder()
    in_class = label_encoder.fit_transform(in_df['Label'].values)

    in_class = in_class.reshape(len(in_class), 1)
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    in_class = one_hot_encoder.fit_transform(in_class)
    
    rand_aug = keras_cv.layers.RandAugment(
        value_range=(0, 255), augmentations_per_image=3, magnitude=magnitude)

    if ds_name == "train":
        if sample_weights is None:
            raise ValueError("Sample weights must be provided for training data")
        
        # Implement weighted sampling manually
        sample_weights = sample_weights / np.sum(sample_weights)
        choice_indices = np.random.choice(len(in_df), size=len(in_df), p=sample_weights)
        
        in_path = tf.gather(in_path, choice_indices)
        in_class = tf.gather(in_class, choice_indices)
        sample_weights = tf.gather(sample_weights, choice_indices)
        
        ds = tf.data.Dataset.from_tensor_slices((in_path, in_class, sample_weights))
        
        ds = (ds
            .map(lambda img_path, img_class, weight: (load(img_path), img_class, weight), 
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .map(lambda x, y, w: (rand_aug(tf.cast(x, tf.uint8)), y, w), 
                 num_parallel_calls=tf.data.AUTOTUNE)
            .repeat()  # Add this line
            .prefetch(tf.data.AUTOTUNE)
        )
    elif ds_name == "validation":
        ds = (tf.data.Dataset.from_tensor_slices((in_path, in_class))
            .map(lambda img_path, img_class: (load(img_path), img_class), 
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .repeat()  # Add this line
            .prefetch(tf.data.AUTOTUNE)
        )
    else:  # test
        ds = (tf.data.Dataset.from_tensor_slices((in_path, in_class))
            .map(lambda img_path, img_class: (load(img_path), img_class), 
                 num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .repeat()  # Add this line
            .prefetch(tf.data.AUTOTUNE)
        )  
    return ds


def get_mild_square_root_sampler(labels, power=0.3):
    class_counts = np.bincount(labels)
    class_weights = 1. / (class_counts ** power)
    sample_weights = class_weights[labels]
    return sample_weights

def load_data(config, model_name):
    img_size = config['models'][model_name]['img_size']
    batch_size = config['data']['batch_size']
    augmentation_magnitude = config['data'].get('augmentation_magnitude', 0.3)

    # Load training data
    train_df = create_fixed(config['data']['train_dir'])
    
    # Calculate sampling weights
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df['Label'].values)
    sample_weights = get_mild_square_root_sampler(train_labels, config['sampling']['power'])
    
    # Create training dataset with weighted sampling
    train_ds = create_tensorset(train_df, img_size, batch_size, augmentation_magnitude, 
                                ds_name="train", sample_weights=sample_weights)

    # Load validation data
    val_df = create_fixed(config['data']['val_dir'])
    val_ds = create_tensorset(val_df, img_size, batch_size, 0, ds_name="validation")

    # Load test data
    test_df = create_fixed(config['data']['test_dir'])
    test_ds = create_tensorset(test_df, img_size, batch_size, 0, ds_name="test")

    return train_ds, val_ds, test_ds, len(train_df), len(val_df)