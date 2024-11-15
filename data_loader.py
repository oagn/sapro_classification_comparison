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
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from imblearn.over_sampling import SMOTE


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
    ds_labels = pd.Series(ds_labels, name='Label').astype(str)  # Convert to string
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
    ds_labels = pd.Series(ds_labels, name='Label').astype(str)  # Convert to string
    # Concatenating...
    ds_df = pd.concat([ds_filepaths, ds_labels], axis=1)
    return ds_df


def create_tensorset(in_df, img_size, batch_size, magnitude, ds_name="train", sample_weights=None, model_name=None, config=None):
    in_path = in_df['File'].values
    
    if config is not None and 'data' in config and 'class_names' in config['data']:
        # Create a mapping from string labels to integer indices
        class_names = config['data']['class_names']
        label_to_index = {label: index for index, label in enumerate(class_names)}
        
        # Convert string labels to integer indices
        in_class = in_df['Label'].map(label_to_index)
        
        # Check for NaN values (labels not in the mapping)
        if in_class.isna().any():
            print("Warning: Some labels are not in the class_names list:")
            print(in_df[in_class.isna()]['Label'].value_counts())
            # Fill NaN values with a default value (e.g., -1)
            in_class = in_class.fillna(-1)
        
        in_class = in_class.values
        
        # Ensure all values are non-negative integers
        if (in_class < 0).any():
            print("Warning: Negative label indices found. Setting them to 0.")
            in_class[in_class < 0] = 0
        
        in_class = in_class.astype(int)
        
        # One-hot encode the integer indices
        in_class = tf.keras.utils.to_categorical(in_class, num_classes=len(class_names))
    else:
        # Fallback to the previous method if config is not available
        label_encoder = LabelEncoder()
        in_class = label_encoder.fit_transform(in_df['Label'].values)
        in_class = in_class.reshape(len(in_class), 1)
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        in_class = one_hot_encoder.fit_transform(in_class)

    def load(file_path, img_size):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        return img

    def get_base_model_name(model_name):
        """Extract base model name from potentially modified name (e.g., with fold number)"""
        # Split the model name by underscore and take the first part
        base_name = model_name.split('_')[0]
        return base_name

    def preprocess(img, model_name):
        """
        Preprocess image according to model requirements
        """
        # Extract base model name
        base_model_name = get_base_model_name(model_name)
        
        if base_model_name == 'ResNet50':
            # ResNet50 preprocessing
            # Manual implementation of ResNet50 preprocessing
            mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
            img = img[..., ::-1]  # RGB to BGR
            img -= mean
            return img
        elif base_model_name in ['MobileNetV3L', 'MobileNetV3S', 'EfficientNetV2B0', 'EfficientNetV2S','EfficientNetV2M']:
            return img  # No preprocessing needed, it's built into the model
        else:
            raise ValueError(f"Unknown model name: {base_model_name}")

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

    print("Unique labels in the dataset:", in_df['Label'].unique())
    print("Class names from config:", config['data']['class_names'])
    
    # After creating in_class
    print("Unique values in in_class:", np.unique(in_class, return_counts=True))
    
    return ds

def print_dsinfo(ds_df, ds_name='default'):
    print('Dataset: ' + ds_name)
    print(f'Number of images in the dataset: {ds_df.shape[0]}')
    print(ds_df['Label'].value_counts())
    print(f'\n')

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
        
    else:
        sample_weights = None
        class_counts = train_df['Label'].value_counts().to_dict()
        print("Class distribution in training set (no sampling):")
        for class_label, count in class_counts.items():
            print(f"Class {class_label}: {count} samples")
    
    # Create training dataset with or without sampling weights
    train_ds = create_tensorset(train_df, img_size, batch_size, augmentation_magnitude, 
                                ds_name="train", sample_weights=sample_weights, 
                                model_name=model_name, config=config)

    # Load validation data
    val_df = create_fixed(config['data']['val_dir'])
    val_ds = create_tensorset(val_df, img_size, batch_size, 0, ds_name="validation", 
                              model_name=model_name, config=config)

    # Load test data
    test_df = create_fixed(config['data']['test_dir'])
    test_ds = create_tensorset(test_df, img_size, batch_size, 0, ds_name="test", 
                               model_name=model_name, config=config)

    num_train_samples = sum(class_counts.values())
    print(f"Loaded {num_train_samples} training samples, {len(val_df)} validation samples, and {len(test_df)} test samples")

    return train_ds, val_ds, test_ds, num_train_samples, len(val_df), train_df  # Add train_df to the return statement

def create_train_test_split(df, test_size=0.2, random_state=42):
    """
    Create initial train-test split with held-out test set
    """
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['sapro']
    )
    return train_df, test_df

def create_dataset(features, labels, batch_size=32, shuffle=True):
    """
    Create TensorFlow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))
    dataset = dataset.batch(batch_size)
    return dataset

def prepare_fold_data(X_train, y_train, X_val, y_val, batch_size=32, random_state=42):
    """
    Prepare data for a single fold with SMOTE
    """
    # Apply SMOTE to training data
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Convert to float32/int32 for TensorFlow
    X_train_balanced = X_train_balanced.astype('float32')
    y_train_balanced = y_train_balanced.astype('int32')
    X_val = X_val.astype('float32')
    y_val = y_val.astype('int32')
    
    # Create TensorFlow datasets
    train_dataset = create_dataset(X_train_balanced, y_train_balanced, batch_size=batch_size)
    val_dataset = create_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)
    
    return train_dataset, val_dataset

def calculate_class_weights(train_fold_df, method='effective'):
    """
    Calculate class weights using different methods
    
    Args:
        train_fold_df: DataFrame containing training data
        method: 'balanced' or 'effective' (default)
            - 'balanced': inverse frequency
            - 'effective': effective number of samples
    """
    class_counts = train_fold_df['Label'].value_counts()
    total = len(train_fold_df)
    n_classes = len(class_counts)
    
    if method == 'balanced':
        # Simple inverse frequency weighting
        weights = {label: total/(n_classes*count) for label, count in class_counts.items()}
    
    elif method == 'effective':
        # Effective number of samples weighting (better for severe imbalance)
        beta = 0.9999  # Hyperparameter for effective samples
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = {label: (1.0 - beta) / effective_num[label] for label in class_counts.index}
        
        # Normalize weights
        weight_sum = sum(weights.values())
        weights = {label: (w * n_classes) / weight_sum for label, w in weights.items()}
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    print("\nClass weights:")
    for label, weight in weights.items():
        print(f"Class {label}: {weight:.3f}")
    
    return weights

def prepare_cross_validation_data(data_dir, config, model_name, random_state=42):
    """
    Prepare data for k-fold cross validation with image data and class balancing
    """
    # Load all image paths and labels
    df = create_fixed(data_dir)

    # Load metadata
    metadata = pd.read_csv(config['data']['metadata_path'])
    
    # Extract filename from full path in df
    df['filename'] = df['File'].apply(lambda x: os.path.basename(x))
    
    # Merge with metadata
    df = df.merge(
        metadata[['data_row.external_id', 'user', 'source', 'scientific_name_fixed']],
        left_on='filename',
        right_on='data_row.external_id',
        how='left'
    )

        # Create combined stratification column
    df['strat_col'] = df['source'] + '_' + df['scientific_name_fixed'] + '_' + df['Label'].astype(str)
    
    # Print merge statistics
    print("\nMerge Statistics:")
    print(f"Original df shape: {len(df)}")
    print(f"Rows with missing metadata: {df[['user', 'source', 'scientific_name_fixed']].isna().any(axis=1).sum()}")
    
    # Choose fold strategy
    use_groups = config['training'].get('use_groups', False)
    
    if use_groups:
        kfold = GroupKFold(n_splits=config['training']['n_folds'])
        splits = kfold.split(df, y=df['strat_col'], groups=df['user'])
        print("\nUsing GroupKFold with user grouping")
    else:
        kfold = StratifiedKFold(n_splits=config['training']['n_folds'], shuffle=True, random_state=42)
        splits = kfold.split(df, df['strat_col'])
        print("\nUsing StratifiedKFold without grouping")

    # Create datasets for each fold
    fold_datasets = []
    
    # Get parameters from config
    img_size = config['models'][model_name]['img_size']
    batch_size = config['data']['batch_size']
    augmentation_magnitude = config['data'].get('augmentation_magnitude', 0.3)
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\nPreparing Fold {fold_idx + 1}")
        
        # Split data for this fold
        train_fold_df = df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = df.iloc[val_idx].reset_index(drop=True)
        
        # Calculate class weights for this fold
        class_weights = calculate_class_weights(train_fold_df, method='effective')
        
        # Convert class weights to sample weights
        sample_weights = train_fold_df['Label'].map(class_weights).values

        print("\nClass distribution in training:")
        print(train_fold_df['Label'].value_counts())
        print("\nClass distribution in validation:")
        print(val_fold_df['Label'].value_counts())
        print("\nSource distribution in training:")
        print(train_fold_df['source'].value_counts())
        print("\nTaxonomic distribution in training:")
        print(train_fold_df['scientific_name_fixed'].value_counts())
        
        if use_groups:
            print("\nUser distribution:")
            print(f"Training users: {len(train_fold_df['user'].unique())}")
            print(f"Validation users: {len(val_fold_df['user'].unique())}")

        # Create datasets
        train_dataset = create_tensorset(
            train_fold_df,
            img_size,
            batch_size,
            augmentation_magnitude,
            ds_name="train",
            sample_weights=sample_weights,
            model_name=model_name,
            config=config
        )
        
        val_dataset = create_tensorset(
            val_fold_df,
            img_size,
            batch_size,
            0,  # No augmentation for validation
            ds_name="validation",
            model_name=model_name,
            config=config
        )
        
        fold_datasets.append((train_dataset, val_dataset))
        
        # Print fold statistics
        print(f"\nFold {fold_idx + 1} Statistics:")
        print(f"Training samples: {len(train_fold_df)}")
        print(f"Validation samples: {len(val_fold_df)}")
        print("\nClass distribution in training:")
        print(train_fold_df['Label'].value_counts())
        print("\nClass distribution in validation:")
        print(val_fold_df['Label'].value_counts())
    
    return fold_datasets


def get_class_weights(df, label_col='sapro'):
    """
    Calculate class weights for weighted loss function
    """
    class_counts = df[label_col].value_counts()
    total = len(df)
    weights = {i: total/(2*count) for i, count in class_counts.items()}
    return weights