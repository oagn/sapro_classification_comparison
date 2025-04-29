import keras_cv
import keras
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from pathlib import Path
import os
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold


def create_fixed(ds_path):
    """Scans a directory for image files and creates a DataFrame.

    Recursively searches for .jpg, .JPG, .jpeg, .png, .PNG files within the 
    given directory path. Extracts the parent directory name as the initial label.

    Args:
        ds_path (str or Path): Path to the root directory containing class subdirectories.

    Returns:
        pd.DataFrame: DataFrame with columns 'File' (str path) and 'Label' (str).
    """
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
    """Creates a tf.data.Dataset from a DataFrame of image paths and labels.

    Handles image loading, resizing, preprocessing (model-specific), 
    augmentation (RandAugment for training set), batching, and prefetching.
    Labels are one-hot encoded based on config['data']['class_names'].
    Includes optional handling for sample weights.

    Args:
        in_df (pd.DataFrame): DataFrame with 'File' and 'Label' columns.
        img_size (int): Target image size (height and width).
        batch_size (int): Dataset batch size.
        magnitude (float): Magnitude for RandAugment (0 to disable).
        ds_name (str): Name of the dataset ('train', 'validation', 'test'). 
                       Controls whether augmentation and shuffling are applied.
        sample_weights (np.array, optional): Array of sample weights to include 
                                         in the dataset (only for ds_name='train').
        model_name (str): Name of the model architecture (used for preprocessing).
        config (dict): Configuration dictionary (used for class_names).

    Returns:
        tf.data.Dataset: The configured TensorFlow dataset.
    """
    in_path = in_df['File'].values
    
    # --- Label Encoding ---
    if config is not None and 'data' in config and 'class_names' in config['data']:
        # Preferred method: Use class_names from config for consistent encoding
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
        # Fallback method: Use sklearn LabelEncoder/OneHotEncoder if config is missing
        print("Warning: Using LabelEncoder/OneHotEncoder for labels. Ensure config['data']['class_names'] is set for consistency.")
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
        """Preprocesses image according to specific model requirements.
        
        Args:
            img (tf.Tensor): Input image tensor.
            model_name (str): Name of the model architecture.
        
        Returns:
            tf.Tensor: Preprocessed image tensor.
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


def oversample_minority_class(train_fold_df, strategy='threshold', threshold_ratio=0.5, random_state=42):
    """Applies simple random oversampling to the minority class in a DataFrame.

    Based on selected strategy:
    - 'threshold': Oversample minority to reach (majority_count * threshold_ratio).
    - 'match_majority': Oversample minority to match majority count.
    - 'progressive': Oversample minority to minority_count + sqrt(difference).

    Args:
        train_fold_df (pd.DataFrame): DataFrame for the training fold ('Label' column required).
        strategy (str): Oversampling strategy ('threshold', 'match_majority', 'progressive').
        threshold_ratio (float): Ratio used for 'threshold' strategy.
        random_state (int): Random state for sampling.

    Returns:
        pd.DataFrame: The training DataFrame potentially with oversampled minority class rows.
    """
    print("\nBefore oversampling:")
    print(train_fold_df['Label'].value_counts())
    
    # Get class counts
    class_counts = train_fold_df['Label'].value_counts()
    majority_class = class_counts.index[0]
    majority_count = class_counts[majority_class]
    minority_class = class_counts.index[-1]
    minority_count = class_counts[minority_class]
    
    # Calculate target count based on strategy
    if strategy == 'threshold':
        target_count = int(majority_count * threshold_ratio)
        print(f"\nUsing threshold strategy with ratio {threshold_ratio}")
        print(f"Target count: {target_count}")
        
    elif strategy == 'match_majority':
        target_count = majority_count
        print(f"\nUsing match_majority strategy")
        print(f"Target count: {target_count}")
        
    elif strategy == 'progressive':
        # Square root of the difference plus original count
        # This provides a middle ground between no oversampling and full matching
        diff = majority_count - minority_count
        target_count = minority_count + int(np.sqrt(diff))
        print(f"\nUsing progressive strategy")
        print(f"Original difference: {diff}")
        print(f"Target count: {target_count}")
        
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    # Perform oversampling if needed
    if minority_count < target_count:
        minority_df = train_fold_df[train_fold_df['Label'] == minority_class]
        n_oversample = target_count - minority_count
        
        oversampled = minority_df.sample(
            n=n_oversample, 
            replace=True, 
            random_state=random_state
        )
        
        train_fold_df = pd.concat([
            train_fold_df[train_fold_df['Label'] != minority_class],
            minority_df,
            oversampled
        ]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        print("\nAfter oversampling:")
        print(train_fold_df['Label'].value_counts())
        print(f"Added {n_oversample} minority samples")
    
    return train_fold_df

def _prepare_single_fold_data(df, train_idx, val_idx, config, model_name, fold_idx, random_state):
    """Prepares training and validation datasets for a single CV fold.

    Handles splitting, oversampling (optional), and dataset creation.

    Args:
        df (pd.DataFrame): The complete dataset DataFrame.
        train_idx (np.array): Indices for the training split.
        val_idx (np.array): Indices for the validation split.
        config (dict): Configuration dictionary.
        model_name (str): Name of the model architecture.
        fold_idx (int): Current fold index.
        random_state (int): Base random state for this fold.

    Returns:
        tuple: (train_dataset, val_dataset, train_fold_df, val_fold_df)
            train_dataset (tf.data.Dataset): Training dataset for the fold.
            val_dataset (tf.data.Dataset): Validation dataset for the fold.
            train_fold_df (pd.DataFrame): DataFrame used for the training split.
            val_fold_df (pd.DataFrame): DataFrame used for the validation split.
    """
    print(f"\nPreparing Fold {fold_idx + 1}")

    # Split data for this fold
    train_fold_df = df.iloc[train_idx].reset_index(drop=True)
    val_fold_df = df.iloc[val_idx].reset_index(drop=True)

    # Apply oversampling to training data if enabled
    if config['training'].get('use_oversampling', False):
        train_fold_df = oversample_minority_class(
            train_fold_df,
            strategy=config['training'].get('sampling_strategy', 'threshold'),
            threshold_ratio=config['training'].get('threshold_ratio', 0.5),
            random_state=random_state # Use fold-specific seed
        )

    # --- Print Fold Statistics (before creating tensors) ---
    print("\nClass distribution in training fold:")
    print(train_fold_df['Label'].value_counts())
    print("\nClass distribution in validation fold:")
    print(val_fold_df['Label'].value_counts())
    print("\nSource distribution in training fold:")
    print(train_fold_df['source'].value_counts())
    print("\nTaxonomic distribution in training fold:")
    print(train_fold_df['scientific_name_fixed'].value_counts())
    
    if config['training'].get('use_groups', False):
        print("\nUser distribution in fold:")
        train_users = train_fold_df['user'].unique()
        val_users = val_fold_df['user'].unique()
        print(f"  Training users: {len(train_users)}")
        print(f"  Validation users: {len(val_users)}")
        # Optional: Check for user overlap if strict separation is critical
        # overlap = set(train_users).intersection(set(val_users))
        # if overlap:
        #     print(f"  Warning: User overlap between train/val: {len(overlap)} users")

    # --- Create Datasets ---
    img_size = config['models'][model_name]['img_size']
    batch_size = config['data']['batch_size']
    augmentation_magnitude = config['data'].get('augmentation_magnitude', 0.3)
    
    train_dataset = create_tensorset(
        train_fold_df,
        img_size,
        batch_size,
        augmentation_magnitude,
        ds_name="train",
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
    
    print(f"\nFold {fold_idx + 1} Dataset Statistics:")
    print(f"  Training samples (after oversampling): {len(train_fold_df)}")
    print(f"  Validation samples: {len(val_fold_df)}")

    return train_dataset, val_dataset, train_fold_df, val_fold_df

def prepare_cross_validation_data(data_dir, config, model_name, random_state=42):
    """Prepares data for k-fold cross-validation.

    Loads images, merges metadata, performs stratified/grouped splitting,
    handles optional oversampling within each fold's training split, 
    and creates tf.data.Dataset objects for each fold.

    Args:
        data_dir (str): Path to the root directory containing class subdirectories.
        config (dict): Configuration dictionary.
        model_name (str): Name of the model architecture.
        random_state (int): Base random state for splitting and sampling.

    Returns:
        list: A list of tuples, where each tuple contains (train_dataset, val_dataset) 
              for a single fold.
    """
    # Load all image paths and labels
    df = create_fixed(data_dir)

    # Load metadata
    metadata = pd.read_csv(config['data']['metadata_path'])
    
    # Extract filename from full path in df
    df['filename'] = df['File'].apply(lambda x: os.path.basename(x))
    
    # Merge with metadata
    print("\nMerging image data with metadata...")
    df = df.merge(
        metadata[['data_row.external_id', config['data']['group_column']] + config['data']['stratify_columns']],
        left_on='filename',
        right_on='data_row.external_id',
        how='left'
    )
    # Ensure 'Label' is included after merge if dropped
    if 'Label' not in df.columns:
        df = df.merge(create_fixed(data_dir)[['File', 'Label']], on='File')

    # Create combined stratification column from specified columns + Label
    strat_cols = config['data']['stratify_columns']
    df['strat_col'] = df[strat_cols].astype(str).agg('_'.join, axis=1) + '_' + df['Label'].astype(str)
    
    # Handle potential missing metadata after merge
    missing_metadata_count = df[config['data']['group_column']].isna().sum()
    if missing_metadata_count > 0:
        print(f"Warning: {missing_metadata_count} rows have missing group column ('{config['data']['group_column']}') after merge. Filling with 'unknown'.")
        df[config['data']['group_column']] = df[config['data']['group_column']].fillna('unknown')
        # Similarly handle missing stratify columns if necessary
        for col in strat_cols:
             if df[col].isna().any():
                  print(f"Warning: Filling missing values in stratification column '{col}' with 'unknown'.")
                  df[col] = df[col].fillna('unknown')
        # Recreate strat_col after filling NaNs
        df['strat_col'] = df[strat_cols].astype(str).agg('_'.join, axis=1) + '_' + df['Label'].astype(str)

    
    # Print merge statistics
    print("\nMerge Statistics:")
    print(f"  Original df shape: {len(df)}")
    print(f"  Rows with missing group ('{config['data']['group_column']}'): {df[config['data']['group_column']].isna().sum()}") # Should be 0 now
    
    # --- Prepare Cross-Validation Splits ---
    use_groups = config['training'].get('use_groups', False)
    n_folds = config['training']['n_folds']
    group_col = config['data']['group_column']
    
    if use_groups:
        kfold = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        # Ensure groups are not NaN
        groups = df[group_col].fillna('unknown').astype(str) # Ensure groups are strings and handle NaN just in case
        splits = kfold.split(df, y=df['strat_col'], groups=groups)
        print(f"\nUsing StratifiedGroupKFold with group column '{group_col}' and {n_folds} folds.")
    else:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = kfold.split(df, df['strat_col'])
        print(f"\nUsing StratifiedKFold with {n_folds} folds (no grouping).")

    # --- Process Each Fold ---
    all_fold_datasets = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_ds, val_ds, _, _ = _prepare_single_fold_data(
            df,
            train_idx,
            val_idx,
            config,
            model_name,
            fold_idx,
            random_state + fold_idx # Use unique seed per fold for sampling
        )
        all_fold_datasets.append((train_ds, val_ds))
        
    return all_fold_datasets