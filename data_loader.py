import keras_cv
import keras
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from pathlib import Path
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold
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


def oversample_minority_class(train_fold_df, strategy='threshold', threshold_ratio=0.5, random_state=42):
    """
    Apply oversampling to minority class using different strategies
    
    Based on:
    - Buda et al. (2018) "A systematic study of the class imbalance problem in CNNs"
    - More et al. (2016) "Survey of resampling techniques for improving classification
      performance in unbalanced datasets"
    
    Strategies:
    - 'threshold': Oversample to reach a specified ratio of majority class
    - 'match_majority': Oversample to match majority class count
    - 'progressive': Progressively increase minority samples (sqrt of difference)
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

def prepare_cross_validation_data(data_dir, config, model_name, random_state=42):
    """
    Prepare data for k-fold cross validation with image data and oversampling
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
        kfold = StratifiedGroupKFold(n_splits=config['training']['n_folds'], shuffle=True,random_state=42)
        splits = kfold.split(df, y=df['strat_col'], groups=df['user'])
        print("\nUsing StratifiedGroupKFold with user grouping")
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
        
        # Apply oversampling to training data
        if config['training'].get('use_oversampling', False):
            train_fold_df = oversample_minority_class(
                train_fold_df,
                strategy=config['training'].get('sampling_strategy', 'threshold'),
                threshold_ratio=config['training'].get('threshold_ratio', 0.5),
                random_state=random_state + fold_idx  # Unique seed per fold
            )
        

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