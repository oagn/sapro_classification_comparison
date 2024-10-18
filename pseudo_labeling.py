import os
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from data_loader import create_tensorset, load_data
from models import create_model
from train import train_model

def generate_pseudo_labels(model, unlabeled_data_dir, config, model_name):
    """
    Generate pseudo-labels for unlabeled data using the given model.
    """
    img_size = config['models'][model_name]['img_size']
    batch_size = config['data']['batch_size']
    confidence_threshold = config['pseudo_labeling']['confidence_threshold']

    # Get all image files in the directory
    image_files = [os.path.join(unlabeled_data_dir, f) for f in os.listdir(unlabeled_data_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_files:
        raise ValueError(f"No images found in directory {unlabeled_data_dir}")

    # Function to load and preprocess images
    def load_and_preprocess_image(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
        return img

    # Create a dataset from the image files
    unlabeled_ds = tf.data.Dataset.from_tensor_slices(image_files)
    unlabeled_ds = unlabeled_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    unlabeled_ds = unlabeled_ds.batch(batch_size)

    pseudo_labeled_data = []
    for batch_images in unlabeled_ds:
        batch_predictions = model.predict(batch_images)
        for pred, file_path in zip(batch_predictions, image_files):
            confidence = np.max(pred)
            if confidence >= confidence_threshold:
                predicted_label = np.argmax(pred)
                pseudo_labeled_data.append((file_path, predicted_label, confidence))

    return pd.DataFrame(pseudo_labeled_data, columns=['File', 'Label', 'Confidence'])

def combine_datasets(original_df, pseudo_df):
    """
    Combine original labeled data with pseudo-labeled data.
    """
    return pd.concat([original_df, pseudo_df[['File', 'Label']]], ignore_index=True)

def retrain_with_pseudo_labels(model, combined_df, config, model_name):
    """
    Retrain the model using the combined dataset of original and pseudo-labeled data.
    """
    if config['pseudo_labeling'].get('use_all_samples', True):
        train_ds = create_tensorset(combined_df, 
                                    config['models'][model_name]['img_size'],
                                    config['data']['batch_size'],
                                    config['data'].get('augmentation_magnitude', 0.3),
                                    ds_name="train",
                                    model_name=model_name)
    else:
        num_samples = config['pseudo_labeling'].get('num_samples_per_epoch', len(combined_df))
        initial_sample = combined_df.sample(n=num_samples, replace=False)
        train_ds = create_tensorset(initial_sample, 
                                    config['models'][model_name]['img_size'],
                                    config['data']['batch_size'],
                                    config['data'].get('augmentation_magnitude', 0.3),
                                    ds_name="train",
                                    model_name=model_name)

    _, val_ds, _, _, _, _ = load_data(config, model_name)  # Get val_ds

    history = train_model(
        model,
        train_ds,
        val_ds,
        config,
        learning_rate=config['training']['learning_rate'],
        epochs=config['pseudo_labeling']['retraining_epochs'],
        image_size=config['models'][model_name]['img_size'],
        model_name=model_name,
        is_fine_tuning=True
    )

    return model, history

def pseudo_labeling_pipeline(config):
    """
    Main function to run the pseudo-labeling pipeline.
    """
    # Ensure model_name is specified in the config
    if 'model_name' not in config['pseudo_labeling']:
        raise ValueError("'model_name' must be specified in the 'pseudo_labeling' section of the config.")
    
    model_name = config['pseudo_labeling']['model_name']
    
    # Load the trained model
    model = load_model(config['pseudo_labeling']['model_path'])
    
    # Generate pseudo-labels
    pseudo_labeled_data = generate_pseudo_labels(model, config['pseudo_labeling']['unlabeled_data_dir'], config, model_name)
    
    print(f"Pseudo-labeled data distribution:")
    pseudo_labeled_data['Label'].count()
    
    # Load original labeled data
    _, _, _, _, _, train_df = load_data(config, model_name)  # Get train_df
    
    # Combine datasets
    combined_df = combine_datasets(train_df, pseudo_labeled_data)
    
    # Retrain the model
    retrained_model, history = retrain_with_pseudo_labels(model, combined_df, config, model_name)
    
    # Save the retrained model
    retrained_model.save(os.path.join(config['data']['output_dir'], f'retrained_model_{model_name}.keras'))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pseudo-labeling pipeline")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    pseudo_labeling_pipeline(config)
