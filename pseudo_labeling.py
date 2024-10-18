import os
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from data_loader import create_tensorset, load_data
from models import create_model
from train import train_model

def generate_pseudo_labels(model, unlabeled_data_dir, config):
    """
    Generate pseudo-labels for unlabeled data using the given model.
    """
    img_size = config['models'][config['model_name']]['img_size']
    batch_size = config['data']['batch_size']
    confidence_threshold = config['pseudo_labeling']['confidence_threshold']

    unlabeled_ds = tf.keras.preprocessing.image_dataset_from_directory(
        unlabeled_data_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False
    )

    pseudo_labeled_data = []
    for images, file_paths in unlabeled_ds:
        predictions = model.predict(images)
        for pred, file_path in zip(predictions, file_paths.numpy()):
            confidence = np.max(pred)
            if confidence >= confidence_threshold:
                predicted_label = np.argmax(pred)
                pseudo_labeled_data.append((file_path.decode(), predicted_label, confidence))

    return pd.DataFrame(pseudo_labeled_data, columns=['File', 'Label', 'Confidence'])

def combine_datasets(original_df, pseudo_df):
    """
    Combine original labeled data with pseudo-labeled data.
    """
    return pd.concat([original_df, pseudo_df[['File', 'Label']]], ignore_index=True)

def retrain_with_pseudo_labels(model, combined_df, config):
    """
    Retrain the model using the combined dataset of original and pseudo-labeled data.
    """
    train_ds = create_tensorset(combined_df, 
                                config['models'][config['model_name']]['img_size'],
                                config['data']['batch_size'],
                                config['data'].get('augmentation_magnitude', 0.3),
                                ds_name="train",
                                model_name=config['model_name'])

    _, val_ds, _, _, _ = load_data(config, config['model_name'])  # Get val_ds

    history = train_model(
        model,
        train_ds,
        val_ds,
        config,
        learning_rate=config['training']['learning_rate'],
        epochs=config['pseudo_labeling']['retraining_epochs'],
        image_size=config['models'][config['model_name']]['img_size'],
        model_name=config['model_name'],
        is_fine_tuning=True
    )

    return model, history

def pseudo_labeling_pipeline(config):
    """
    Main function to run the pseudo-labeling pipeline.
    """
    # Load the trained model
    model = load_model(config['pseudo_labeling']['model_path'])
    
    # Generate pseudo-labels
    pseudo_labeled_data = generate_pseudo_labels(model, config['pseudo_labeling']['unlabeled_data_dir'], config)
    
    pseudo_labeled_data.groupby('Label').count()
    # Load original labeled data
    _, _, _, _, _, train_df = load_data(config, config['model_name'])  # Get train_df
    
    # Combine datasets
    combined_df = combine_datasets(train_df, pseudo_labeled_data, config)
    
    # Retrain the model
    retrained_model, history = retrain_with_pseudo_labels(model, combined_df, config)
    
    # Save the retrained model
    retrained_model.save(os.path.join(config['data']['output_dir'], 'retrained_model.keras'))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pseudo-labeling pipeline")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    pseudo_labeling_pipeline(config)
