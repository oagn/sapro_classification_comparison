import os
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from data_loader import create_tensorset, load_data, create_fixed, print_dsinfo
from models import create_model, unfreeze_model
from train import train_model

def generate_pseudo_labels(model, unlabeled_data_dir, config, model_name):
    """
    Generate pseudo-labels for unlabeled data using the given model.
    """
    img_size = config['models'][model_name]['img_size']
    batch_size = config['data']['batch_size']
    confidence_threshold = config['pseudo_labeling']['confidence_threshold']

    no_label_df = create_fixed(unlabeled_data_dir)
    print(f"Unique labels in no_label_df: {no_label_df['Label'].unique()}")
    print(f"Label distribution in no_label_df:\n{no_label_df['Label'].value_counts()}")

    no_label_set = create_tensorset(
        no_label_df, 
        img_size=img_size, 
        batch_size=batch_size, 
        magnitude=0, # no augmentation on test set
        ds_name="test", # set to test to turn off augmentation
        model_name=model_name,
        config=config  # Pass the config here
    )
    test_pred_raw = model.predict(no_label_set)
    test_pred = np.argmax(test_pred_raw, axis=1)
    
    # Convert predicted indices back to string labels
    class_names = config['data']['class_names']
    no_label_df['Label'] = [class_names[i] for i in test_pred]
    no_label_df['confidence'] = [max(x) for x in test_pred_raw]
    no_label_df['0_conf'] = [x[0] for x in test_pred_raw]
    no_label_df['12_conf'] = [x[1] for x in test_pred_raw]
    no_label_df['99_conf'] = [x[2] for x in test_pred_raw]

    return no_label_df[no_label_df['confidence'] >= confidence_threshold]

def combine_datasets(original_df, pseudo_df, config):
    # Ensure all labels are strings
    original_df['Label'] = original_df['Label'].astype(str)
    pseudo_df['Label'] = pseudo_df['Label'].astype(str)

    print_dsinfo(original_df, 'Old training Data')
    print_dsinfo(pseudo_df, 'Pseudo training Data')
    
    # Validate that all labels are in the expected set
    valid_labels = set(config['data']['class_names'])
    assert set(original_df['Label']).issubset(valid_labels), "Invalid labels in original dataset"
    assert set(pseudo_df['Label']).issubset(valid_labels), "Invalid labels in pseudo-labeled dataset"
    
    combined = pd.concat([original_df, pseudo_df[['File', 'Label']]], ignore_index=True)

    print_dsinfo(combined, 'New combined training Data')

    return combined

def retrain_with_pseudo_labels(model, combined_df, config, model_name):
    """
    Retrain the model using the combined dataset of original and pseudo-labeled data.
    """

    _, val_ds, _, _, _, _ = load_data(config, model_name)  # Get val_ds

    if len(combined_df) == 0:
        print("No pseudo-labeled data available. Skipping retraining.")
        return model, None

    if config['pseudo_labeling'].get('use_all_samples', True):
        train_ds = create_tensorset(combined_df, 
                                    config['models'][model_name]['img_size'],
                                    config['data']['batch_size'],
                                    config['data'].get('augmentation_magnitude', 0.3),
                                    ds_name="train",
                                    model_name=model_name,
                                    config=config)  # Pass the config here
    else:
        num_samples = min(config['pseudo_labeling'].get('num_samples_per_epoch', len(combined_df)), len(combined_df))
        initial_sample = combined_df.sample(n=num_samples, replace=False)
        train_ds = create_tensorset(initial_sample, 
                                    config['models'][model_name]['img_size'],
                                    config['data']['batch_size'],
                                    config['data'].get('augmentation_magnitude', 0.3),
                                    ds_name="train",
                                    model_name=model_name,
                                    config=config)  # Pass the config here

    history = train_model(
        model,
        train_ds,
        val_ds,
        config,
        learning_rate=config['training']['learning_rate'],
        epochs=config['pseudo_labeling']['retraining_epochs'],
        image_size=config['models'][model_name]['img_size'],
        model_name=model_name,
        is_fine_tuning=True,
        combined_df=combined_df  # Pass the combined DataFrame
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

    model = unfreeze_model(model, config['models'][model_name]['unfreeze_layers'])
    
    # Generate pseudo-labels
    pseudo_labeled_data = generate_pseudo_labels(model, config['pseudo_labeling']['unlabeled_data_dir'], config, model_name)
    
    print(f"Pseudo-labeled data distribution:")
    print(pseudo_labeled_data['Label'].value_counts())
    
    if pseudo_labeled_data.empty:
        print("No pseudo-labeled data generated. Check the confidence threshold and model predictions.")
        return
    
    pseudo_labeled_data.to_csv(os.path.join(config['data']['output_dir'], f'pseudo_labeled_data_{model_name}.csv'), index=False)
    
    # Load original labeled data
    _, _, _, _, _, train_df = load_data(config, model_name)  # Get train_df
    
    # Combine datasets
    combined_df = combine_datasets(train_df, pseudo_labeled_data, config)
    
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
