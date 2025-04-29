import os
os.environ["KERAS_BACKEND"] = "jax"

import yaml
import datetime
import numpy as np
import jax

# Local imports
from data_loader import prepare_cross_validation_data
from models import create_model
from train import train_fold
from evaluate import evaluate_model

def train_with_cross_validation(config, model_name):
    """
    Trains a model using k-fold cross-validation.

    Args:
        config (dict): Configuration dictionary.
        model_name (str): Name of the model architecture to use.

    Returns:
        list: A list of dictionaries, one for each fold, containing 
              training history and evaluation metrics.
    """
    fold_datasets = prepare_cross_validation_data(
        config['data']['train_dir'],
        config,
        model_name
    )
    
    fold_results = []
    for fold_idx, (train_ds, val_ds) in enumerate(fold_datasets):
        # Create new model instance for this fold
        model = create_model(model_name, config=config)
        
        # Train both phases
        history, model = train_fold(
            model, 
            train_ds, 
            val_ds, 
            config, 
            model_name,
            fold_idx
        )
        
        # Evaluate fold using validation data instead of test data
        fold_eval = evaluate_model(
            model, 
            val_ds,  # Changed from test_dir to val_ds
            config['data']['class_names'],
            output_path=os.path.join(config['data']['output_dir'], f'fold_{fold_idx + 1}')
        )
        
        fold_results.append({
            'history': history,
            'metrics': fold_eval
        })
        
        # Clean up to free memory
        del model
        jax.clear_caches()
    
    return fold_results

def generate_summary(results, class_names):
    """
    Generates a formatted string summarizing cross-validation results.

    Args:
        results (dict): Dictionary where keys are model names and values 
                        are lists of fold results from train_with_cross_validation.
        class_names (list): List of class names.

    Returns:
        str: A formatted summary string.
    """
    summary = "\nDetailed Summary of Cross-Validation Results:\n"
    for model_name, model_results in results.items():
        summary += f"\n{model_name}:\n"
        
        # Calculate averages across folds
        avg_accuracy = np.mean([fold['metrics']['accuracy'] for fold in model_results])
        avg_macro_f1 = np.mean([fold['metrics']['macro_f1'] for fold in model_results])
        avg_weighted_f1 = np.mean([fold['metrics']['weighted_f1'] for fold in model_results])
        
        # Calculate per-class metrics
        avg_precision = np.mean([fold['metrics']['precision_per_class'] for fold in model_results], axis=0)
        avg_recall = np.mean([fold['metrics']['recall_per_class'] for fold in model_results], axis=0)
        avg_f1 = np.mean([fold['metrics']['f1_per_class'] for fold in model_results], axis=0)
        
        # Write summary
        summary += f"  Average Accuracy: {avg_accuracy:.4f}\n"
        summary += f"  Average Macro F1: {avg_macro_f1:.4f}\n"
        summary += f"  Average Weighted F1: {avg_weighted_f1:.4f}\n"
        summary += "\n  Per-class metrics:\n"
        
        for i, class_name in enumerate(class_names):
            summary += f"    {class_name}:\n"
            summary += f"      Precision: {avg_precision[i]:.4f}\n"
            summary += f"      Recall: {avg_recall[i]:.4f}\n"
            summary += f"      F1: {avg_f1[i]:.4f}\n"
    return summary

def main():
    """Loads config, runs cross-validation for each model, prints and saves summary."""
    # Load configuration
    with open('config.yaml', 'r') as f: # Use relative path
        config = yaml.safe_load(f)

    output_dir = config['data']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Run training and evaluation for each model
    results = {}
    for model_name in config['models']:
        print(f"--- Training and Evaluating {model_name} ---")
        results[model_name] = train_with_cross_validation(config, model_name)
        print(f"--- Completed {model_name} ---")

    # Generate and save summary
    summary = generate_summary(results, config['data']['class_names'])
    print(summary)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f'cv_results_summary_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()