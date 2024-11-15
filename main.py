import os
os.environ["KERAS_BACKEND"] = "jax"

import yaml
from data_loader import load_data, prepare_cross_validation_data
from models import create_model, unfreeze_model
from train import train_model, train_fold
from evaluate import evaluate_model
import matplotlib.pyplot as plt
import datetime
import numpy as np
import jax

def plot_training_history(history, model_name, output_dir):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title(f'{model_name} - Training and Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_training_history.png'))
    plt.close()

def train_with_cross_validation(config, model_name):
    """
    Train model using cross-validation with JAX backend
    """
    fold_datasets, test_dataset = prepare_cross_validation_data(
        config['data']['train_dir'],
        config,
        model_name,
        n_splits=5
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
        
        # Evaluate fold
        fold_eval = evaluate_model(
            model, 
            config['data']['test_dir'],
            config['data']['class_names'],
            batch_size=config['data']['batch_size'],
            img_size=config['models'][model_name]['img_size'],
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

def plot_fold_histories(fold_results, model_name, output_dir):
    """
    Plot training histories for all folds
    """
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    for i, result in enumerate(fold_results):
        if 'frozen' in result['history']:
            plt.plot(result['history']['frozen']['val_loss'], 
                    label=f'Fold {i+1} Frozen', linestyle='--')
        if 'unfrozen' in result['history']:
            plt.plot(result['history']['unfrozen']['val_loss'], 
                    label=f'Fold {i+1} Unfrozen', linestyle='-')
    plt.title(f'{model_name} - Validation Loss Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    for i, result in enumerate(fold_results):
        if 'frozen' in result['history']:
            plt.plot(result['history']['frozen']['val_accuracy'], 
                    label=f'Fold {i+1} Frozen', linestyle='--')
        if 'unfrozen' in result['history']:
            plt.plot(result['history']['unfrozen']['val_accuracy'], 
                    label=f'Fold {i+1} Unfrozen', linestyle='-')
    plt.title(f'{model_name} - Validation Accuracy Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_fold_histories.png'))
    plt.close()

def main():
    with open('config_SMOTE.yaml', 'r') as f:
        config = yaml.safe_load(f)

    output_dir = config['data']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for model_name in config['models']:
        results[model_name] = train_with_cross_validation(config, model_name)

    # Print and save detailed summary
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
        
        for i, class_name in enumerate(config['data']['class_names']):
            summary += f"    {class_name}:\n"
            summary += f"      Precision: {avg_precision[i]:.4f}\n"
            summary += f"      Recall: {avg_recall[i]:.4f}\n"
            summary += f"      F1: {avg_f1[i]:.4f}\n"
    
    print(summary)
    
    # Save summary
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(output_dir, f'cv_results_summary_{timestamp}.txt'), 'w') as f:
        f.write(summary)

if __name__ == "__main__":
    main()